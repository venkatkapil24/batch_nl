import torch
from torch.nn.utils.rnn import pad_sequence

import warnings

class NeighbourList:
    """
    Batched neighbour-list builder using PyTorch.

    This class constructs neighbour lists for a set of periodic atomic
    configurations using a common cutoff radius. All configurations are
    treated as a single batch: they are padded to a common shape and
    processed together on the specified device, enabling fully vectorised
    neighbour-list construction across the batch.
    """

    def __init__(self,
             list_of_positions: list,
             list_of_cells,
             cutoff: float,
             device: str | torch.device | None = None):
        """
        Initialize a batched neighbour-list calculator for a single batch
        of configurations.

        Parameters
        ----------
        list_of_positions : list of array-like
            A list where each element is an (N_i, 3) array containing the
            Cartesian positions for configuration i in the batch.

        list_of_cells : list of array-like
            A list where each element is a (3, 3) cell matrix corresponding
            to configuration i. Must have the same length as `list_of_positions`.

        cutoff : float
            Cutoff radius used for neighbour detection. Must be positive.

        device : str or torch.device, optional
            Device on which all batched tensors will be allocated
            ("cpu", "cuda", or a torch.device instance). If None, CUDA is
            used when available, otherwise CPU.
        """

        if len(list_of_positions) != len(list_of_cells):
            raise ValueError(f"length of position and cell lists should be the same, got len(pos_list) = {len(list_of_positions)} and len(cell_list) = {len(list_of_cells)}")
        
        self.positions_list = list_of_positions
        self.cell_list = list_of_cells
        self.num_configs = len(self.positions_list)

        # checks cutoff
        if isinstance(cutoff, int):
            warnings.warn("Converting cutoff from int to float.", stacklevel=2)
        try:
            cutoff = float(cutoff)
        except Exception:
            raise TypeError(f"cutoff must be convertible to float, got {cutoff}.")
        
        if cutoff <= 0.0:
            raise ValueError(f"cutoff must be positive, got {cutoff}.")

        self.cutoff = cutoff

        self.float_dtype = torch.float32
        self.int_dtype = torch.int64

        # checks device

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        elif not isinstance(device,  (str, torch.device)):
            raise TypeError(f"device should be a string or torch.device, got {type(device).__name__}.")
        else:
            self.device = torch.device(device)

        # an internal tolerance for check self images after wrapping
        self._tolerance = 1e-6

        # compoled neighbourlist function
        self._nlist_ON2_compiled = torch.compile(self._nlist_ON2)

    def load_data(self):
        """
        Convert input positions and cells into padded batched tensors.

        This populates `batch_positions_tensor`, `batch_mask_tensor`,
        and `batch_cell_tensor`, and moves them to the configured device.

        Must be called before `calculate_neighbourlist`.
        """
        
        self.batch_positions_tensor = pad_sequence(
            [
                torch.tensor(t, dtype=self.float_dtype)
                for t in self.positions_list
            ],
            batch_first=True,
            padding_value=float("nan"),
        )

        self.batch_mask_tensor = (self.batch_positions_tensor == self.batch_positions_tensor).any(dim=-1)

        self.batch_positions_tensor = torch.nan_to_num(self.batch_positions_tensor, nan=0).to(self.device)

        self.batch_cell_tensor = torch.tensor(self.cell_list, dtype=self.float_dtype).to(self.device)

        # move to GPU
        self.batch_positions_tensor = self.batch_positions_tensor.to(self.device)
        self.batch_cell_tensor      = self.batch_cell_tensor.to(self.device)
        self.batch_mask_tensor      = self.batch_mask_tensor.to(self.device)


    def calculate_neighbourlist(self, use_torch_compile: bool = True):
        """
        Compute the batched neighbour list for all configurations.

        Notes
        -----
        `load_data()` must be called before invoking this method.

        Parameters
        ----------
        use_torch_compile : bool, optional
            If True, use the torch.compile-optimised backend; otherwise use
            the plain PyTorch implementation.

        Returns
        -------
        r_edges : torch.Tensor
            Tensor of shape (2, E) with flattened source and neighbour
            atom indices.
        r_integer_lattice_shifts : torch.Tensor
            Tensor of shape (E, 3) with integer lattice shift vectors.
        r_cartesian_lattice_shifts : torch.Tensor
            Tensor of shape (E, 3) with Cartesian lattice shift vectors.
        r_distances : torch.Tensor
            Tensor of shape (E,) with interatomic distances.
        """

        if use_torch_compile:
            neighbourlist_fn = self._nlist_ON2_compiled
        else:
            neighbourlist_fn = self._nlist_ON2

        # the tensors are aready in GPU at this point
        # use compiled function if user wants

        (
            distance_matrix,                            # (B, L, N, N)
            criterion,                                  # (B, L, N, N)
            batch_lattice_shifts_tensor,                # (L, 3)
            batch_cartesian_lattice_shifts_tensor,      # (B, L, 3)
        ) = neighbourlist_fn(
            self.batch_positions_tensor,
            self.batch_cell_tensor,
            self.batch_mask_tensor,
            self.cutoff,
            self._tolerance,
        )

        config_idx, lattice_shift_idx, atom_idx, neighbour_idx = torch.nonzero(
                criterion, as_tuple=True
            )
        
        # Need to offset edge indices with config_idx
        lengths = self.batch_mask_tensor.sum(dim=-1, dtype=self.int_dtype)
        offsets = torch.cumsum(lengths, dim=0) - lengths
        r_edges = torch.stack([atom_idx + offsets[config_idx], neighbour_idx + offsets[config_idx]])

        r_integer_lattice_shifts = batch_lattice_shifts_tensor[lattice_shift_idx]
        r_cartesian_lattice_shifts = batch_cartesian_lattice_shifts_tensor[config_idx, lattice_shift_idx, :]
        r_distances = distance_matrix[config_idx, lattice_shift_idx, atom_idx, neighbour_idx]

        return (r_edges, r_integer_lattice_shifts, r_cartesian_lattice_shifts, r_distances)


    def _calculate_batch_lattice_shifts(self, batch_cells_tensor, cutoff):
        """
        Compute the lattice shift vectors and their Cartesian images.

        Parameters
        ----------
        batch_cells_tensor : torch.Tensor
            Tensor of shape (B, 3, 3) with cell matrices.
        cutoff : float
            Cutoff radius used to determine the required lattice shifts.

        Returns
        -------
        batch_lattice_shifts_tensor : torch.Tensor
            Tensor of shape (L, 3) with integer lattice shift vectors.
        batch_cartesian_lattice_shifts_tensor : torch.Tensor
            Tensor of shape (B, L, 3) with corresponding Cartesian shifts.
        """

        # estimate from cell-vector norms
        cell_lengths = torch.linalg.norm(batch_cells_tensor, dim=-1)
        n_from_lengths = torch.ceil(cutoff / torch.clamp(cell_lengths, 1e-8)).amax(dim=0)

        # estimate from coordinate extents
        extents = (
            batch_cells_tensor.max(dim=1).values
            - batch_cells_tensor.min(dim=1).values
        )
        n_from_extents = torch.ceil(cutoff / torch.clamp(extents, 1e-8)).amax(dim=0)

        # take the larger
        max_n = torch.maximum(n_from_lengths, n_from_extents).to(self.int_dtype)

        mesh = torch.meshgrid(
            torch.arange(-max_n[0], max_n[0] + 1, dtype=self.int_dtype,
                        device=batch_cells_tensor.device),
            torch.arange(-max_n[1], max_n[1] + 1, dtype=self.int_dtype,
                        device=batch_cells_tensor.device),
            torch.arange(-max_n[2], max_n[2] + 1, dtype=self.int_dtype,
                        device=batch_cells_tensor.device),
            indexing='ij'
        )

        mesh = torch.stack(mesh, dim=-1).reshape(-1, 3)

        batch_cartesian_lattice_shifts_tensor = torch.einsum(
            "li,bij->blj", mesh.to(batch_cells_tensor.dtype), batch_cells_tensor
        )
        return mesh, batch_cartesian_lattice_shifts_tensor

    def _nlist_ON2(
        self,
        batch_positions_tensor,
        batch_cells_tensor,
        batch_mask_tensor,
        cutoff,
        tolerance,
    ):
        """
        Full O(N^2) neighbour-list backend for a single batch.

        Parameters
        ----------
        batch_positions_tensor : torch.Tensor
            Tensor of shape (B, N, 3) with batched atomic positions.
        batch_cells_tensor : torch.Tensor
            Tensor of shape (B, 3, 3) with batched cell matrices.
        batch_mask_tensor : torch.Tensor
            Boolean tensor of shape (B, N) marking valid atoms.
        cutoff : float
            Cutoff radius for neighbour detection.
        tolerance : float
            Lower distance bound used to exclude self/near-self images.

        Returns
        -------
        distance_matrix : torch.Tensor
            Tensor of shape (B, L, N, N) with pairwise distances.
        criterion : torch.Tensor
            Boolean tensor of shape (B, L, N, N) marking neighbour pairs.
        batch_lattice_shifts_tensor : torch.Tensor
            Tensor of shape (L, 3) with integer lattice shift vectors.
        batch_cartesian_lattice_shifts_tensor : torch.Tensor
            Tensor of shape (B, L, 3) with Cartesian lattice shift vectors.
        """

        # (L, 3), (B, L, 3)
        batch_lattice_shifts_tensor, batch_cartesian_lattice_shifts_tensor = \
            self._calculate_batch_lattice_shifts(batch_cells_tensor, cutoff=cutoff)

        # (B, L, 1, 3) + (B, 1, N, 3) -> (B, L, N, 3)
        batch_shifted_positions_tensor = (
            batch_cartesian_lattice_shifts_tensor.unsqueeze(-2)
            + batch_positions_tensor.unsqueeze(1)
        )

        # (B, 1, 1, N, 3) - (B, L, N, 1, 3) -> (B, L, N, N)
        diff = (
            batch_positions_tensor.unsqueeze(1).unsqueeze(3)
            - batch_shifted_positions_tensor.unsqueeze(2)
        )
        distance_matrix = torch.sqrt((diff ** 2).sum(dim=-1))

        distance_matrix_criterion = (
            (distance_matrix < cutoff) & (distance_matrix >= tolerance)
        )

        # get the appropriate mask for atom pair connectivity
        default_mask = (
            batch_mask_tensor.unsqueeze(-2) & batch_mask_tensor.unsqueeze(-1)
        )  # (B, N, N)
        default_mask = default_mask.unsqueeze(1)  # (B, 1, N, N)

        criterion = distance_matrix_criterion & default_mask  # (B, L, N, N)

        return (
            distance_matrix,                       # (B, L, N, N)
            criterion,                             # (B, L, N, N)
            batch_lattice_shifts_tensor,           # (L, 3) integer shifts
            batch_cartesian_lattice_shifts_tensor  # (B, L, 3) cartesian shifts
        )

    def get_matscipy_output_from_batch_output(
        self,
        r_edges: torch.Tensor,                    # (2, E) 
        r_integer_lattice_shifts: torch.Tensor,   # (E, 3)
        r_cartesian_lattice_shifts: torch.Tensor, # (E, 3)
        r_distances: torch.Tensor,                # (E,)
    ):
        """
        Convert flattened batched neighbour-list output to per-configuration
        matscipy-style lists.

        Parameters
        ----------
        r_edges : torch.Tensor
            Tensor of shape (2, E) with flattened source and neighbour
            atom indices.
        r_integer_lattice_shifts : torch.Tensor
            Tensor of shape (E, 3) with integer lattice shift vectors.
        r_cartesian_lattice_shifts : torch.Tensor
            Tensor of shape (E, 3) with Cartesian lattice shift vectors.
        r_distances : torch.Tensor
            Tensor of shape (E,) with interatomic distances.

        Returns
        -------
        atom_index_list : list of torch.Tensor
            Per-configuration tensors of source atom indices.
        neighbor_index_list : list of torch.Tensor
            Per-configuration tensors of neighbour atom indices.
        int_shift_list : list of torch.Tensor
            Per-configuration tensors of integer lattice shifts.
        cart_shift_list : list of torch.Tensor
            Per-configuration tensors of Cartesian lattice shifts.
        distance_list : list of torch.Tensor
            Per-configuration tensors of interatomic distances.
        """
        
        # atoms per config: (B,)
        lengths = self.batch_mask_tensor.sum(dim=-1, dtype=torch.long)
        # offsets[i] = total atoms in all previous configs
        offsets = torch.cumsum(lengths, dim=0) - lengths  # (B,)

        n_configs = lengths.size(0)

        atom_index_list = []
        neighbor_index_list = []
        int_shift_list = []
        cart_shift_list = []
        distance_list = []

        # loop over configs only (not over atoms/edges)
        for cfg in range(n_configs):
            start = offsets[cfg]
            stop = start + lengths[cfg]

            # which edges belong to this config? (source atom in this range)
            mask = (r_edges[0] >= start) & (r_edges[0] < stop)

            # global → local atom indices
            i_global = r_edges[0, mask]
            j_global = r_edges[1, mask]
            i_local = i_global - start
            j_local = j_global - start

            atom_index_list.append(i_local.to('cpu'))
            neighbor_index_list.append(j_local.to('cpu'))
            int_shift_list.append(r_integer_lattice_shifts[mask].to('cpu'))
            cart_shift_list.append(r_cartesian_lattice_shifts[mask].to('cpu'))
            distance_list.append(r_distances[mask].to('cpu'))

        return (
            atom_index_list,
            neighbor_index_list,
            int_shift_list,
            cart_shift_list,
            distance_list,
        )