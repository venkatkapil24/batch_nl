import torch
from torch.nn.utils.rnn import pad_sequence

import warnings

class NeighbourList:
    """
    Batched neighbour-list builder using PyTorch.

    Given a list of periodic configurations, this class builds neighbour
    lists for all structures using a common cutoff radius. Configurations 
    are grouped into fixed-size batches and padded, so the heavy work
    can be done with vectorised tensor operations on a chosen device.
    """

    def __init__(self,
             list_of_positions: list,
             list_of_cells,
             cutoff: float,
             device: str | torch.device | None = None):
        """
        Parameters
        ----------
        list_of_configurations : list of Atoms
            List of ASE-like Atoms objects with `.positions` and `.cell`.

        cutoff : float
            Cutoff radius.

        device : str or torch.device
            Compute device ("cpu", "cuda", or torch.device(...)).
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
        Convert the input configurations to tensor form and prepare batches.

        This method:
        - converts each configuration's cell and positions to PyTorch tensors
          with the class float dtype, and stores them in `cell_list` and
          `positions_list`;
        - calls `_batch_and_mask_positions_and_cells` to build the padded
          batched tensors and masks used internally by the neighbour-list
          routines.

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
        Compute a common set of lattice shift vectors for a batch of cells.
        ...
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
        Full O(N^2) neighbour-list backend for a batch:
        - computes lattice shifts
        - computes distances
        - computes criterion mask
        Returns distances + both integer and cartesian lattice shifts.
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
        r_edges: torch.Tensor,                    # (2, E) global atom indices
        r_integer_lattice_shifts: torch.Tensor,   # (E, 3)
        r_cartesian_lattice_shifts: torch.Tensor, # (E, 3)
        r_distances: torch.Tensor,                # (E,)
    ):
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