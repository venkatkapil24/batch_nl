import torch
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
             batch_size: int,
             device: str | torch.device | None = None):
        """
        Parameters
        ----------
        list_of_configurations : list of Atoms
            List of ASE-like Atoms objects with `.positions` and `.cell`.

        cutoff : float
            Cutoff radius.

        batch_size : int
            Number of configurations processed together per batch.

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

        # checks batch_size 
        
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive int, got {batch_size}")

        if batch_size > self.num_configs:
            raise ValueError(
                "batch_size must be smaller than or equal to the total number of configurations."
            )

        self.batch_size = batch_size
        self.num_batches = (self.num_configs + self.batch_size - 1) // self.batch_size

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
        self.tolerance = 1e-6

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

        self.batch_positions_tensor_list = []
        self.batch_cells_tensor_list = []
        self.batch_masks_tensor_list = []

        device = self.device  # <- you said to use this

        for batch_id in range(self.num_batches):

            structure_id_min = batch_id * self.batch_size
            structure_id_max = min((batch_id + 1) * self.batch_size, self.num_configs)
            n_in_batch = structure_id_max - structure_id_min  # number of real structures

            # Slice once
            pos_list = [
                torch.as_tensor(self.positions_list[i], dtype=self.float_dtype, device=device)
                for i in range(structure_id_min, structure_id_max)
            ]
            cell_list = [
                torch.as_tensor(self.cell_list[i], dtype=self.float_dtype, device=device)
                for i in range(structure_id_min, structure_id_max)
            ]

            # Vectorised lengths / max
            lengths = torch.tensor(
                [p.shape[0] for p in pos_list],
                device=device,
                dtype=torch.long,
            )
            position_size_max = int(lengths.max().item())

            # Allocate batch tensors on target device
            batch_positions_tensor = torch.zeros(
                self.batch_size, position_size_max, 3,
                dtype=self.float_dtype, device=device
            )
            batch_cells_tensor = torch.eye(
                3, dtype=self.float_dtype, device=device
            ).unsqueeze(0).repeat(self.batch_size, 1, 1)
            batch_masks_tensor = torch.zeros(
                self.batch_size, position_size_max,
                dtype=torch.bool, device=device
            )

            # Pad positions to common length and insert into batch
            padded_pos = torch.nn.utils.rnn.pad_sequence(
                pos_list, batch_first=True  # (n_in_batch, position_size_max, 3)
            )
            batch_positions_tensor[:n_in_batch] = padded_pos

            # Stack cells
            batch_cells_tensor[:n_in_batch] = torch.stack(cell_list, dim=0)

            # Build mask: True for indices < length, False otherwise
            arange_n = torch.arange(position_size_max, device=device)  # (position_size_max,)
            batch_masks_tensor[:n_in_batch] = arange_n.unsqueeze(0) < lengths.unsqueeze(1)

            self.batch_positions_tensor_list.append(batch_positions_tensor)
            self.batch_cells_tensor_list.append(batch_cells_tensor)
            self.batch_masks_tensor_list.append(batch_masks_tensor)

    def calculate_neighbourlist(self, use_torch_compile: bool = True):

        r = []

        if use_torch_compile:
            neighbourlist_fn = self._nlist_ON2_compiled
        else:
            neighbourlist_fn = self._nlist_ON2

        for batch_id in range(self.num_batches):

            batch_positions_tensor = self.batch_positions_tensor_list[batch_id].to(self.device)
            batch_cells_tensor     = self.batch_cells_tensor_list[batch_id].to(self.device)
            batch_mask_tensor      = self.batch_masks_tensor_list[batch_id].to(self.device)

            (
                distance_matrix,
                criterion,
                batch_lattice_shifts_tensor,
                batch_cartesian_lattice_shifts_tensor,
            ) = neighbourlist_fn(
                batch_positions_tensor,
                batch_cells_tensor,
                batch_mask_tensor,
                self.cutoff,
                self.tolerance,
            )

            b_r = self._unpack_batch_neighbourlist(
                batch_positions_tensor,
                distance_matrix,
                criterion,
                batch_lattice_shifts_tensor,
                batch_cartesian_lattice_shifts_tensor,
            )

            r.append(b_r)

        return r

    def _unpack_batch_neighbourlist(
        self,
        batch_positions_tensor: torch.Tensor,              # (B, N, 3)
        distance_matrix: torch.Tensor,                     # (B, L, N, N)
        criterion: torch.Tensor,                           # (B, L, N, N)
        batch_lattice_shifts_tensor: torch.Tensor,         # (L, 3) integer shifts
        batch_cartesian_lattice_shifts_tensor: torch.Tensor,  # (B, L, 3)
    ) -> list:
        """
        Convert dense (distance_matrix, criterion, shifts) for a batch into
        a Python list of [atom_idx, neighbour_idx, S, D, d] per configuration.

        Returns
        -------
        b_r : list
            Length B. Each element is a list for one configuration:
            [atom_idx, neighbour_idx, S, D, d]
        """
        B = distance_matrix.shape[0]
        b_r = []

        for i in range(B):
            # Find all (lattice_shift, atom_i, atom_j) pairs that satisfy the criterion
            lattice_shift_idx, atom_idx, neighbour_idx = torch.nonzero(
                criterion[i], as_tuple=True
            )

            # Integer lattice shifts: (n_pairs, 3)
            S = batch_lattice_shifts_tensor[lattice_shift_idx]

            # Cartesian distance vectors: r_j + shift - r_i
            D = (
                batch_positions_tensor[i, neighbour_idx]
                - batch_positions_tensor[i, atom_idx]
                + batch_cartesian_lattice_shifts_tensor[i, lattice_shift_idx]
            )

            # Scalar distances
            d = distance_matrix[i, lattice_shift_idx, atom_idx, neighbour_idx]

            b_r.append([atom_idx, neighbour_idx, S, D, d])

        return b_r


    def _calculate_batch_lattice_shifts(self, batch_cells_tensor, cutoff=None):
        """
        Compute a common set of lattice shift vectors for a batch of cells.
        ...
        """
        if cutoff is None:
            cutoff = self.cutoff

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
            distance_matrix,
            criterion,
            batch_lattice_shifts_tensor,           # (L, 3) integer shifts
            batch_cartesian_lattice_shifts_tensor  # (B, L, 3) cartesian shifts
        )

    
