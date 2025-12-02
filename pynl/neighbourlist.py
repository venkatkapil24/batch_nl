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
             list_of_configurations: list,
             radius: float,
             batch_size: int,
             device: str | torch.device | None = None):
        """
        Parameters
        ----------
        list_of_configurations : list of Atoms
            List of ASE-like Atoms objects with `.positions` and `.cell`.

        radius : float
            Cutoff radius.

        batch_size : int
            Number of configurations processed together per batch.

        device : str or torch.device
            Compute device ("cpu", "cuda", or torch.device(...)).
        """

        self.list_of_configurations = list_of_configurations
        self.num_configs = len(self.list_of_configurations)

        # checks radius
        if isinstance(radius, int):
            warnings.warn("Converting radius from int to float.", stacklevel=2)
        try:
            radius = float(radius)
        except Exception:
            raise TypeError(f"radius must be convertible to float, got {radius!r}.")
        
        if radius <= 0.0:
            raise ValueError(f"radius must be positive, got {radius}.")

        self.radius = radius

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
       
       self.cell_list = [torch.tensor(atoms.cell.array, dtype=self.float_dtype) for atoms in self.list_of_configurations]
       self.positions_list = [torch.tensor(atoms.positions, dtype=self.float_dtype) for atoms in self.list_of_configurations]
       
       self._batch_and_mask_positions_and_cells()

    def _batch_and_mask_positions_and_cells(self):
        """
        Build padded batched tensors for positions and cells, plus a mask.

        For each batch of configurations, this method constructs:
        - `batch_positions_tensor` with shape (batch_size, n_max, 3), where
          `n_max` is the maximum number of atoms in any configuration in that
          batch; entries beyond the true atom count are zero-padded.
        - `batch_cells_tensor` with shape (batch_size, 3, 3), containing one
          cell matrix per configuration in the batch.
        - `batch_masks_tensor` with shape (batch_size, n_max), where entries
          are True for valid atoms and False for padding.

        The tensors for all batches are stored in the lists:
        `batch_positions_tensor_list`, `batch_cells_tensor_list`,
        and `batch_masks_tensor_list`.

        Notes
        -----
        This is an internal helper and assumes that `positions_list` and
        `cell_list` have already been populated by `load_data`.
        """

        self.batch_positions_tensor_list = []
        self.batch_cells_tensor_list = []
        self.batch_masks_tensor_list = []

        for batch_id in range(self.num_batches):

            structure_id_min = batch_id * self.batch_size
            structure_id_max = min((batch_id + 1) * self.batch_size, self.num_configs)

            position_size_max = max(len(p) for p in self.positions_list[structure_id_min : structure_id_max])

            batch_positions_tensor = torch.zeros(self.batch_size, position_size_max, 3, dtype=self.float_dtype)
            batch_cells_tensor = torch.eye(3, dtype=self.float_dtype).unsqueeze(0).repeat(self.batch_size, 1, 1)
            batch_masks_tensor = torch.zeros(self.batch_size, position_size_max, dtype=torch.bool)
            del position_size_max

            for batch_structure_id in range(structure_id_max - structure_id_min):

                pos = self.positions_list[batch_structure_id + structure_id_min]
                cell = self.cell_list[batch_structure_id + structure_id_min]
                batch_positions_tensor[batch_structure_id, 0 : len(pos)] = pos
                batch_cells_tensor[batch_structure_id] = cell
                batch_masks_tensor[batch_structure_id, 0 : len(pos)] = 1

            self.batch_positions_tensor_list.append(batch_positions_tensor)
            self.batch_cells_tensor_list.append(batch_cells_tensor)
            self.batch_masks_tensor_list.append(batch_masks_tensor)

    def calculate_neighbourlist(self, use_torch_compile=True):
        """
        Compute neighbour lists for all configurations in all batches.

        Parameters
        ----------
        use_torch_compile : bool, optional
            If True (default), use the `torch.compile`-optimised O(N^2)
            backend; if False, use the uncompiled reference implementation.

        Returns
        -------
        list of list
            Outer list has length `num_batches`. Each element corresponds to
            one batch and is itself a list of length `batch_size`, where each
            entry is `[atom_idx, neighbour_idx, S, d]` for a single
            configuration:
                atom_idx : 1D LongTensor
                    Indices of central atoms.
                neighbour_idx : 1D LongTensor
                    Indices of neighbour atoms.
                S : LongTensor of shape (n_pairs, 3)
                    Integer lattice shift vectors for each pair.
                d : 1D Tensor
                    Distances corresponding to each pair.

        Notes
        -----
        Assumes `load_data()` has been called so that the batched position,
        cell and mask tensors are available in
        `batch_positions_tensor_list`, `batch_cells_tensor_list`, and
        `batch_masks_tensor_list`.
        """

        r = []

        if use_torch_compile:
            neighbourlist_fn = self._nlist_ON2_compiled
        else:
            neighbourlist_fn = self._nlist_ON2

        for batch_id in range(self.num_batches):

            batch_positions_tensor = self.batch_positions_tensor_list[batch_id].to(self.device)
            batch_cells_tensor = self.batch_cells_tensor_list[batch_id].to(self.device)
            batch_mask_tensor = self.batch_masks_tensor_list[batch_id].to(self.device)
            batch_lattice_shifts_tensor, batch_cartesian_lattice_shifts_tensor = self._calculate_batch_lattice_shifts(batch_cells_tensor,)

            distance_matrix, criterion = neighbourlist_fn(batch_positions_tensor, batch_cells_tensor, batch_mask_tensor, batch_cartesian_lattice_shifts_tensor, self.radius, self.tolerance)

            b_r = []

            for i in range(self.batch_size):
              
                lattice_shift_idx, atom_idx, neighbour_idx = torch.nonzero(criterion[i], as_tuple=True)

                S = batch_lattice_shifts_tensor[lattice_shift_idx]

                D = batch_positions_tensor[i, neighbour_idx] - batch_positions_tensor[i, atom_idx] + batch_cartesian_lattice_shifts_tensor[i, lattice_shift_idx]

                d = distance_matrix[i, lattice_shift_idx, atom_idx, neighbour_idx]

                b_r.append([atom_idx, neighbour_idx, S, D, d])

            r.append(b_r)

        return r
    
    def _calculate_batch_lattice_shifts(self, batch_cells_tensor):
        """
        Compute a common set of lattice shift vectors for a batch of cells.

        For a given cutoff radius and a batch of 3×3 cell matrices, this
        estimates how many periodic images along each lattice direction must
        be considered so that all neighbours within `self.radius` are captured
        for every cell in the batch. A single, conservative set of shifts is
        returned and reused for the whole batch.

        Parameters
        ----------
        batch_cells_tensor : torch.Tensor, shape (n_cells, 3, 3)
            Cell matrices for the configurations in the current batch.

        Returns
        -------
        shifts : torch.Tensor, shape (n_shifts, 3)
            Integer lattice shift vectors (in lattice coordinates) that
            should be applied when building neighbour lists.
        """

        # estimate from cell-vector norms
        cell_lengths = torch.linalg.norm(batch_cells_tensor, dim=-1)
        n_from_lengths = torch.ceil(self.radius / torch.clamp(cell_lengths, 1e-8)).amax(dim=0)

        # estimate from coordinate extents
        extents = (
            batch_cells_tensor.max(dim=1).values
            - batch_cells_tensor.min(dim=1).values
        )
        n_from_extents = torch.ceil(self.radius / torch.clamp(extents, 1e-8)).amax(dim=0)

        # take the larger
        max_n = torch.maximum(n_from_lengths, n_from_extents).to(self.int_dtype)

        mesh = torch.meshgrid(
            torch.arange(-max_n[0], max_n[0] + 1, dtype=self.int_dtype, device=batch_cells_tensor.device),
            torch.arange(-max_n[1], max_n[1] + 1, dtype=self.int_dtype, device=batch_cells_tensor.device),
            torch.arange(-max_n[2], max_n[2] + 1, dtype=self.int_dtype, device=batch_cells_tensor.device),
            indexing='ij'
        )

        mesh = torch.stack(mesh, dim=-1).reshape(-1, 3)

        batch_cartesian_lattice_shifts_tensor = torch.einsum("li,bij->blj", mesh.to(batch_cells_tensor.dtype), batch_cells_tensor)
        return mesh, batch_cartesian_lattice_shifts_tensor

    def _nlist_ON2(self, batch_positions_tensor, batch_cells_tensor, batch_mask_tensor, batch_cartesian_lattice_shifts_tensor, radius, tolerance):
        """
        O(N²) neighbour-list backend operating on a batched distance matrix.

        Parameters
        ----------
        batch_positions_tensor : torch.Tensor, shape (B, N, 3)
            Cartesian positions for a batch of configurations, padded to the
            maximum number of atoms `N` in the batch.
        batch_cells_tensor : torch.Tensor, shape (B, 3, 3)
            Cell matrices for each configuration in the batch.
        batch_mask_tensor : torch.Tensor, shape (B, N)
            Boolean mask indicating which entries in `batch_positions_tensor`
            correspond to real atoms (True) versus padding (False).
        lattice_shifts : torch.Tensor, shape (L, 3)
            Integer lattice shift vectors to apply (in lattice coordinates).
        radius : float
            Cutoff radius; pairs with distance < `radius` are kept.
        tolerance : float
            Lower distance bound used to filter out self-pairs and very
            close numerical artefacts; pairs with distance < `tolerance`
            are discarded.

        Returns
        -------
        distance_matrix : torch.Tensor, shape (B, L, N, N)
            Pairwise distances between atoms and their periodic images for
            each batch (B), lattice shift (L) and atom pair (N × N).
        criterion : torch.Tensor, shape (B, L, N, N)
            Boolean mask indicating which pairs satisfy the cutoff and masking
            criteria, i.e. within `[tolerance, radius)` and both atoms valid
            according to `batch_mask_tensor`.
        """

        # (b, lc, 1, 3) + (b, 1, n, 3) -> b, lc, n, 3
        batch_shifted_positions_tensor = batch_cartesian_lattice_shifts_tensor.unsqueeze(-2) + batch_positions_tensor.unsqueeze(1)

        # b, 1, 1, n, 3 - b, lc, n, 1, 3 ->  b, lc, n, n
        distance_matrix = torch.sqrt(((batch_positions_tensor.unsqueeze(1).unsqueeze(3) - batch_shifted_positions_tensor.unsqueeze(2))**2).sum(dim=-1))

        distance_matrix_criterion = (distance_matrix < radius) & (distance_matrix >= tolerance)

        # get the appropriate mask for atom pair connectivity
        default_mask = batch_mask_tensor.unsqueeze(-2) & batch_mask_tensor.unsqueeze(-1)
        
        # adds dim for lattice_shifts
        default_mask = default_mask.unsqueeze(1)

        criterion = torch.logical_and(distance_matrix_criterion, default_mask)

        return distance_matrix, criterion
    
