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

    def __init__(
        self,
        list_of_positions: list,
        list_of_cells: list,
        cutoff: float | torch.Tensor,
        float_dtype: torch.dtype = torch.float32,
        device: str | torch.device | None = None,
    ):
        """
        Initialize a batched neighbour-list calculator.

        Parameters
        ----------
        list_of_positions : list of array-like
            List of length n_configs; each entry is an (n_i, 3) array with
            Cartesian positions for configuration i with n_i atoms.
        list_of_cells : list of array-like
            List of length n_configs; each entry is a (3, 3) cell matrix
            corresponding to configuration i.
        cutoff : float
            Cutoff radius used for neighbour detection. Must be positive.
        device : str or torch.device, optional
            Device on which all batched tensors will be allocated
            ("cpu", "cuda", or torch.device(...)). If None, CUDA is used
            when available, otherwise CPU.
        """

        if len(list_of_positions) != len(list_of_cells):
            raise ValueError(
                "length of position and cell lists should be the same, "
                f"got len(pos_list) = {len(list_of_positions)} and "
                f"len(cell_list) = {len(list_of_cells)}"
            )

        self.positions_list = list_of_positions
        self.cell_list = list_of_cells
        self.num_configs = len(self.positions_list)

        # dtype check 
        if float_dtype not in [
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        ]:
            raise TypeError(
                f"float_dtype must be a floating torch.dtype, got {float_dtype}."
            )

        self.float_dtype = float_dtype
        self.int_dtype = torch.long

        # cutoff
        self.cutoff = torch.as_tensor(cutoff, dtype=self.float_dtype)

        if self.cutoff.ndim != 0:
            raise ValueError(
                f"cutoff must be a scalar, got tensor with shape {tuple(self.cutoff.shape)}."
            )

        if self.cutoff.item() <= 0.0:
            raise ValueError(
                f"cutoff must be positive, got {self.cutoff.item()}."
            )

        # device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif not isinstance(device, (str, torch.device)):
            raise TypeError(
                "device should be a string or torch.device, "
                f"got {type(device).__name__}."
            )
        else:
            self.device = torch.device(device)

        # internal tolerance to filter self / near-self images
        self._tolerance = 1e-6

        # compiled neighbour-list function
        self._nlist_ON2_compiled = torch.compile(self._nlist_ON2)

    def load_data(self):
        """
        Convert input positions and cells into padded batched tensors.

        This populates `batch_positions_tensor`, `batch_mask_tensor`,
        and `batch_cell_tensor`, and moves them to `self.device`.

        Must be called before `calculate_neighbourlist`.

        After this call:
        ----------------
        batch_positions_tensor : torch.Tensor
            (n_configs, n_max, 3) padded atomic positions.
        batch_mask_tensor : torch.Tensor
            (n_configs, n_max) boolean mask (True for valid atoms).
        batch_cell_tensor : torch.Tensor
            (n_configs, 3, 3) cell matrices.
        """

        # padded positions with NaN as padding
        self.batch_positions_tensor = pad_sequence(
            [
                torch.tensor(t, dtype=self.float_dtype)     # (n_i, 3)
                for t in self.positions_list
            ],
            batch_first=True,
            padding_value=float("nan"),
        )                                                   # (n_configs, n_max, 3)

        # mask: True where at least one coordinate is non-NaN
        self.batch_mask_tensor = (
            self.batch_positions_tensor == self.batch_positions_tensor
        ).any(dim=-1)                                       # (n_configs, n_max)

        # replace NaN with zeros
        self.batch_positions_tensor = torch.nan_to_num(
            self.batch_positions_tensor,
            nan=0.0,
        )                                                   # (n_configs, n_max, 3)

        # cells
        self.batch_cell_tensor = torch.tensor(
            self.cell_list,
            dtype=self.float_dtype,
        )                                                   # (n_configs, 3, 3)

        # move everything to target device
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
            Tensor of shape (2, n_edges) with flattened source and neighbour
            atom indices in the global (batched) indexing.
        r_integer_lattice_shifts : torch.Tensor
            Tensor of shape (n_edges, 3) with integer lattice shift vectors.
        r_cartesian_lattice_shifts : torch.Tensor
            Tensor of shape (n_edges, 3) with Cartesian lattice shift vectors.
        r_distances : torch.Tensor
            Tensor of shape (n_edges,) with interatomic distances.
        """

        if use_torch_compile:
            neighbourlist_fn = self._nlist_ON2_compiled
        else:
            neighbourlist_fn = self._nlist_ON2

        # compute full batched neighbour data
        (
            distance_matrix,                       # (n_configs, n_lattice_shifts, n_max, n_max)
            criterion,                             # (n_configs, n_lattice_shifts, n_max, n_max)
            batch_lattice_shifts_tensor,           # (n_lattice_shifts, 3)
            batch_cartesian_lattice_shifts_tensor, # (n_configs, n_lattice_shifts, 3)
        ) = neighbourlist_fn(
            self.batch_positions_tensor,           # (n_configs, n_max, 3)
            self.batch_cell_tensor,                # (n_configs, 3, 3)
            self.batch_mask_tensor,                # (n_configs, n_max)
            self.cutoff,
            self._tolerance,
        )

        # indices of all neighbour pairs
        config_idx, lattice_shift_idx, atom_idx, neighbour_idx = torch.nonzero(
            criterion,
            as_tuple=True,
        )                                          # each (n_edges,)

        # global atom indices via per-config offsets
        lengths = self.batch_mask_tensor.sum(dim=-1, dtype=self.int_dtype)  # (n_configs,)
        offsets = torch.cumsum(lengths, dim=0) - lengths                    # (n_configs,)

        r_edges = torch.stack(
            [
                atom_idx      + offsets[config_idx],    # (n_edges,)
                neighbour_idx + offsets[config_idx],    # (n_edges,)
            ],
            dim=0,
        )                                               # (2, n_edges)

        r_integer_lattice_shifts = batch_lattice_shifts_tensor[lattice_shift_idx]          # (n_edges, 3)
        r_cartesian_lattice_shifts = batch_cartesian_lattice_shifts_tensor[
            config_idx,
            lattice_shift_idx,
        ]                                                                                  # (n_edges, 3)

        r_distances = distance_matrix[
            config_idx,
            lattice_shift_idx,
            atom_idx,
            neighbour_idx,
        ]                                                                                  # (n_edges,)

        return (
            r_edges,
            r_integer_lattice_shifts,
            r_cartesian_lattice_shifts,
            r_distances,
        )


    def _calculate_batch_lattice_shifts(self, batch_cells_tensor, cutoff):
        """
        Compute Matscipy-style lattice shift vectors based on cell geometry.

        Parameters
        ----------
        batch_cells_tensor : torch.Tensor
            Tensor of shape (n_configs, 3, 3) with cell matrices
            (rows = lattice vectors for each configuration).
        cutoff : float
            Cutoff radius.

        Returns
        -------
        batch_lattice_shifts_tensor : torch.Tensor
            Tensor of shape (n_lattice_shifts, 3) containing all integer
            lattice shift vectors S = (s_x, s_y, s_z) needed to cover the
            cutoff for every configuration in the batch.
        """

        # normal to the lattice vectors
        N = torch.cross(
            batch_cells_tensor[:, [1, 2, 0], :],      # (n_configs, 3, 3)
            batch_cells_tensor[:, [2, 0, 1], :],      # (n_configs, 3, 3)
            dim=-1,
        )                                             # (n_configs, n_lattice, n_cartesian)

        # face-to-face distances of the parallelepiped
        L = (
            torch.linalg.det(batch_cells_tensor).abs().unsqueeze(-1)    # (n_configs,) -> (n_configs, 1)
            / torch.linalg.norm(N, dim=-1)                              # (n_configs, n_lattice)
        )                                                               # (n_configs, n_lattice)

        # matscipy-style binning
        n_bins = torch.clamp(
            torch.floor(L / cutoff),                                    # (n_configs, n_lattice)
            min=1,
        )                                                               # (n_configs, n_lattice)

        # max lattice reach across configurations
        S_max = torch.max(
            torch.ceil(cutoff * n_bins / L),                            # (n_configs, n_lattice)
            dim=0,
        ).values.long()                                                 # (n_lattice,)

        # integer lattice shifts
        batch_lattice_shifts_tensor = torch.cartesian_prod(
            torch.arange(-S_max[0], S_max[0] + 1,
                         device=S_max.device, dtype=S_max.dtype),       # (2*S_max[0]+1,)
            torch.arange(-S_max[1], S_max[1] + 1,
                         device=S_max.device, dtype=S_max.dtype),       # (2*S_max[1]+1,)
            torch.arange(-S_max[2], S_max[2] + 1,
                         device=S_max.device, dtype=S_max.dtype),       # (2*S_max[2]+1,)
        )                                                               # (n_lattice_shifts, 3)

        return batch_lattice_shifts_tensor

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
            Tensor of shape (n_configs, n_max, 3) with batched atomic positions.
        batch_cells_tensor : torch.Tensor
            Tensor of shape (n_configs, 3, 3) with batched cell matrices
            (rows = lattice vectors).
        batch_mask_tensor : torch.Tensor
            Boolean tensor of shape (n_configs, n_max) marking valid atoms.
        cutoff : float
            Cutoff radius for neighbour detection.
        tolerance : float
            Lower distance bound used to exclude self/near-self images.

        Returns
        -------
        distance_matrix : torch.Tensor
            Tensor of shape (n_configs, n_lattice_shifts, n_max, n_max)
            with pairwise distances for all lattice shifts.
        criterion : torch.Tensor
            Boolean tensor of shape (n_configs, n_lattice_shifts, n_max, n_max)
            marking neighbour pairs within the cutoff (and >= tolerance).
        batch_lattice_shifts_tensor : torch.Tensor
            Tensor of shape (n_lattice_shifts, 3) with integer lattice shift
            vectors S = (s_x, s_y, s_z).
        batch_cartesian_lattice_shifts_tensor : torch.Tensor
            Tensor of shape (n_configs, n_lattice_shifts, 3) with Cartesian
            lattice shift vectors S · h.
        """

        # integer lattice shifts
        batch_lattice_shifts_tensor = self._calculate_batch_lattice_shifts(
            batch_cells_tensor,                # (n_configs, 3, 3)
            cutoff=cutoff,
        )                                      # (n_lattice_shifts, 3)

        # same shifts in Cartesian coordinates
        batch_cartesian_lattice_shifts_tensor = torch.matmul(
            batch_lattice_shifts_tensor.to(batch_cells_tensor.dtype),  # (n_lattice_shifts, 3)
            batch_cells_tensor,                                        # (n_configs, 3, 3)
        )                                                              # (n_configs, n_lattice_shifts, 3)

        # shifted positions for each lattice shift
        batch_shifted_positions_tensor = (
            batch_cartesian_lattice_shifts_tensor.unsqueeze(-2)        # (n_configs, n_lattice_shifts, 1, 3)
            + batch_positions_tensor.unsqueeze(1)                      # (n_configs, 1, n_max, 3)
        )                                                              # (n_configs, n_lattice_shifts, n_max, 3)

        # pairwise differences
        diff = (
            batch_positions_tensor.unsqueeze(1).unsqueeze(3)           # (n_configs, 1, 1, n_max, 3)
            - batch_shifted_positions_tensor.unsqueeze(2)              # (n_configs, n_lattice_shifts, n_max, 1, 3)
        )                                                              # (n_configs, n_lattice_shifts, n_max, n_max, 3)

        # pairwise distances
        distance_matrix = torch.sqrt((diff ** 2).sum(dim=-1))          # (n_configs, n_lattice_shifts, n_max, n_max)

        # neighbour criterion based on cutoff / tolerance
        distance_matrix_criterion = (
            (distance_matrix < cutoff) &                               # (n_configs, n_lattice_shifts, n_max, n_max)
            (distance_matrix >= tolerance)
        )                                                              # (n_configs, n_lattice_shifts, n_max, n_max)

        # atom-pair mask from validity mask
        default_mask = (
            batch_mask_tensor.unsqueeze(-2)                            # (n_configs, n_max, 1)
            & batch_mask_tensor.unsqueeze(-1)                          # (n_configs, 1, n_max)
        )                                                              # (n_configs, n_max, n_max)
        default_mask = default_mask.unsqueeze(1)                       # (n_configs, 1, n_max, n_max)

        # final neighbour criterion
        criterion = distance_matrix_criterion & default_mask           # (n_configs, n_lattice_shifts, n_max, n_max)

        return (
            distance_matrix,                       # (n_configs, n_lattice_shifts, n_max, n_max)
            criterion,                             # (n_configs, n_lattice_shifts, n_max, n_max)
            batch_lattice_shifts_tensor,           # (n_lattice_shifts, 3)
            batch_cartesian_lattice_shifts_tensor  # (n_configs, n_lattice_shifts, 3)
        )

    def get_matscipy_output_from_batch_output(
        self,
        r_edges: torch.Tensor,                    # (2, n_edges) 
        r_integer_lattice_shifts: torch.Tensor,   # (n_edges, 3)
        r_cartesian_lattice_shifts: torch.Tensor, # (n_edges, 3)
        r_distances: torch.Tensor,                # (n_edges,)
        device: str | torch.device | None = None,
    ):
        """
        Convert flattened batched neighbour-list output to per-configuration
        matscipy-style lists.

        Parameters
        ----------
        r_edges : torch.Tensor
            Tensor of shape (2, n_edges) with flattened source and neighbour
            atom indices in the global (batched) indexing.
        r_integer_lattice_shifts : torch.Tensor
            Tensor of shape (n_edges, 3) with integer lattice shift vectors.
        r_cartesian_lattice_shifts : torch.Tensor
            Tensor of shape (n_edges, 3) with Cartesian lattice shift vectors.
        r_distances : torch.Tensor
            Tensor of shape (n_edges,) with interatomic distances.
        device : str or torch.device or None, optional
            If "cpu" / torch.device("cpu"), outputs are moved to CPU.
            If None, outputs stay on the input device.

        Returns
        -------
        atom_index_list : list[torch.Tensor]
            List of length n_configs; each entry is a (n_edges_cfg,) tensor
            of source atom indices (local to that configuration).
        neighbor_index_list : list[torch.Tensor]
            List of length n_configs; each entry is a (n_edges_cfg,) tensor
            of neighbour atom indices (local to that configuration).
        int_shift_list : list[torch.Tensor]
            List of length n_configs; each entry is a (n_edges_cfg, 3) tensor
            of integer lattice shifts.
        cart_shift_list : list[torch.Tensor]
            List of length n_configs; each entry is a (n_edges_cfg, 3) tensor
            of Cartesian lattice shifts.
        distance_list : list[torch.Tensor]
            List of length n_configs; each entry is a (n_edges_cfg,) tensor
            of interatomic distances.
        """

        if device is None:
            target_device = r_edges.device
        elif not isinstance(device, (str, torch.device)):
            raise TypeError(
                f"device should be a string or torch.device, got {type(device).__name__}."
            )
        else:
            target_device = torch.device(device)

        move_to_cpu = (target_device.type == "cpu")

        lengths = self.batch_mask_tensor.sum(dim=-1, dtype=torch.long)  # (n_configs,)
        offsets = torch.cumsum(lengths, dim=0) - lengths                # (n_configs,)
        n_configs = lengths.size(0)

        atom_index_list = []
        neighbor_index_list = []
        int_shift_list = []
        cart_shift_list = []
        distance_list = []

        for cfg in range(n_configs):
            start = offsets[cfg]
            stop = start + lengths[cfg]

            mask = (r_edges[0] >= start) & (r_edges[0] < stop)          # (n_edges,)

            i_global = r_edges[0, mask]                                 # (n_edges_cfg,)
            j_global = r_edges[1, mask]                                 # (n_edges_cfg,)
            i_local = i_global - start
            j_local = j_global - start

            if move_to_cpu:
                atom_index_list.append(i_local.to("cpu"))
                neighbor_index_list.append(j_local.to("cpu"))
                int_shift_list.append(r_integer_lattice_shifts[mask].to("cpu"))
                cart_shift_list.append(r_cartesian_lattice_shifts[mask].to("cpu"))
                distance_list.append(r_distances[mask].to("cpu"))
            else:
                atom_index_list.append(i_local)
                neighbor_index_list.append(j_local)
                int_shift_list.append(r_integer_lattice_shifts[mask])
                cart_shift_list.append(r_cartesian_lattice_shifts[mask])
                distance_list.append(r_distances[mask])

        return (
            atom_index_list,
            neighbor_index_list,
            int_shift_list,
            cart_shift_list,
            distance_list,
        )

