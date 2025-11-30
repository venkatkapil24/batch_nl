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
        Converts the list of configurations into a form suitable for batching. 
        """

        self.cell_list = [torch.tensor(atoms.cell.array, dtype=self.float_dtype) for atoms in self.list_of_configurations]
        self.positions_list = [torch.tensor(atoms.positions, dtype=self.float_dtype) for atoms in self.list_of_configurations]

        self._batch_and_mask_positions_and_cells()

    def _batch_and_mask_positions_and_cells(self):
        """
        Returns the position and cell data in [num_batches, batch_size, 
        dim(position / cell)] format and a mask tensor which allows to
        map back to the lists.
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

    def calculate_neighbourlist_ON2(self, use_torch_compile=True):
        """
        The money shot. 
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
            lattice_shifts = self.calculate_batch_lattice_shifts(batch_cells_tensor,)

            distance_matrix, criterion = neighbourlist_fn(batch_positions_tensor, batch_cells_tensor, batch_mask_tensor, lattice_shifts, self.radius, self.tolerance)

            b_r = []

            for i in range(self.batch_size):
              
                lattice_shift_idx, atom_idx, neighbour_idx = torch.nonzero(criterion[i], as_tuple=True)

                d = distance_matrix[i, lattice_shift_idx, atom_idx, neighbour_idx]
                S = lattice_shifts[lattice_shift_idx]

                b_r.append([atom_idx, neighbour_idx, S, d])

            r.append(b_r)

        return r
    
    def calculate_neighbourlist_ON1(self, use_torch_compile=True):
        """
        The money shot. 
        """

        r = []

        neighbourlist_fn = self._nlist_ON1

        #if use_torch_compile:
        #    neighbourlist_fn = self._nlist_ON1_compiled
        #else:
        #    neighbourlist_fn = self._nlist_ON1

        for batch_id in range(self.num_batches):

            batch_positions_tensor = self.batch_positions_tensor_list[batch_id].to(self.device)
            batch_cells_tensor = self.batch_cells_tensor_list[batch_id].to(self.device)
            batch_mask_tensor = self.batch_masks_tensor_list[batch_id].to(self.device)
            lattice_shifts = self.calculate_batch_lattice_shifts(batch_cells_tensor,)

            out = neighbourlist_fn(batch_positions_tensor, batch_cells_tensor, batch_mask_tensor, lattice_shifts, self.radius, self.tolerance)

            b_r = []
            
            for b in range(self.batch_size):
                mask = (out[:, 0] == b)
                i_b = out[mask, 1].to(torch.long)
                j_b = out[mask, 2].to(torch.long)
                shift_b = out[mask, 3:6]
                d_b = out[mask, 6]
                b_r.append([i_b, j_b, shift_b, d_b])

            r.append(b_r)

        return r

    def calculate_batch_lattice_shifts(self, batch_cells_tensor):
        """
        Estimates which periodic images of atoms need to be considered
        given a cell and a radius. Keeps the image lattice shifts constant within a batch.
        """

        device = batch_cells_tensor.device

        batch_cell_lengths = torch.linalg.norm(batch_cells_tensor, dim=-1)
        max_n = torch.max(torch.ceil(self.radius / batch_cell_lengths), dim=0).values

        mesh = torch.meshgrid(
            torch.arange(-max_n[0], max_n[0] + 1, dtype=self.int_dtype, device=device),
            torch.arange(-max_n[1], max_n[1] + 1, dtype=self.int_dtype, device=device),
            torch.arange(-max_n[2], max_n[2] + 1, dtype=self.int_dtype, device=device),
            indexing='ij'
        )

        mesh = torch.stack(mesh, dim=-1).reshape(-1, 3)
        return mesh

    def _nlist_ON2(self, batch_positions_tensor, batch_cells_tensor, batch_mask_tensor, lattice_shifts, radius, tolerance):
        """
        Computes a distance squared matrix. In the comments, to explain the implementation, 
        I use b for batch structure idx. 
        """

        # 1, lc, 3  X b, 1, 3, 3 -> b, lc, 3
        lattice_shifts = torch.einsum("li,bij->blj", lattice_shifts.to(batch_cells_tensor.dtype), batch_cells_tensor)

        # (b, lc, 1, 3) + (b, 1, n, 3) -> b, lc, n, 3
        batch_shifted_positions_tensor = lattice_shifts.unsqueeze(-2) + batch_positions_tensor.unsqueeze(1)

        # b, 1, 1, n, 3 - b, lc, n, 1, 3 ->  b, lc, n, n
        distance_matrix = torch.sqrt(((batch_positions_tensor.unsqueeze(1).unsqueeze(3) - batch_shifted_positions_tensor.unsqueeze(2))**2).sum(dim=-1))

        distance_matrix_criterion = (distance_matrix <= radius) & (distance_matrix >= tolerance)

        # get the appropriate mask for atom pair connectivity
        default_mask = batch_mask_tensor.unsqueeze(-2) & batch_mask_tensor.unsqueeze(-1)
        
        # adds dim for lattice_shifts
        default_mask = default_mask.unsqueeze(1)

        criterion = torch.logical_and(distance_matrix_criterion, default_mask)

        return distance_matrix, criterion
    

    def _nlist_ON1(self, batch_positions_tensor, batch_cells_tensor, batch_mask_tensor, lattice_shifts, radius, tolerance):
        """
        Computes a linked-cell to identify neighbours.
        """

        # estimate maximum bins for the system
        
        # b, cell_idx, cartesian_idx -> b, cell_idx
        batch_cell_lengths_tensor = torch.linalg.norm(batch_cells_tensor, dim=-1)

        # b, cell_idx
        max_bins = torch.clamp((batch_cell_lengths_tensor / self.radius).floor(), min=1).to(self.int_dtype)

        # estimate linked cell indices of the system

        # b, cell_idx, cartesian_idx -> b, cell_idx, cartesian_idx
        batch_cell_inverses_tensor = torch.linalg.inv(batch_cells_tensor)
        
        # b, atom_idx, cartesian_idx X b, cell_idx, cartesian_idx (?)
        batch_fractional_positions = batch_positions_tensor @ batch_cell_inverses_tensor
        batch_fractional_positions = batch_fractional_positions - torch.floor(batch_fractional_positions)

        # b, atom_idx, cartesian_idx X b, 1, cell_idx
        batch_c = (batch_fractional_positions * max_bins.unsqueeze(1).to(batch_fractional_positions.dtype))
        batch_c = batch_c.floor().to(self.int_dtype)

        # estimate linked cell indices of periodic images
        batch_fractional_image_positions = batch_fractional_positions.unsqueeze(1) + lattice_shifts.view(1, -1, 1, 3).to(batch_fractional_positions.dtype)

        batch_image_c = (batch_fractional_image_positions * max_bins.view(-1, 1, 1, 3).to(batch_fractional_positions.dtype)).floor().to(self.int_dtype)

        batch_cell_mask = ((batch_image_c.unsqueeze(3) - batch_c.unsqueeze(1).unsqueeze(1)).abs() <= 2).all(dim=-1)

        b_idx, k_idx, j_idx, i_idx = batch_cell_mask.nonzero(as_tuple=True)

        r_i = batch_positions_tensor[b_idx, i_idx]

        # (atom_idx, cartesian_idx) + (atom_idx, 1, cell_idx)  X (atom_idx, cell_idx, cartesian_idx) 
        r_j = batch_positions_tensor[b_idx, j_idx] + (lattice_shifts[k_idx].to(batch_positions_tensor.dtype).unsqueeze(1) @ batch_cells_tensor[b_idx]).squeeze(1)

        ds = (r_i - r_j).norm(dim=-1)

        default_mask = batch_mask_tensor[b_idx, i_idx] & batch_mask_tensor[b_idx, j_idx]
        distance_mask = (ds < radius) & (ds >= tolerance)
        final_criterion = default_mask & distance_mask        

        return torch.stack([b_idx, i_idx, j_idx, lattice_shifts[k_idx][:,0], lattice_shifts[k_idx][:,1], lattice_shifts[k_idx][:,2], ds], dim=1)[final_criterion]

    
