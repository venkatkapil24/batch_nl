import torch

class NeighbourList:
    """
    """

    def __init__(self, list_of_configurations, radius, batch_size, device):

        self.list_of_configurations = list_of_configurations
        self.radius = radius
        self.batch_size = batch_size
        self.device = device

        self.num_configs = len(self.list_of_configurations)
        self.num_batches = (self.num_configs + self.batch_size - 1) // self.batch_size

        self._nlist_ON2 = self._nlist_ON2

    def load_data(self):
        """
        Converts the list of configurations into a form suitable for batching. 
        """

        self.cell_list = [torch.tensor(atoms.cell, dtype=torch.float32) for atoms in self.list_of_configurations]
        self.positions_list = [torch.tensor(atoms.positions, dtype=torch.float32) for atoms in self.list_of_configurations]

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

            batch_positions_tensor = torch.zeros(self.batch_size, position_size_max, 3)
            batch_cells_tensor = torch.eye(3).unsqueeze(0).repeat(self.batch_size, 1, 1)
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

    def calculate_neighbourlist(self):
        """
        The money shot. 
        """

        r = []

        for batch_id in range(self.num_batches):

            batch_positions_tensor = self.batch_positions_tensor_list[batch_id].to(self.device)
            batch_cells_tensor = self.batch_cells_tensor_list[batch_id].to(self.device)
            batch_mask_tensor = self.batch_masks_tensor_list[batch_id].to(self.device)

            batch_nl = self._nlist_ON2(batch_positions_tensor, batch_cells_tensor, batch_mask_tensor, self.radius)
            r.append(batch_nl)

        return r
    
    def calculate_batch_lattice_shifts(self, batch_cells_tensor, radius):
        """
        Estimates which periodic images of atoms need to be considered
        given a cell and a radius. Keeps the image lattice shifts constant within a batch.
        """

        device = batch_cells_tensor.device

        batch_cell_lengths = torch.linalg.norm(batch_cells_tensor, dim=1)
        max_n = torch.max(torch.ceil(radius / batch_cell_lengths), dim=0).values

        mesh = torch.meshgrid(
            torch.arange(-max_n[0], max_n[0] + 1, dtype=torch.int8, device=device),
            torch.arange(-max_n[1], max_n[1] + 1, dtype=torch.int8, device=device),
            torch.arange(-max_n[2], max_n[2] + 1, dtype=torch.int8, device=device),
            indexing='ij'
        )

        mesh = torch.stack(mesh, dim=-1).reshape(-1, 3)
        return mesh

    def _nlist_ON2(self, batch_positions_tensor, batch_cells_tensor, batch_mask_tensor, radius):
        """
        Computes a distance squared matrix. In the comments, to explain the implementation, 
        I use b for batch structure idx. 
        """

        # b, lc, 3
        lattice_shifts = self.calculate_batch_lattice_shifts(batch_cells_tensor, radius)

        # (b, lc, 1, 3) + (b, 1, n, 3) -> b, lc, n, 3
        batch_shifted_positions_tensor = lattice_shifts.unsqueeze(0).unsqueeze(2) + batch_positions_tensor.unsqueeze(1)

        # b, 1, 1, n, 3 - b, lc, n, 1, 3 ->  b, lc, n, n
        distance_matrix = ((batch_positions_tensor.unsqueeze(-3).unsqueeze(-3) - batch_shifted_positions_tensor.unsqueeze(-2))**2).sum(dim=-1)

        distance_matrix_criteron = distance_matrix <= radius**2

        # get the appropriate mask for atom pair connectivity
        default_mask =  batch_mask_tensor.unsqueeze(-2) & batch_mask_tensor.unsqueeze(-1)

        final_criteron = torch.logical_and(distance_matrix_criteron, default_mask)

        return final_criteron

