## Installation

From source (recommended for now):

    git clone https://github.com/venkatkapil24/pynl.git
    cd pynl
    pip install -e .

---------------------------------------------------------------------

## Quick start

Single configuration:

    import torch
    from ase.build import bulk
    from pynl import NeighbourList

    device = "cuda:0"   # or "cpu"
    cutoff = 3.0        # cutoff in Angstrom

    # Build a simple test system: diamond C 2x2x2
    base = bulk("C", "diamond", a=3.57)
    carbon = base * (2, 2, 2)
    print("num_atoms:", len(carbon))

    # Construct neighbour list object
    nl = NeighbourList(
        list_of_positions=[carbon.positions],
        list_of_cells=[carbon.cell.array],
        cutoff=cutoff,
        device=device,
    )

    # Convert input arrays to batched tensors
    nl.load_data()

    # Compute neighbour list (no torch.compile)
    r_edges, r_S_int, r_S_cart, r_d = nl.calculate_neighbourlist(
        use_torch_compile=False
    )

    # Convert to matscipy-style output
    (
        atom_index_list,
        neighbor_index_list,
        int_shift_list,
        cart_shift_list,
        distance_list,
    ) = nl.get_matscipy_output_from_batch_output(
        r_edges, r_S_int, r_S_cart, r_d
    )

    i = atom_index_list[0]
    j = neighbor_index_list[0]
    S = int_shift_list[0]
    d = distance_list[0]

    print("Number of neighbour pairs:", len(i))
    print("First few pairs:")
    for k in range(min(5, len(i))):
        print(
            f"{int(i[k])} -> {int(j[k])}, "
            f"S={S[k].tolist()}, "
            f"d={float(d[k]):.3f} Å"
        )

---------------------------------------------------------------------

Batched usage (multiple configurations):

    from ase.build import bulk
    from pynl import NeighbourList

    cutoff = 3.0
    device = "cuda:0"

    base = bulk("C", "diamond", a=3.57)
    configs = [
        base * (2, 2, 2),
        base * (3, 3, 3),
        base * (4, 4, 4),
    ]

    list_of_positions = [atoms.positions for atoms in configs]
    list_of_cells     = [atoms.cell.array for atoms in configs]

    nl = NeighbourList(
        list_of_positions=list_of_positions,
        list_of_cells=list_of_cells,
        cutoff=cutoff,
        device=device,
    )
    nl.load_data()

    r_edges, r_S_int, r_S_cart, r_d = nl.calculate_neighbourlist(
        use_torch_compile=True
    )

    (
        atom_index_list,
        neighbor_index_list,
        int_shift_list,
        cart_shift_list,
        distance_list,
    ) = nl.get_matscipy_output_from_batch_output(
        r_edges, r_S_int, r_S_cart, r_d
    )

    for cfg_idx in range(len(configs)):
        i = atom_index_list[cfg_idx]
        j = neighbor_index_list[cfg_idx]
        print(
            f"Configuration {cfg_idx}: {len(i)} neighbour pairs"
        )

---------------------------------------------------------------------

## API overview

NeighbourList:

    from pynl import NeighbourList

    nl = NeighbourList(
        list_of_positions: list,
        list_of_cells: list,
        cutoff: float,
        device: str | torch.device | None = None,
    )

Parameters:

- list_of_positions:
  List of (N_i, 3) position arrays (one per configuration).

- list_of_cells:
  List of (3, 3) cell matrices (one per configuration).
  Must match the length of list_of_positions.

- cutoff:
  Scalar cutoff radius (Angstrom).

- device:
  "cpu", "cuda", "cuda:0", etc., or torch.device.
  If None, defaults to CUDA if available, otherwise CPU.

All configurations are treated as a single batch.

Main methods:

- load_data()

      Converts position and cell lists into padded tensors:
      - batch_positions_tensor : (B, N_max, 3)
      - batch_mask_tensor      : (B, N_max)
      - batch_cell_tensor      : (B, 3, 3)
      Moves all to the configured device.

- calculate_neighbourlist(use_torch_compile=True)

      Computes neighbour lists using the O(N^2) backend.
      If use_torch_compile=True, uses torch.compile when available.

      Returns:
          r_edges                 (2, E)
          r_integer_lattice_shifts (E, 3)
          r_cartesian_lattice_shifts (E, 3)
          r_distances              (E,)

      Indices in r_edges are global over all configurations.

- get_matscipy_output_from_batch_output(...)

      Converts flattened batched neighbour-list output into
      per-configuration lists:

      Returns:
          atom_index_list      (list of 1D tensors)
          neighbor_index_list  (list of 1D tensors)
          int_shift_list       (list of (n_i, 3) tensors)
          cart_shift_list      (list of (n_i, 3) tensors)
          distance_list        (list of 1D tensors)

---------------------------------------------------------------------

## Testing

From the repository root:

    pytest

Tests compare the O(N^2) neighbour-list output with matscipy over
various crystal systems and cell geometries (cubic, tetragonal,
orthorhombic, hcp, rhombohedral, triclinic, oblique, etc.).

---------------------------------------------------------------------

## License

The pynl project is licensed under the Academic Software License v1.0 (ASL).
See the LICENSE file for details.

[pynl] is (c) 2024, Venkat Kapil

Distributed under the ASL v1.0 without any warranty; see the license
for details.

Contact: venkat.kapil@gmail.com
