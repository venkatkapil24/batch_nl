# pynl

Efficient and batched PyTorch-based neighbour list builder for atomistic simulations.

pynl provides a simple, GPU-friendly neighbour-list API that takes a list
of periodic ASE Atoms objects, packs them into fixed-size batches, and
builds neighbour lists using vectorised PyTorch operations.

---------------------------------------------------------------------

## Installation

From source (recommended for now):

    git clone https://github.com/venkatkapil24/pynl.git
    cd pynl
    pip install -e .

This installs the package as "pynl-torch", importable as:

    import pynl
    from pynl import NeighbourList

Once published on PyPI, installation will be as simple as:

    pip install pynl-torch

---------------------------------------------------------------------

## Quick start

Single configuration:

    import torch
    from ase.build import bulk
    from pynl import NeighbourList

    device = "cuda:0"   # or "cpu"
    radius = 3.0        # cutoff in Angstrom

    # Build a simple test system: diamond C 2x2x2
    base = bulk("C", "diamond", a=3.57)
    carbon = base * (2, 2, 2)
    print("num_atoms:", len(carbon))

    # Construct neighbour list object
    nl = NeighbourList(
        list_of_configurations=[carbon],
        radius=radius,
        batch_size=1,
        device=device,
    )

    # Convert ASE structures to batched tensors
    nl.load_data()

    # Compute neighbour list with the O(N^2) backend (no torch.compile)
    out = nl.calculate_neighbourlist(use_torch_compile=False)

    # Unpack neighbours for the first (and only) configuration in the first batch
    i, j, S, d = out[0][0]  # indices, neighbour indices, lattice shifts, distances

    print("Number of neighbour pairs:", len(i))
    print("First few pairs:")
    for k in range(min(5, len(i))):
        print(
            f"{int(i[k])} -> {int(j[k])}, "
            f"S={S[k].tolist()}, "
            f"d={float(d[k]):.3f} Angstrom"
        )

Batched usage (multiple structures):

    from ase.build import bulk
    from pynl import NeighbourList

    radius = 3.0
    device = "cuda:0"

    base = bulk("C", "diamond", a=3.57)
    configs = [
        base * (2, 2, 2),
        base * (3, 3, 3),
        base * (4, 4, 4),
    ]

    nl = NeighbourList(
        list_of_configurations=configs,
        radius=radius,
        batch_size=2,   # process two configurations per batch
        device=device,
    )
    nl.load_data()
    batched_out = nl.calculate_neighbourlist(use_torch_compile=True)

    for batch_id, batch_result in enumerate(batched_out):
        for local_idx, (i, j, S, d) in enumerate(batch_result):
            print(
                f"Batch {batch_id}, structure {local_idx}: "
                f"{len(i)} neighbour pairs"
            )

---------------------------------------------------------------------

## API overview

NeighbourList:

    from pynl import NeighbourList

    nl = NeighbourList(
        list_of_configurations: list,
        radius: float,
        batch_size: int,
        device: str | torch.device | None = None,
    )

Parameters:

- list_of_configurations:
  list of ASE-like Atoms objects (with .positions, .cell, and periodic
  boundary conditions).

- radius:
  scalar cutoff radius (Angstrom).

- batch_size:
  number of configurations to process together in one batch.

- device:
  "cpu", "cuda", "cuda:0", etc., or a torch.device. If None, falls back
  to a default device.

Main methods:

- load_data()

  Converts input configurations into PyTorch tensors and packs positions
  and cells into padded tensors of shape (batch_size, n_max, 3) with a
  corresponding boolean mask.

- calculate_neighbourlist(use_torch_compile: bool = True)

  Computes neighbour lists for all configurations in all batches using
  the O(N^2) backend.

  If use_torch_compile=True, uses a torch.compile-optimised kernel
  (where available).

  Returns a nested list:

      [
        [
          [i, j, S, d],   # configuration 0 in batch 0
          [i, j, S, d],   # configuration 1 in batch 0
          ...
        ],
        [
          [i, j, S, d],   # configuration 0 in batch 1
          ...
        ],
        ...
      ]

  where each [i, j, S, d] consists of:

  - i: 1D tensor of central atom indices
  - j: 1D tensor of neighbour atom indices
  - S: integer lattice shifts (n_pairs, 3)
  - d: 1D tensor of distances

---------------------------------------------------------------------

## Dependencies

Runtime dependencies (managed via pyproject.toml):

- torch>=1.12
- ase
- matscipy

Install a PyTorch build appropriate for your hardware first (CPU or CUDA),
then install pynl-torch from source as shown above.

---------------------------------------------------------------------

## Testing

From the repository root:

    pytest

Tests compare the O(N^2) neighbour list against matscipy neighbour-list
output across a range of crystal structures and cell types (oblique, cubic,
tetragonal, orthorhombic, hcp, rhombohedral, triclinic, etc.).

---------------------------------------------------------------------

## License

The pynl project is licensed under the Academic Software License v1.0 (ASL) - see the LICENSE file for details.

Copyright

[pynl] is (c) 2024, [Venkat Kapil]

[pynl] is published and distributed under the Academic Software License v1.0 (ASL).

[pynl] is distributed in the hope that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the ASL for more details.

You should have received a copy of the ASL along with this program; if not, write to [venkat.kapil@gmail.com]. It is also published at [ASL-link-here].

You may contact the original licensor at [venkat.kapil@gmail.com].
