# pynl — Batched neighbour-list builder in PyTorch

`pynl` provides a fully vectorised, batched neighbour-list construction
for periodic atomistic systems using PyTorch. Configurations of different
sizes are padded into a single batch and processed entirely on the chosen
device (CPU or CUDA), enabling fast and memory-efficient neighbour list
generation.

---

## Installation

### One-line install (directly from GitHub)

```bash
pip install git+https://github.com/venkatkapil24/pynl.git
```

### Install from source (recommended for development)

```bash
git clone https://github.com/venkatkapil24/pynl.git
cd pynl
pip install -e .
```

---

## Quick start (single configuration)

```python
import torch
from ase.build import bulk
from pynl import NeighbourList

device = "cuda:0"   # or "cpu"
cutoff = 3.0        # cutoff in Angstrom

# Build test system: diamond C 2x2x2
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
```

---

## Batched usage (multiple configurations)

```python
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
    print(f"Configuration {cfg_idx}: {len(i)} neighbour pairs")
```

---

## API overview

### `NeighbourList`

```python
from pynl import NeighbourList

nl = NeighbourList(
    list_of_positions=list_of_positions,
    list_of_cells=list_of_cells,
    cutoff=cutoff,
    device=device,
)
```

Parameters:

- **list_of_positions**  
  List of `(n_i, 3)` arrays of Cartesian positions (one per configuration).

- **list_of_cells**  
  List of `(3, 3)` cell matrices (one per configuration). Must match the
  length of `list_of_positions`.

- **cutoff**  
  Scalar cutoff radius in Angstrom.

- **device**  
  `"cpu"`, `"cuda"`, `"cuda:0"`, or `torch.device`.  
  If `None`, defaults to CUDA when available, otherwise CPU.

All configurations are handled as a single batch.

### Main methods

#### `load_data()`

Converts input lists into padded batched tensors:

- `batch_positions_tensor` : `(n_configs, n_max, 3)`
- `batch_mask_tensor`      : `(n_configs, n_max)`
- `batch_cell_tensor`      : `(n_configs, 3, 3)`

and moves them to `self.device`.

#### `calculate_neighbourlist(use_torch_compile=True)`

Computes neighbour lists using the O(N²) backend.

If `use_torch_compile=True`, the function is executed through
`torch.compile(self._nlist_ON2)`.

Returns:

- `r_edges`                    : `(2, n_edges)`
- `r_integer_lattice_shifts`   : `(n_edges, 3)`
- `r_cartesian_lattice_shifts` : `(n_edges, 3)`
- `r_distances`                : `(n_edges,)`

Indices in `r_edges` are global over all configurations in the batch.

#### `get_matscipy_output_from_batch_output(...)`

Converts flattened batched neighbour-list output into per-configuration
lists:

- `atom_index_list`      : list of 1D tensors of local source indices
- `neighbor_index_list`  : list of 1D tensors of local neighbour indices
- `int_shift_list`       : list of `(n_edges_cfg, 3)` integer shift tensors
- `cart_shift_list`      : list of `(n_edges_cfg, 3)` Cartesian shift tensors
- `distance_list`        : list of 1D tensors of distances

Each list has length `n_configs`.

---

## Testing

From the repository root:

```bash
pytest
```

Tests verify agreement with `matscipy` across a range of crystal
structures and cell geometries (cubic, tetragonal, orthorhombic, hcp,
rhombohedral, triclinic, oblique, etc.).

---

## License

The `pynl` project is licensed under the Academic Software License v1.0 (ASL).
See the `LICENSE` file for details.

`pynl` is © 2024, Venkat Kapil. Distributed under the ASL v1.0 without
any warranty; see the license for details.

Contact: venkat.kapil@gmail.com