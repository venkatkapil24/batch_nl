# batch_nl — Batched neighbour-list builder in PyTorch

`batch_nl` provides fully vectorised, GPU‑accelerated batched neighbour‑list construction
for periodic atomistic systems using PyTorch.

All configurations are processed together in a single tensor batch, enabling high‑throughput
NL computation and seamless integration with MLIP workflows.

---

## Installation

### One-line install (from GitHub)

```bash
pip install git+https://github.com/venkatkapil24/batch_nl.git
```

### Install from source (recommended for development)

```bash
git clone https://github.com/venkatkapil24/batch_nl.git
cd batch_nl
pip install -e .
```

---

## Quick start (batched usage)

```python
from ase.build import bulk
from batch_nl import NeighbourList

cutoff = 3.0
device = "cuda:0"

base = bulk("C", "diamond", a=3.57)

configs = [
    base * (2, 2, 2),   # config 0
    base * (3, 3, 3),   # config 1
    base * (4, 4, 4),   # config 2
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

for cfg in range(len(configs)):
    print(f"Configuration {cfg}: {len(atom_index_list[cfg])} neighbour pairs")
```

---

## Understanding the output

### 1. Global batched output (`calculate_neighbourlist`)

```
r_edges                 (2, n_edges)
r_S_int                 (n_edges, 3)
r_S_cart                (n_edges, 3)
r_distances             (n_edges,)
```

Atoms from different configurations are concatenated into a **global index**:

- Config 0: atoms `[0 … N0−1]`
- Config 1: atoms `[N0 … N0+N1−1]`
- Config 2: atoms `[N0+N1 … N0+N1+N2−1]`

Thus a pair like:

```
r_edges[:, k] = [42, 99]
```

means that *global* atom 42 has neighbour 99 under some lattice shift.  
This representation is ideal for high‑throughput GPU workflows.

---

### 2. Per‑configuration (matscipy-style) output

`get_matscipy_output_from_batch_output(...)` produces lists of length `n_configs`:

- `atom_index_list[cfg]`      → `(n_edges_cfg,)` local source indices  
- `neighbor_index_list[cfg]`  → `(n_edges_cfg,)` local neighbour indices  
- `int_shift_list[cfg]`       → `(n_edges_cfg, 3)` integer shifts  
- `cart_shift_list[cfg]`      → `(n_edges_cfg, 3)` Cartesian shifts  
- `distance_list[cfg]`        → `(n_edges_cfg,)` distances  

Local indices always run from `0 … n_atoms_in_cfg−1`.  
This matches the standard matscipy interface.

---

## API overview

### `NeighbourList`

```python
nl = NeighbourList(
    list_of_positions=list_of_positions,
    list_of_cells=list_of_cells,
    cutoff=cutoff,
    device=device,
)
```

#### Parameters

- **list_of_positions** — list of `(n_i, 3)` Cartesian coordinates  
- **list_of_cells** — list of `(3, 3)` cell matrices  
- **cutoff** — scalar cutoff (float, int, or tensor)  
- **device** — `"cpu"`, `"cuda"`, `"cuda:0"`, or `torch.device`  

---

### `load_data()`

Creates padded batched tensors on the target device:

- `batch_positions_tensor : (n_configs, n_max, 3)`
- `batch_mask_tensor      : (n_configs, n_max)`
- `batch_cell_tensor      : (n_configs, 3, 3)`

---

### `calculate_neighbourlist(use_torch_compile=True)`

Returns:

```
r_edges                 (2, n_edges)
r_S_int                 (n_edges, 3)
r_S_cart                (n_edges, 3)
r_distances             (n_edges,)
```

---

### `get_matscipy_output_from_batch_output(...)`

Converts global indexing → per‑configuration local indexing.

---

## Testing

```bash
pytest
```

---

## License

`batch_nl` is distributed under the **Apache License 2.0**.  
Users are free to use, modify, and redistribute the software, provided attribution
and license terms are preserved.