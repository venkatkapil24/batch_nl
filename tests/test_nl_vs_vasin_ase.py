import torch
from collections import Counter
from ase import Atoms
from vesin import ase_neighbor_list

from neighbourlist import NeighbourList

def test_neighbourlist_counter_based():

    # --- test system ---
    cell = [
        [2.460394,  0.0,       0.0],
        [-1.26336,  2.044166,  0.0],
        [-0.139209, -0.407369, 6.809714],
    ]

    positions = [
        [-0.03480225, -0.10184225, 1.70242850],
        [-0.10440675, -0.30552675, 5.10728550],
        [-0.05691216,  1.26093576, 1.70242850],
        [ 1.11473716,  0.37586124, 5.10728550],
    ]

    carbon = Atoms(
        symbols="CCCC",
        positions=positions,
        cell=cell,
        pbc=True,
    )

    # --- our neighbour list ---
    nl = NeighbourList(
        list_of_configurations=[carbon],
        radius=3.0,
        batch_size=1,
        device=None,   # auto CPU/GPU
    )

    nl.load_data()
    out = nl.calculate_neighbourlist_ON2(use_torch_compile=False)

    # unpack output
    i, j, d, S = out[0][0]

    # --- ASE reference ---
    ASE_i, ASE_j, ASE_S, ASE_d = ase_neighbor_list("ijSd", carbon, cutoff=3.0)

    # --- Counter-based comparisons ---

    # i and j — integers
    assert Counter(int(ii) for ii in i.cpu()) == Counter(int(ii) for ii in ASE_i)
    assert Counter(int(jj) for jj in j.cpu()) == Counter(int(jj) for jj in ASE_j)

    # S — lattice shift rows as tuples
    assert Counter(tuple(row.tolist()) for row in S.cpu()) == \
           Counter(tuple(row) for row in ASE_S)

    # d — floating distances (compare rounded to avoid tiny FP noise)
    assert Counter(round(float(x), 3) for x in d.cpu()) == \
           Counter(round(float(x), 3) for x in ASE_d)


    print("Neighbour list matches ASE — test passed.")
