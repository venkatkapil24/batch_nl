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
    out = nl.calculate_neighbourlist_ON1(use_torch_compile=False)

    # unpack output
    i, j, S, d = out[0][0]
    i = i.cpu()
    j = j.cpu()
    d = d.cpu()
    S = S.cpu()

    # --- ASE reference ---
    ASE_i, ASE_j, ASE_S, ASE_d = ase_neighbor_list("ijSd", carbon, cutoff=3.0)

    # --- build multiset of neighbour tuples for VK ---
    pairs = Counter(
    (
        int(i[k].item()),
        int(j[k].item()),
        tuple(S[k].tolist()),
        round(float(d[k]), 3),
    )
    for k in range(len(i))
    )

    # --- build multiset of neighbour tuples for ASE ---
    ase_pairs = Counter(
    (
        int(ASE_i[k].item()),
        int(ASE_j[k].item()),
        tuple(ASE_S[k].tolist()),
        round(float(ASE_d[k]), 3),
    )
    for k in range(len(ASE_i))
    )

    print (len(ASE_i), len(i))
    print (len(ASE_j), len(j))
    print (len(ASE_S), len(S))
    print (len(ASE_d), len(d))

    assert pairs == ase_pairs

    print("Neighbour list matches ASE — test passed.")
