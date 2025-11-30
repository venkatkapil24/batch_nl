from collections import Counter
import torch
from vesin import ase_neighbor_list
from neighbourlist import NeighbourList


def _check_neighbourlist_ON1_matches_ase(atoms, radius: float):
    nl = NeighbourList(
        list_of_configurations=[atoms],
        radius=radius,
        batch_size=1,
        device=None,
    )
    nl.load_data()
    out = nl.calculate_neighbourlist_ON1(use_torch_compile=False)

    # unpack VK output
    i, j, S, d = out[0][0]
    i = i.cpu()
    j = j.cpu()
    S = S.cpu()
    d = d.cpu()

    # ASE reference
    ASE_i, ASE_j, ASE_S, ASE_d = ase_neighbor_list("ijSd", atoms, cutoff=radius)

    pairs = Counter(
        (
            int(i[k].item()),
            int(j[k].item()),
            tuple(S[k].tolist()),
            round(float(d[k]), 3),
        )
        for k in range(len(i))
    )

    ase_pairs = Counter(
        (
            int(ASE_i[k].item()),
            int(ASE_j[k].item()),
            tuple(ASE_S[k].tolist()),
            round(float(ASE_d[k]), 3),
        )
        for k in range(len(ASE_i))
    )

    assert pairs == ase_pairs


def _check_neighbourlist_ON2_matches_ase(atoms, radius: float):
    nl = NeighbourList(
        list_of_configurations=[atoms],
        radius=radius,
        batch_size=1,
        device=None,
    )
    nl.load_data()
    out = nl.calculate_neighbourlist_ON2(use_torch_compile=False)

    # assuming ON2 returns the same [i, j, S, d] structure
    i, j, S, d = out[0][0]
    i = i.cpu()
    j = j.cpu()
    S = S.cpu()
    d = d.cpu()

    ASE_i, ASE_j, ASE_S, ASE_d = ase_neighbor_list("ijSd", atoms, cutoff=radius)

    pairs = Counter(
        (
            int(i[k].item()),
            int(j[k].item()),
            tuple(S[k].tolist()),
            round(float(d[k]), 3),
        )
        for k in range(len(i))
    )

    ase_pairs = Counter(
        (
            int(ASE_i[k].item()),
            int(ASE_j[k].item()),
            tuple(ASE_S[k].tolist()),
            round(float(ASE_d[k]), 3),
        )
        for k in range(len(ASE_i))
    )

    assert pairs == ase_pairs

