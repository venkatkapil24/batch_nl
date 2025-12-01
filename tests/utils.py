from collections import Counter
import torch
from matscipy.neighbours import neighbour_list as matscipy_neighbour_list
from neighbourlist import NeighbourList


def _check_neighbourlist_ON2_matches_ase(atoms, radius: float, use_torch_compile=False):
    nl = NeighbourList(
        list_of_configurations=[atoms],
        radius=radius,
        batch_size=1,
        device=None,
    )
    nl.load_data()
    out = nl.calculate_neighbourlist(use_torch_compile=use_torch_compile)

    # assuming ON2 returns the same [i, j, S, d] structure
    i, j, S, d = out[0][0]
    i = i.cpu()
    j = j.cpu()
    S = S.cpu()
    d = d.cpu()

    ASE_i, ASE_j, ASE_S, ASE_d = matscipy_neighbour_list("ijSd", atoms, cutoff=radius)

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

    missing = ase_pairs - pairs   # in ASE but not ON2
    extra = pairs - ase_pairs     # in ON2 but not ASE

    assert not missing and not extra, (
        f"Neighbour list mismatch\n"
        f"Missing (ASE - ON2): {missing}\n"
        f"Extra (ON2 - ASE):   {extra}"
    )


    assert pairs == ase_pairs

