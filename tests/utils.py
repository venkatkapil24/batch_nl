import torch
from collections import Counter
from matscipy.neighbours import neighbour_list as matscipy_neighbour_list
from pynl import NeighbourList


def _check_neighbourlist_ON2_matches_matscipy(atoms, radius: float, use_torch_compile=False):
    """
    Compare the O(N²) neighbour list against the matscipy reference.

    Builds a `NeighbourList` for a single configuration and cutoff, runs the
    O(N²) backend (optionally with `torch.compile`), and checks that the
    resulting neighbour pairs match those from `matscipy_neighbour_list`
    up to a small rounding tolerance on the distances.

    Parameters
    ----------
    atoms : Atoms
        Single ASE-like Atoms object to test.
    radius : float
        Cutoff radius for the neighbour list.
    use_torch_compile : bool, optional
        If True, use the `torch.compile`-optimised O(N²) kernel; otherwise
        use the uncompiled version.

    Raises
    ------
    AssertionError
        If there are any pairs present in the matscipy neighbour list but not
        in the O(N²) implementation, or vice versa.
    """

    nl = NeighbourList(
        list_of_configurations=[atoms],
        radius=radius,
        batch_size=1,
        device=None,
    )
    nl.load_data()
    out = nl.calculate_neighbourlist(use_torch_compile=use_torch_compile)

    # assuming ON2 returns the same [i, j, S, d] structure
    i, j, S, D, d = out[0][0]
    i = i.cpu()
    j = j.cpu()
    S = S.cpu()
    d = d.cpu()
    D = D.cpu()

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

    missing = ase_pairs - pairs   # in matscipy but not here
    extra = pairs - ase_pairs     # here but not in matscipy

    assert not missing and not extra, (
        f"Neighbour list mismatch\n"
        f"Missing (ASE - ON2): {missing}\n"
        f"Extra (ON2 - ASE):   {extra}"
    )

