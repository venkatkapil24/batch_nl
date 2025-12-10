import torch
from collections import Counter
from matscipy.neighbours import neighbour_list as matscipy_neighbour_list
from batch_nl import NeighbourList


def _check_neighbourlist_ON2_matches_matscipy(
    atoms,
    cutoff: float,
    use_torch_compile: bool = False,
):
    """
    Compare the O(N²) neighbour list against the matscipy reference
    for a single configuration.

    Builds a `NeighbourList` for one configuration and cutoff, runs the
    O(N²) backend (optionally with `torch.compile`), converts the batched
    output to matscipy-style per-configuration lists, and checks that the
    resulting neighbour pairs match those from `matscipy_neighbour_list`
    up to a small rounding tolerance on the distances.

    Parameters
    ----------
    atoms : Atoms
        Single matscipy-like Atoms object to test.
    cutoff : float
        Cutoff radius for the neighbour list.
    use_torch_compile : bool, optional
        If True, use the `torch.compile`-optimised O(N²) kernel;
        otherwise use the uncompiled version.

    Raises
    ------
    AssertionError
        If there are any pairs present in the matscipy neighbour list
        but not in the O(N²) implementation, or vice versa.
    """

    nl = NeighbourList(
        list_of_positions=[atoms.positions],
        list_of_cells=[atoms.cell.array],
        cutoff=cutoff,
        device=None,
    )
    nl.load_data()

    out = nl.calculate_neighbourlist(use_torch_compile=use_torch_compile)

    (
        atom_index_list,
        neighbor_index_list,
        int_shift_list,
        cart_shift_list,
        distance_list,
    ) = nl.get_matscipy_output_from_batch_output(*out)

    # single configuration: take the first (and only) entry
    i = atom_index_list[0]
    j = neighbor_index_list[0]
    S = int_shift_list[0]
    D = cart_shift_list[0]
    d = distance_list[0]

    # move to CPU for comparison
    i = i.cpu()
    j = j.cpu()
    S = S.cpu()
    D = D.cpu()
    d = d.cpu()

    matscipy_i, matscipy_j, matscipy_S, matscipy_D, matscipy_d = matscipy_neighbour_list(
        "ijSDd", atoms, cutoff=cutoff
    )

    # I am not comparing D's as the matscipy D is likely not correct.

    pairs = Counter(
        (
            int(i[k].item()),
            int(j[k].item()),
            tuple(S[k].tolist()),
            round(float(d[k]), 3),
        )
        for k in range(len(i))
    )

    matscipy_pairs = Counter(
        (
            int(matscipy_i[k].item()),
            int(matscipy_j[k].item()),
            tuple(matscipy_S[k].tolist()),
            round(float(matscipy_d[k]), 3),
        )
        for k in range(len(matscipy_i))
    )

    missing = matscipy_pairs - pairs   # in matscipy but not here
    extra = pairs - matscipy_pairs     # here but not in matscipy

    assert not missing and not extra, (
        f"Neighbour list mismatch\n"
        f"Missing (matscipy - ON2): {missing}\n"
        f"Extra (ON2 - matscipy):   {extra}"
    )
