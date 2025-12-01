# tests/test_carbon_neighbourlists.py
from ase import Atoms
from .utils import (
    _check_neighbourlist_ON1_matches_ase,
    _check_neighbourlist_ON2_matches_ase,
)


def _make_carbon():
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
    return Atoms(symbols="CCCC", positions=positions, cell=cell, pbc=True)

def test_carbon_A2p46_B2p40_C6p82_a92p30_b91p20_g121p20_ON2():
    carbon = _make_carbon()
    _check_neighbourlist_ON2_matches_ase(carbon, radius=3.0)

