"""Regression tests for ON2 neighbour list vs matscipy across diverse cells."""

from ase import Atoms
from ase.build import bulk
import pytest

from .utils import _check_neighbourlist_ON2_matches_matscipy as check_nl


def _make_carbon_oblique() -> Atoms:
    cell = [
        [2.460394, 0.0, 0.0],
        [-1.26336, 2.044166, 0.0],
        [-0.139209, -0.407369, 6.809714],
    ]
    positions = [
        [-0.03480225, -0.10184225, 1.70242850],
        [-0.10440675, -0.30552675, 5.10728550],
        [-0.05691216,  1.26093576, 1.70242850],
        [ 1.11473716,  0.37586124, 5.10728550],
    ]
    return Atoms(symbols="CCCC", positions=positions, cell=cell, pbc=True)


def _make_silicon_diamond_cubic() -> Atoms:
    return bulk("Si", "diamond", a=6.0, cubic=True)


def _make_silicon_diamond() -> Atoms:
    return bulk("Si", "diamond", a=6.0)


def _make_copper_fcc() -> Atoms:
    return bulk("Cu", "fcc", a=3.6)


def _make_silicon_bct() -> Atoms:
    return bulk("Si", "bct", a=6.0, c=3.0)


def _make_titanium_hcp() -> Atoms:
    return bulk("Ti", "hcp", a=2.94, c=4.64, orthorhombic=False)


def _make_bismuth_rhombohedral_alpha20() -> Atoms:
    return bulk("Bi", "rhombohedral", a=6.0, alpha=20.0)


def _make_bismuth_rhombohedral_alpha10() -> Atoms:
    return bulk("Bi", "rhombohedral", a=6.0, alpha=10.0)


def _make_sicu_rocksalt() -> Atoms:
    return bulk("SiCu", "rocksalt", a=6.0)


def _make_sifcu_fluorite() -> Atoms:
    return bulk("SiFCu", "fluorite", a=6.0)


def _make_cacr_p2_o7_triclinic() -> Atoms:
    cell = [
        [6.19330899, 0.0, 0.0],
        [2.4074486111396207, 6.149627748674982, 0.0],
        [0.2117993724186579, 1.0208820183960539, 7.305899571570074],
    ]
    positions = [
        [3.68954016, 5.03568186, 4.64369552],
        [5.12301681, 2.13482791, 2.66220405],
        [1.99411973, 0.94691001, 1.25068234],
        [6.81843724, 6.22359976, 6.05521724],
        [2.63005662, 4.16863452, 0.86090529],
        [6.18250036, 3.00187525, 6.44499428],
        [2.11497733, 1.98032773, 4.53610884],
        [6.69757964, 5.19018203, 2.76979073],
        [1.39215545, 2.94386142, 5.60917746],
        [7.42040152, 4.22664834, 1.69672212],
        [2.43224207, 5.4571615, 6.70305327],
        [6.3803149, 1.71334827, 0.6028463],
        [1.11265639, 1.50166318, 3.48760997],
        [7.69990058, 5.66884659, 3.8182896],
        [3.56971588, 5.20836551, 1.43673437],
        [5.2428411, 1.96214426, 5.8691652],
        [3.12282634, 2.72812741, 1.05450432],
        [5.68973063, 4.44238236, 6.25139525],
        [3.24868468, 2.83997522, 3.99842386],
        [5.56387229, 4.33053455, 3.30747571],
        [2.60835346, 0.74421609, 5.3236629],
        [6.20420351, 6.42629368, 1.98223667],
    ]
    # Ca (20) x2, Cr (24) x2, P (15) x4, O (8) x14
    numbers = [*[20] * 2, *[24] * 2, *[15] * 4, *[8] * 14]
    return Atoms(positions=positions, numbers=numbers, cell=cell, pbc=True)


STRUCTURE_FACTORIES = [
    _make_carbon_oblique,
    _make_silicon_diamond_cubic,
    _make_silicon_diamond,
    _make_copper_fcc,
    _make_silicon_bct,
    _make_titanium_hcp,
    _make_bismuth_rhombohedral_alpha20,
    _make_bismuth_rhombohedral_alpha10,
    _make_sicu_rocksalt,
    _make_sifcu_fluorite,
    _make_cacr_p2_o7_triclinic,
]


@pytest.mark.parametrize("make_atoms", STRUCTURE_FACTORIES)
@pytest.mark.parametrize("use_torch_compile", [False, True])
def test_on2_matches_matscipy_for_various_cells(make_atoms, use_torch_compile):
    atoms = make_atoms()
    check_nl(atoms, cutoff=3.0, use_torch_compile=use_torch_compile)

