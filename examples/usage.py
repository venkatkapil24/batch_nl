import sys
import os

# Add project root to PYTHONPATH
sys.path.append('/Users/venkatkapil24/scratch/nl/torch-nlpp/')

from ase import Atoms
from neighbourlist import NeighbourList

cell = [
    [ 2.460394,  0.0,       0.0     ],
    [-1.26336,   2.044166,  0.0     ],
    [-0.139209, -0.407369,  6.809714]
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

nl = NeighbourList(
    list_of_configurations=[carbon],
    radius=3,
    batch_size=1,
    device='cpu'
)

nl.load_data()
out = nl.calculate_neighbourlist()

