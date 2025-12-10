"""
batch_nl: PyTorch-based neighbour list utilities.

Main entry point:
- NeighbourList: batched neighbour-list builder for periodic systems.
"""

from .neighbourlist import NeighbourList

__all__ = ["NeighbourList"]
