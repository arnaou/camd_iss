"""sgreen: A package for green chemistry computer-aided molecular design."""

from .core import Compound, Compounds
from .data import Dataset

__version__ = "0.1.0"
__all__ = ["Compound", "Compounds", "Dataset"]