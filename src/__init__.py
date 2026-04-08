"""
CAMD-ISS: Computer-Aided Molecular Design for Extractive Distillation

Modular package for entrainer design and screening using the Infinitely Sharp
Split (ISS) method with UNIFAC-based thermodynamic calculations.

Modules:
    molecules: Molecule representation, validation, and generation
    pure: Pure component property estimation (Constantinou-Gani)
    mixture: Mixture thermodynamics (UNIFAC activity coefficients, VLE, ISS)
    data: UNIFAC parameter loading and compatibility checking
    camd: CAMD optimization and screening engine
"""

from . import molecules
from . import pure
from . import mixture
from . import data
from . import camd

from .camd import CAMD
from .pure import PropertyCalculator, AntoineCalculator
from .data import load_unifac_data, DataExporter
from .molecules import StructuralValidator

__all__ = [
    'molecules', 'pure', 'mixture', 'data', 'camd',
    'CAMD', 'PropertyCalculator', 'AntoineCalculator',
    'DataExporter', 'StructuralValidator',
    'load_unifac_data',
]
__version__ = '0.1.0'