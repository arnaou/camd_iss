"""
CAMD - Computer-Aided Molecular Design for Solvent Screening

Modular package for molecular design with clean separation of concerns:
- molecules: Molecule representation and validation
- pure: Pure component property calculations  
- mixture: Mixture properties and thermodynamics
- data: Data loading and management
- camd: CAMD optimization and screening
"""

# Submodule imports
from . import molecules
from . import pure  
from . import mixture
from . import data
from . import camd

# Convenience imports for common functionality
from .camd import CAMD
from .mixture import unifac, unifac_equations, UNIFACCalculator
from .pure import PropertyCalculator, AntoineCalculator
from .data import load_unifac_data, DataExporter
from .molecules import StructuralValidator

__all__ = [
    # Submodules
    'molecules', 'pure', 'mixture', 'data', 'camd',
    # Main classes
    'CAMD', 'PropertyCalculator', 'AntoineCalculator', 'UNIFACCalculator', 
    'DataExporter', 'StructuralValidator',
    # Functions
    'unifac', 'unifac_wrapper', 'load_unifac_data'
]
__version__ = '0.1.0'