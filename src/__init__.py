"""
CAMD - Computer-Aided Molecular Design for Solvent Screening

Includes UNIFAC calculation package for activity coefficient calculations in liquid mixtures.
"""

from .camd import CAMD
from .unifac_core import unifac
from .unifac_wrapper import unifac_wrapper
from .unifac_data_loader import load_unifac_data, get_available_groups, validate_mixture_groups

__all__ = [
    'CAMD',
    'unifac', 
    'unifac_wrapper', 
    'load_unifac_data', 
    'get_available_groups', 
    'validate_mixture_groups'
]
__version__ = '0.1.0'