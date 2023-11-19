"""
This is the curved_integration package init.
"""
from .version import version as __version__
from .surf_integration import *
from .remesh import *
from .quadrature_points import *
from .quadrature_points_gl import *
from .utils import *

# Define __all__ after importing the necessary modules
__all__ = ["__version__"] + surf_integration.__all__ + remesh.__all__ + quadrature_points.__all__ + utils.__all__ + quadrature_points_gl.__all__
