# claude2/__init__.py

"""Claude 2 module for bruise detection project."""

# Import submodules to make them available when importing the package
from . import assets
from . import components
from . import pages
from .convert_svg import convert_svg_to_png

__all__ = [
    'app',
    'assets',
    'components',
    'pages',
    'convert_svg_to_png'
]
