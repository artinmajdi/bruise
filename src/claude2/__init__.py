# claude2/__init__.py

"""Claude 2 module for bruise detection project."""

# Import submodules to make them available when importing the package
from . import assets
from . import components
from . import pages

__all__ = [
    'app',
    'assets',
    'components',
    'pages',
    'convert_svg'
]
