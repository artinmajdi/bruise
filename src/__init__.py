# src/__init__.py

"""Bruise detection project package."""

# Import submodules to make them available when importing the package
from . import claude1
from . import claude2
from . import gemini
from . import gemini2


__all__ = [
    'claude1',
    'claude2',
    'gemini',
    'gemini2'
]
