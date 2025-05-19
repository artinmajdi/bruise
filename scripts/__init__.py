# This file makes the scripts directory a Python package
import os
import sys

# Add the project root to the Python path for proper imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)