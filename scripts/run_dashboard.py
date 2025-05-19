#!/usr/bin/env python
"""
Launcher script for the Bruise Detection Streamlit dashboards.
Searches for app.py files in all modules inside the src folder.
Allows user to select which dashboard to run from available options.

Usage:
  python run_dashboard.py         # Interactive mode, prompts for dashboard selection
  python run_dashboard.py <num>   # Directly select dashboard by number
"""

import os
import sys
import subprocess
import argparse
from typing import List, Tuple, Dict

def find_dashboard_apps(src_dir: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Find all app.py files in all modules inside the src directory.
    Returns a dictionary with module names as keys and lists of tuples (dashboard_name, app_path) as values.
    """
    dashboard_apps = {}
    skip_items = {'.DS_Store', '__pycache__'}

    # First level: check all directories in src
    try:
        for module_name in os.listdir(src_dir):
            module_path = os.path.join(src_dir, module_name)

            # Skip files and hidden directories
            if not os.path.isdir(module_path) or module_name.startswith('__') or module_name.startswith('.') or module_name in skip_items:
                continue

            module_apps = []

            # Check for app.py in the module root
            app_path = os.path.join(module_path, 'app.py')
            if os.path.isfile(app_path):
                dashboard_name = f"{module_name}/app"
                module_apps.append((dashboard_name, app_path))

            # Check subdirectories for app.py files
            for subdir in os.listdir(module_path):
                subdir_path = os.path.join(module_path, subdir)

                if not os.path.isdir(subdir_path) or subdir.startswith('__') or subdir.startswith('.') or subdir in skip_items:
                    continue

                # Check for app.py in subdirectory
                subdir_app_path = os.path.join(subdir_path, 'app.py')
                if os.path.isfile(subdir_app_path):
                    dashboard_name = f"{module_name}/{subdir}/app"
                    module_apps.append((dashboard_name, subdir_app_path))

            if module_apps:
                dashboard_apps[module_name] = module_apps
    except Exception as e:
        print(f"Error searching for dashboards: {e}")

    return dashboard_apps

def select_dashboard(all_apps: List[Tuple[str, str]]) -> Tuple[str, str]:
    """
    Get user selection from the list of available dashboards.
    Returns the selected (dashboard_name, app_path).
    """
    if not all_apps:
        print("No dashboards found in src directory!")
        sys.exit(1)

    while True:
        try:
            choice = input("\nSelect a dashboard number to run: ")
            idx = int(choice) - 1
            if 0 <= idx < len(all_apps):
                return all_apps[idx]
            print(f"Invalid selection. Please enter a number between 1 and {len(all_apps)}.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run a Streamlit dashboard from the src directory')
    parser.add_argument('dashboard_number', nargs='?', type=int, help='Dashboard number to run (optional)')
    args = parser.parse_args()

    # Get the project root directory (one level up from scripts directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    src_dir = os.path.join(project_root, 'src')

    # Make sure the project root is in the Python path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Print environment info
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Source directory: {src_dir}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")

    # Find available dashboards in all modules
    all_dashboard_apps = find_dashboard_apps(src_dir)

    if not all_dashboard_apps:
        print("No dashboards found in any module in the src directory. Exiting.")
        sys.exit(1)

    # Flatten the dictionary into a list for easier selection
    all_apps = []
    for module_name, apps in all_dashboard_apps.items():
        all_apps.extend(apps)

    # Display available dashboards
    print("\nAvailable Dashboards:")
    for idx, (name, path) in enumerate(all_apps, 1):
        print(f"{idx}. {name}")

    # Select dashboard to run
    if args.dashboard_number is not None:
        # Use command-line argument if provided
        if 1 <= args.dashboard_number <= len(all_apps):
            selected_name, app_path = all_apps[args.dashboard_number - 1]
            print(f"\nSelected dashboard {args.dashboard_number}: {selected_name}")
        else:
            print(f"Error: Invalid dashboard number. Please select a number between 1 and {len(all_apps)}.")
            sys.exit(1)
    else:
        # Interactive selection
        try:
            selected_name, app_path = select_dashboard(all_apps)
        except (EOFError, KeyboardInterrupt):
            print("\nDashboard selection cancelled. Exiting.")
            sys.exit(1)
    print(f"\nSelected dashboard: {selected_name} ({app_path})")

    # Setup environment for the subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"

    # Check if a virtual environment exists and use it if available
    venv_path = os.path.join(project_root, '.venv')
    if os.path.isdir(venv_path):
        print(f"\nUsing virtual environment at: {venv_path}")
        if os.name == 'nt':  # Windows
            python_executable = os.path.join(venv_path, 'Scripts', 'python.exe')
        else:  # Unix/Mac
            python_executable = os.path.join(venv_path, 'bin', 'python')

        if os.path.exists(python_executable):
            print(f"Using Python executable from virtual environment: {python_executable}")
            streamlit_executable = os.path.join(os.path.dirname(python_executable), 'streamlit')
            streamlit_cmd = [streamlit_executable]
        else:
            print(f"Virtual environment found but Python executable not found at {python_executable}")
            print("Falling back to system Python and Streamlit")
            streamlit_cmd = ["streamlit"]
    else:
        print("No virtual environment found. Using system Python and Streamlit.")
        streamlit_cmd = ["streamlit"]

    # Run Streamlit
    print(f"\nStarting {selected_name} dashboard with: {app_path}")
    try:
        # Construct the full command
        cmd = streamlit_cmd + ["run", app_path]
        print(f"Running command: {' '.join(cmd)}")

        # Run the streamlit app
        result = subprocess.run(
            cmd,
            env=env,
            check=True
        )
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        return e.returncode
    except FileNotFoundError:
        print("Streamlit not found. Attempting to install it...")

        # Determine which pip to use
        if os.path.isdir(venv_path):
            if os.name == 'nt':  # Windows
                pip_executable = os.path.join(venv_path, 'Scripts', 'pip.exe')
            else:  # Unix/Mac
                pip_executable = os.path.join(venv_path, 'bin', 'pip')
        else:
            pip_executable = 'pip'

        try:
            print(f"Running: {pip_executable} install streamlit")
            subprocess.run([pip_executable, "install", "streamlit"], check=True)
            print("Streamlit installed successfully. Trying to run the dashboard again...")

            # Try running streamlit again
            if os.path.isdir(venv_path):
                if os.name == 'nt':  # Windows
                    streamlit_executable = os.path.join(venv_path, 'Scripts', 'streamlit.exe')
                else:  # Unix/Mac
                    streamlit_executable = os.path.join(venv_path, 'bin', 'streamlit')
                cmd = [streamlit_executable, "run", app_path]
            else:
                cmd = ["streamlit", "run", app_path]

            return subprocess.run(cmd, env=env, check=True).returncode
        except subprocess.CalledProcessError as e:
            print(f"Error installing Streamlit: {e}")
            print("Please install Streamlit manually with: pip install streamlit")
            return 1
        except FileNotFoundError:
            print("Could not find pip. Please install Streamlit manually with: pip install streamlit")
            return 1

if __name__ == "__main__":
    sys.exit(main())
