# Bruise Detection

A data science and machine learning framework for nursing research focused on bruise detection.

## Installation

### Using UV

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
uv sync
```

### Using Pip

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

## Optional ML Dependencies

To install TensorFlow and Keras:

```bash
pip install -e ".[ml]"
```
