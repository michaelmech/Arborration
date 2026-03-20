# Arborration

Arborration is a Python library for multiclass generative classifiers built on top of tree-based anomaly detection models.

## Project layout

```text
.
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── src/
│   └── arborration/
│       ├── __init__.py
│       ├── __main__.py
│       └── classifiers.py
└── tests/
    └── test_imports.py
```

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

For development:

```bash
python -m pip install -r requirements-dev.txt
```

## Usage

```python
from arborration import PineForestGenerativeClassifier

clf = PineForestGenerativeClassifier(random_state=0)
```

You can also run the package directly to execute the bundled example:

```bash
python -m arborration
```
