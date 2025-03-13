"""
tests/__init__.py
-----------------
Pytest test package for SAM2 Lung Nodule Segmentation.

Ensures the project root is on sys.path so all modules are importable
without a ``pip install -e .`` when running:

    pytest tests/

All tests are self-contained and use synthetic data; no LUNA16 dataset is required.
"""

import sys
from pathlib import Path

# Add project root so `data`, `models`, `evaluation`, `training` are importable
_PROJECT_ROOT = Path(__file__).parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
