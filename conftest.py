"""Root conftest.py — adds the project root to sys.path so that
``from src.lexer import ...`` style imports work regardless of how pytest
is invoked.
"""
from __future__ import annotations

import sys
import os

# Ensure the project root is on sys.path so ``src`` is importable as a package.
sys.path.insert(0, os.path.dirname(__file__))
