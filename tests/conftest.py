"""Pytest configuration.

This file makes the project root importable in tests.

With this file present, pytest adds the directory containing this file
(``tests/``) to ``sys.path``.  However, the test files live one level
deeper (``tests/unit/``) and import ``src``, so we explicitly compute the
project root (two levels up from this file) and prepend it to ``sys.path``.

This lets any test in the project simply do::

    from src.conv import conv2d_forward

without ``sys.path`` hacking in individual test files. ``pytest.ini`` also
sets ``pythonpath = .`` for the same purpose; keeping both means the suite
still imports cleanly when collected directly by an IDE or with the
``pytest.ini`` missing.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
