"""Make `import _harness` work under pytest as well as `unittest discover`.

The suite uses unittest.TestCase classes and needs no pytest-specific
features; pytest will also collect and run them if it is installed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
