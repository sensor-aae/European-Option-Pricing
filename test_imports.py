# Test whether all dependencies are installed correctly
import numpy as np
import scipy
import matplotlib.pyplot as plt


def test_imports():
    """Ensure required packages import correctly."""
    assert np.__version__
    assert scipy.__version__
    assert plt is not None
