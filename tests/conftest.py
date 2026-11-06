import numpy as np
import pytest


@pytest.fixture
def rectangular_grid():
    """Two adjacent clockwise cells, 2 x 3 m and 4 x 3 m, in two layers."""
    vertices = [
        [0, 0.0, 3.0],
        [1, 2.0, 3.0],
        [2, 6.0, 3.0],
        [3, 0.0, 0.0],
        [4, 2.0, 0.0],
        [5, 6.0, 0.0],
    ]
    cell2d = [
        [0, 1.0, 1.5, 4, 0, 1, 4, 3],
        [1, 4.0, 1.5, 4, 1, 2, 5, 4],
    ]
    return {
        "vertices": vertices,
        "cell2d": cell2d,
        "top": np.array([10.0, 10.0]),
        "botm": np.array([[6.0, 6.0], [0.0, 0.0]]),
        "staggered": False,
    }
