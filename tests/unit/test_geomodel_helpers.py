import numpy as np
import pytest

from loopflopy.geomodel import (
    find_angle1,
    find_angle1_transect,
    find_angle2,
    find_angle2_transect,
    reshape_loop2mf,
)


@pytest.mark.parametrize(
    ("normal", "direction", "dip"),
    [
        ([0.0, 0.0, 1.0], 0.0, 0.0),
        ([1.0, 0.0, 1.0], 180.0, 45.0),
        ([0.0, 1.0, 1.0], 90.0, 45.0),
        ([0.0, -1.0, 1.0], -90.0, 45.0),
        ([1e-10, 0.0, 1e-10], 180.0, 45.0),
    ],
)
def test_surface_normal_angles(normal, direction, dip):
    assert find_angle1(normal) == pytest.approx(direction)
    assert find_angle2(normal) == pytest.approx(dip)


@pytest.mark.parametrize(
    ("normal", "rotation", "dip"),
    [
        ([0.0, 0.0, 1.0], 37.5, 0.0),
        ([1.0, 0.0, 1.0], 0.0, -45.0),
        ([0.0, 1.0, 1.0], 45.0, 45.0),
    ],
)
def test_transect_angles_use_rotation_and_projected_normal(normal, rotation, dip):
    assert find_angle1_transect(normal, rotation) == pytest.approx(rotation)
    assert find_angle2_transect(normal, rotation) == pytest.approx(dip)


def test_loop_layers_are_reversed_without_reordering_cells_or_mutating_input():
    values = np.array([10, 11, 20, 21, 30, 31])
    result = reshape_loop2mf(values, nlay=3, ncpl=2)
    np.testing.assert_array_equal(result, [[30, 31], [20, 21], [10, 11]])
    np.testing.assert_array_equal(values, [10, 11, 20, 21, 30, 31])
