import pytest

from loopflopy.disv2disu import (
    area_of_polygon,
    deg_anglex,
    distance,
    distance_normal,
)


def test_polygon_area_preserves_orientation_and_handles_concavity():
    # An L shape: 3 x 3 square with the upper-right 2 x 2 square removed.
    x = [0, 3, 3, 1, 1, 0]
    y = [0, 0, 1, 1, 3, 3]
    assert area_of_polygon(x, y) == pytest.approx(5.0)
    assert area_of_polygon(x[::-1], y[::-1]) == pytest.approx(-5.0)


def test_distances_use_euclidean_length_and_perpendicular_projection():
    assert distance(1, 2, 4, 6) == pytest.approx(5.0)
    # The infinite line through (0, 0), (4, 4) has equation x = y.
    assert distance_normal(0, 2, 0, 0, 4, 4) == pytest.approx(2**0.5)
    assert distance_normal(0, 2, 4, 4, 0, 0) == pytest.approx(2**0.5)


@pytest.mark.parametrize(
    ("end", "expected"),
    [
        ((0, -1), 0.0),
        ((1, 0), 90.0),
        ((0, 1), 180.0),
        ((-1, 0), 270.0),
        ((-1, -1), 315.0),
    ],
)
def test_edge_angle_uses_outward_normal_convention(end, expected):
    assert deg_anglex(0, 0, *end) == pytest.approx(expected)
