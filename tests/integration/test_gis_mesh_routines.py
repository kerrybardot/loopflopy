import numpy as np
import pytest

pytestmark = pytest.mark.gis


def test_resampling_preserves_rectangle_and_adds_regular_boundary_points():
    geometry = pytest.importorskip("shapely.geometry")
    pytest.importorskip("geopandas")
    from loopflopy.mesh_routines import resample_shapely_poly

    original = geometry.Polygon([(0, 0), (4, 0), (4, 2), (0, 2)])
    result = resample_shapely_poly(original, distance=1.0)
    assert result.is_valid
    assert result.equals(original)
    np.testing.assert_allclose(
        result.exterior.coords,
        [
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (4, 1),
            (4, 2),
            (3, 2),
            (2, 2),
            (1, 2),
            (0, 2),
            (0, 1),
            (0, 0),
        ],
    )
