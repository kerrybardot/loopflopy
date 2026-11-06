from types import SimpleNamespace

import numpy as np
import pytest

from loopflopy import utils


@pytest.mark.parametrize(
    ("coordinates", "cell"),
    [
        ((0, 0, 0), 0),
        ((0, 0, 3), 3),
        ((0, 1, 0), 4),
        ((1, 0, 0), 12),
        ((2, 2, 3), 35),
    ],
)
def test_structured_indices_use_zero_based_layer_row_column_order(coordinates, cell):
    dimensions = {"nlay": 3, "nrow": 3, "ncol": 4}
    # Independent expected values prevent matching mistakes in both functions.
    assert utils.find_cellid(*coordinates, **dimensions) == cell
    assert utils.find_kji(cell, **dimensions) == coordinates


def test_disv_disu_mapping_skips_inactive_cells():
    model = SimpleNamespace(
        cellid_disv=np.arange(6).reshape(2, 3),
        cellid_disu=np.array([[0, -1, 1], [2, 3, -1]]),
    )
    assert utils.disvcell_to_disucell(model, 4) == 3
    assert utils.disucell_to_disvcell(model, 3) == 4
    assert utils.disvcell_to_layicpl(model, 4) == (1, 1)
    assert utils.disucell_to_layicpl(model, 3) == (1, 1)


def test_structured_coordinate_helpers():
    assert utils.x_to_col(19.9, delr=10) == 1
    assert utils.y_to_row(20.1, delc=10) == 2
    assert utils.z_to_lay(75.0, delz=10.0, zmax=100.0) == 2
    assert utils.xyz_to_discell(x=25, y=75, x0=0, y1=100, dx=10, dy=10) == (
        2,
        2,
        (25.0, 75.0),
    )


@pytest.mark.parametrize(
    ("layer", "cell", "expected"),
    [(0, 0, 9.0), (1, 0, 4.5), (0, 1, 7.5), (1, 1, 3.0)],
)
def test_layer_center_uses_top_or_previous_layer_bottom(layer, cell, expected):
    assert utils.lay_to_z(
        np.array([[8.0, 6.0], [1.0, 0.0]]),
        np.array([10.0, 9.0]),
        layer,
        icpl=cell,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("flows", "expected"),
    [
        ([], (0.0, 0.0)),
        ([(0, 0, -2.0), (1, 0, 3.0), (2, 0, -4.0), (3, 0, 0.0)], (-6.0, 3.0)),
    ],
)
def test_boundary_flow_totals_preserve_sign(flows, expected):
    assert utils.ch_flow(flows) == pytest.approx(expected)
