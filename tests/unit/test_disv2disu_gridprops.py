import numpy as np
import pytest

from loopflopy.disv2disu import Disv2Disu


def connections(gridprops):
    """Read public compressed rows, excluding their diagonal entries."""
    result = {}
    offset = 0
    for node, count in enumerate(gridprops["iac"]):
        assert gridprops["ja"][offset] == node
        for position in range(offset + 1, offset + count):
            neighbor = int(gridprops["ja"][position])
            assert (node, neighbor) not in result
            result[node, neighbor] = tuple(
                gridprops[key][position] for key in ("ihc", "cl12", "hwva", "angldegx")
            )
        offset += count
    assert offset == gridprops["nja"] == len(gridprops["ja"])
    return result


def test_default_idomain_produces_expected_layer_geometry(rectangular_grid):
    # Also regresses the array-to-int error when no idomain is supplied.
    props = Disv2Disu(**rectangular_grid).get_gridprops_disu6()
    assert props["nodes"] == 4
    np.testing.assert_array_equal(props["top"], [10, 10, 6, 6])
    np.testing.assert_array_equal(props["bot"], [6, 6, 0, 0])
    np.testing.assert_array_equal(props["area"], [6, 12, 6, 12])
    assert props["cell2d"] == [
        [0, 1.0, 1.5, 4, 0, 1, 4, 3],
        [1, 4.0, 1.5, 4, 1, 2, 5, 4],
        [2, 1.0, 1.5, 4, 0, 1, 4, 3],
        [3, 4.0, 1.5, 4, 1, 2, 5, 4],
    ]


def test_connections_match_hand_calculated_faces(rectangular_grid):
    props = Disv2Disu(**rectangular_grid).get_gridprops_disu6()
    np.testing.assert_array_equal(props["iac"], [3, 3, 3, 3])
    # Values: connection type, half-distance (m), face width/area, angle.
    expected = {
        (0, 1): (1, 1, 3, 0),
        (1, 0): (1, 2, 3, 180),
        (2, 3): (1, 1, 3, 0),
        (3, 2): (1, 2, 3, 180),
        (0, 2): (0, 2, 6, 0),
        (2, 0): (0, 3, 6, 0),
        (1, 3): (0, 2, 12, 0),
        (3, 1): (0, 3, 12, 0),
    }
    actual = connections(props)
    assert actual.keys() == expected.keys()
    for pair, values in expected.items():
        assert actual[pair] == pytest.approx(values)


def test_inactive_cell_is_omitted_and_remaining_nodes_are_renumbered(
    rectangular_grid,
):
    idomain = np.array([[1, 0], [1, 1]])
    original = idomain.copy()
    props = Disv2Disu(**rectangular_grid, disv_idomain=idomain).get_gridprops_disu6()
    np.testing.assert_array_equal(idomain, original)
    assert props["nodes"] == 3
    np.testing.assert_array_equal(props["top"], [10, 6, 6])
    np.testing.assert_array_equal(props["bot"], [6, 0, 0])
    np.testing.assert_array_equal(props["area"], [6, 6, 12])
    assert [cell[0] for cell in props["cell2d"]] == [0, 1, 2]
    assert [cell[1] for cell in props["cell2d"]] == [1, 1, 4]
    assert connections(props) == {
        (0, 1): pytest.approx((0, 2, 6, 0)),
        (1, 0): pytest.approx((0, 3, 6, 0)),
        (1, 2): pytest.approx((1, 1, 3, 0)),
        (2, 1): pytest.approx((1, 2, 3, 180)),
    }


def test_staggered_layers_connect_only_overlapping_neighbor_cells(
    rectangular_grid,
):
    rectangular_grid["staggered"] = True
    rectangular_grid["botm"] = np.array([[6.0, 4.0], [0.0, 0.0]])
    props = Disv2Disu(**rectangular_grid).get_gridprops_disu6()
    actual = connections(props)
    expected = {
        (0, 1): (2, 1, 3, 0),
        (1, 0): (2, 2, 3, 180),
        (1, 2): (2, 2, 3, 180),
        (2, 1): (2, 1, 3, 0),
        (2, 3): (2, 1, 3, 0),
        (3, 2): (2, 2, 3, 180),
        (0, 2): (0, 2, 6, 0),
        (2, 0): (0, 3, 6, 0),
        (1, 3): (0, 3, 12, 0),
        (3, 1): (0, 2, 12, 0),
    }
    assert actual.keys() == expected.keys()
    for pair, values in expected.items():
        assert actual[pair] == pytest.approx(values)
