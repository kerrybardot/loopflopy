import flopy
import numpy as np
import pytest

from loopflopy.disv2disu import Disv2Disu

pytestmark = pytest.mark.integration


def test_converted_grid_writes_and_loads_as_flopy_disu(rectangular_grid, tmp_path):
    props = Disv2Disu(**rectangular_grid).get_gridprops_disu6()
    simulation = flopy.mf6.MFSimulation(sim_name="grid", sim_ws=tmp_path)
    flopy.mf6.ModflowTdis(simulation)
    flopy.mf6.ModflowIms(simulation)
    model = flopy.mf6.ModflowGwf(simulation, modelname="grid")
    flopy.mf6.ModflowGwfdisu(model, **props)
    simulation.write_simulation(silent=True)

    loaded = flopy.mf6.MFSimulation.load(sim_ws=tmp_path, verbosity_level=0)
    disu = loaded.get_model("grid").disu
    assert disu.nodes.get_data() == 4
    # FloPy must serialize one-based IDs and recover these zero-based rows.
    np.testing.assert_array_equal(disu.iac.array, [3, 3, 3, 3])
    np.testing.assert_array_equal(disu.ja.array, [0, 1, 2, 1, 0, 3, 2, 0, 3, 3, 1, 2])
    np.testing.assert_allclose(disu.area.array, [6, 12, 6, 12])
    np.testing.assert_allclose(disu.top.array, [10, 10, 6, 6])
    np.testing.assert_allclose(disu.bot.array, [6, 6, 0, 0])
