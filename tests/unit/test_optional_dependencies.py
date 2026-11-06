import subprocess
import sys
from pathlib import Path


def test_helpers_and_properties_import_without_optional_dependencies():
    # A fresh interpreter prevents already-imported GIS modules hiding errors.
    script = """
import importlib.abc
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"geopandas", "shapely", "fiona", "shapefile", "Krigger"}:
            raise ModuleNotFoundError(fullname, name=fullname)

sys.meta_path.insert(0, BlockOptional())
sys.path.insert(0, sys.argv[1])
from loopflopy.utils import find_kji
from loopflopy.geomodel import reshape_loop2mf
from loopflopy.properties import Properties
from loopflopy.properties2 import Properties as Properties2
import numpy as np
assert find_kji(0, 1, 1, 1) == (0, 0, 0)
assert reshape_loop2mf(np.array([1, 2]), 2, 1).tolist() == [[2], [1]]

# Constructors work without GIS/Krigger; methods request their own dependency.
properties = Properties()
properties2 = Properties2()
operations = [
    (lambda: properties.kriging(None, None), "Krigger"),
    (lambda: properties.stochasticfieldgeneration(None, None), "Krigger"),
    (lambda: properties2.kriging(None, None), "Krigger"),
    (lambda: properties2.stochasticfieldgeneration(None, None, None), "Krigger"),
    (lambda: properties.make_gdf(None, None, None, None), "geopandas"),
    (lambda: properties2.make_randomfield_3d(None, None, None, None, None, None), "geopandas"),
]
for operation, dependency in operations:
    try:
        operation()
    except ModuleNotFoundError as error:
        assert error.name == dependency, str(error)
    else:
        raise AssertionError(f"Expected a request for {dependency}")
"""
    source = Path(__file__).resolve().parents[2] / "src"
    result = subprocess.run(
        [sys.executable, "-c", script, str(source)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
