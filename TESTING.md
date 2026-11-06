# Testing

Install the core package and test tools:

```bash
pip install -e ".[test]"
python -m pytest -q
```

The default run selects fast unit tests. Pytest configuration in
`pyproject.toml` limits discovery to `tests/`, adds `src/` to the import path,
uses importlib imports, and rejects unknown markers and configuration options.

## Scope

- `tests/unit/` checks polygon geometry, edge angles, zero-based indexing,
  structured coordinates, DISV/DISU cell mapping, layer order, surface and
  transect angles, and signed flow totals.
- Small DISV-to-DISU converter tests compare public outputs against
  hand-calculated connections, distances, face geometry, and cell elevations.
  They cover inactive cells and staggered layer overlap.
- An import-isolation test starts a fresh Python process with GIS packages and
  `Krigger` blocked. It checks that numerical helpers and property constructors
  remain available, and that methods request their optional dependency when used.
- `tests/integration/test_flopy_disu.py` writes a converted grid through FloPy
  into `tmp_path` and loads it back, checking connectivity IDs and geometry.
  It does not run MODFLOW or establish groundwater-flow correctness.
- `tests/integration/test_gis_mesh_routines.py` checks polygon preservation and
  actual resampled boundary coordinates, beyond just matching bounds.

Regression tests cover zero-based cell decoding, default DISV cell numbering,
the zero-degree convention for a horizontal surface's undefined dip direction,
and dip directions for small-magnitude normal vectors.

Fixtures are tiny and deterministic. Tests use public functions and output
properties rather than private implementation state. No tests write to example
data. A duplicate large-grid consistency test is omitted; add a scale test when
there is a specific scaling risk or regression to protect.

Kriging methods in `properties` and `properties2` require the external `Krigger`
module to be installed or available on `PYTHONPATH`. It is imported when a kriging
method is called; no machine-specific search path is added by the package.

## Selecting tests

```bash
# Unit tests (also the default selection)
python -m pytest tests/unit -q

# FloPy integration, using core dependencies
python -m pytest -q -m integration

# Optional GIS test: only these GIS packages are needed by the current case
pip install "geopandas>=1.0,<2" "shapely>=2.0,<3"
python -m pytest -q -m gis

# All current tests, after installing the GIS dependencies
python -m pytest -q -m "not slow"

# Coverage, without a threshold
python -m pytest --cov=loopflopy --cov-report=term-missing
```

An explicit `-m` overrides the default marker selection. GIS imports occur
inside the GIS test and skip if its optional packages are absent, so collecting
the default suite does not require them.

Register external-library or executable workflow tests as `integration`, GIS
tests as `gis`, and tests with significant measured runtime as `slow`. The
default selection excludes these three markers. Current tests need no `slow`
marker. Avoid adding coverage targets or elaborate fixtures without a concrete
correctness or maintenance benefit.

## CI

`.github/workflows/tests.yml` runs unit and FloPy integration tests on Python
3.10, 3.11, and 3.12 with core/test dependencies. A separate Python 3.12 job
installs the current GIS test's dependencies and runs that test explicitly.
CI checks that the GIS packages import before testing, preventing a missing
dependency from silently turning that job into a skipped test.

## Local verification

Validation uses the local `loopflopy` Conda environment (Python 3.12).
In restricted environments, pytest's default temporary directory may be
inaccessible. Pass `--basetemp` pointing to a **new, disposable directory** inside
the workspace. Pytest clears the directory supplied to this option.

Remote CI and actual MODFLOW execution require separate validation.
