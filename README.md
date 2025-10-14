# Loopflopy

A small but powerful bridge between **LoopStructural** (3‑D geological modelling) and **MODFLOW 6** via **FloPy**. Loopflopy lets you:

* build groundwater flow models directly from geological models;
* generate **multiple structural realisations** (faults, folds, pinch‑outs, etc.)
* propagate **structural uncertainty** into flow predictions;
* discretise complex geometry with layered vertex (DISV) and unstructured (DISU) grids;
* assign properties (including anisotropy) from geology; and
* write, run and post‑process MODFLOW 6 simulations with minimal boilerplate.

> This code grew out of PhD work exploring the effect of structural assumptions on groundwater predictions. It’s still evolving—contributions and issue reports are very welcome!

---

## Table of contents

* [Key features](#key-features)
* [Install](#install)
* [Quick start](#quick-start)
* [Typical workflow](#typical-workflow)
* [Data inputs](#data-inputs)
* [Grids & discretisation](#grids--discretisation)
* [Property assignment](#property-assignment)
* [Boundary conditions & stresses](#boundary-conditions--stresses)
* [Running and post‑processing](#running-and-post-processing)
* [Examples](#examples)
* [Troubleshooting](#troubleshooting)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [Citing / acknowledgement](#citing--acknowledgement)
* [License](#license)

---

## Key features

* **Geology → Flow, end‑to‑end**: evaluate LoopStructural models on a mesh, construct hydrostratigraphic layers/sublayers, and write a ready‑to‑run MODFLOW 6 model.
* **Multiple structural realisations**: drive ensembles by varying **structural parameters** (e.g., control‑point elevation for pinch‑outs, fault displacement/ellipsoid, fold geometry) to propagate **Type‑3 (conceptual/structural) uncertainty** into flow predictions.
* **Unstructured, full‑connectivity flow**: build **DISV** meshes and convert to **DISU** with enhanced cell connectivity ("full connectivity") so offset/dipping layers remain hydraulically connected.
* **XT3D flow formulation**: activate **XT3D** by default for robust fluxes with irregular geometries and anisotropy.
* **Anisotropy aware**: convert LoopStructural scalar‑field **gradients** to NPF tensor angles (Angle1/Angle2) so K is aligned with bedding/dip/strike.
* **Property engines**: map lithology → K (scalar/tensor), storage, Sy; add simple heterogeneity hooks.
* **FloPy under the hood**: full access to FloPy model objects and writers.
* **ParaView exports**: VTK writers for mesh/results; uncertainty maps (e.g., σ(head)) across ensembles.

> **What loopflopy is not:** a geological modeller or a full uncertainty engine. It focuses on the glue between a geological realisation and a runnable MF6 model.

---

## Install

Loopflopy is packaged with a **PEP 621** `pyproject.toml` uses the **Hatchling** build backend, and a **`src/` layout**.

### Python & OS

* Python **≥ 3.10** (tested up to 3.12)
* Windows / Linux (macOS likely fine but less tested)

### Option 1 — Create a fresh environment (recommended)

```bash
# from the repo root
conda create -n loopflopy python=3.12 -y
conda activate loopflopy
# core package (minimal deps from [project.dependencies])
pip install  .
```

### Option 2 — Install with extras

Optional dependency groups defined in `pyproject.toml`:

* `gis`  – GeoPandas/Shapely/Fiona/Proj, raster IO & stats
* `viz-3d` – PyVista/VTK/Trimesh for 3‑D viewing
* `loop3d` – LoopStructural + helpers (shares PyVista/VTK)
* `excel` – Excel IO (openpyxl/xlrd)
* `notebook` – Jupyter/Lab/widgets
* `examples` – *one‑stop pack* = `gis` + `loop3d` + `viz-3d` + `excel` + `notebook` + matplotlib
* `dev`, `test`, `docs` – contributor toolchains

Install one or many groups, e.g.:

```bash
# everything needed to run notebooks in examples/
pip install -e '.[examples]'

# for contributors
pip install -e '.[dev,test]'
```

> Note: `viz-3d` pins `pyvista<0.47` and `vtk==9.3.*` because newer combos can break rendering. Keep these versions in sync.

### Option 3 — From Git (no local clone)

```bash
# install from the main repo
pip install "loopflopy @ git+https://github.com/kerrybardot/loopflopy.git"

# or from a fork (replace with your user/repo)
pip install "loopflopy @ git+https://github.com/<your-user>/loopflopy.git"
```
With extras over Git:

```bash
# examples extra directly from Git
pip install "loopflopy[examples] @ git+https://github.com/kerrybardot/loopflopy.git"

# or from your fork
pip install "loopflopy[examples] @ git+https://github.com/<your-user>/loopflopy.git"
```

### Build a wheel / sdist (for distribution)

```bash
pip install -U build
python -m build
# artifacts in dist/*.whl and dist/*.tar.gz (Hatchling backend as configured in pyproject.toml)
```

### External prerequisites

* (optional) **Triangle/Mesh tooling** if your workflow uses triangular meshing.

## Quick start

(Soon)Below is a minimal, end‑to‑end sketch. Exact class names and arguments may evolve—see the examples for a working script.

```python
from loopflopy.geomodel import Geomodel
from loopflopy.mesh import Mesh
from loopflopy.flowmodel import Flowmodel

```

---

## Typical workflow

1. **Prepare a LoopStructural model** (faults, folds, stratigraphy).
2. **Evaluate geology** over a 3‑D point cloud covering your domain; extract lithology codes and bedding/normal vectors.
3. **Derive model layers/sublayers** from lithology, handling onlaps/pinch‑outs.
4. **Create a mesh** (DISV or DISU) with optional local refinement.
5. **Map lithologies → hydraulic properties** (scalars or tensors) and optionally add heterogeneity.
6. **Assemble MODFLOW 6 packages** (NPF, IC, CHD/GHB/WEL/RCH/EVT, IMS, etc.).
7. **Run MF6** and **post‑process** heads/flows/budgets.
8. **Repeat for alternative structural parameters** to propagate uncertainty.

---

## Data inputs

* **LoopStructural model** (Python object) or a function that evaluates lithology and gradients at `xyz` points.
* **Spatial domain**: bounding box and resolution settings.
* **Lithology ↔ unit map**: integer codes for layers/rock types.
* **Hydraulic property table**: K, anisotropy ratios, storage; optional porosity/alpha for stresses like EVT if required.
* **Boundary/stress definitions**: lists/dicts consumable by FloPy writers (CHD, GHB, WEL, RCH, EVT…).

---

## Grids & discretisation

Loopflopy supports:

* **Structured** Cartesian (DIS) for simple cases.
* **Layered vertex grids (DISV)** for complex layering.
* **Unstructured (DISU)** conversion when geometry requires full flexibility (e.g., along faults or pinch‑outs).

Grid building utilities provide:

* target cell size and optional local refinement;
* vertical subdivision into sublayers; and
* safe handling of erosional truncations and no‑thickness units.

---

## Property assignment

* Map lithologies to scalar or tensor **hydraulic conductivity**.
* Optional **anisotropy orientation** from local bedding/gradients.
* Hooks for **stochastic fields** or simple kriging to infill sparse data.

---

## Boundary conditions & stresses

Convenience writers exist for common MF6 packages:

* **NPF/IC/IMS** (core numerics)
* **CHD**, **GHB**, **WEL**, **RCH**, **EVT** (stress packages)

You can always access the underlying FloPy model to customise further (add packages, change solver tolerances, etc.).

---

## Running and post‑processing

* Models are written to a working folder per realisation (e.g., `runs/real_###`).
* Use FloPy’s `Output` helpers to read heads, budgets, cell flows; optional VTK export for ParaView.
* Built‑in helpers to compute **ensemble statistics** (mean, std, percentiles) and to rasterise **uncertainty maps** of heads/fluxes.

**Performance note**: in the published synthetic case, regenerating the geomodel + grid + MF6 inputs was ~**10%** of the transient MF6 runtime on a laptop‑class machine; total MCMC (5 chains) completed on commodity hardware. Your mileage will vary with grid size, physics, and solver settings.

---

## Examples

See the `examples/` folder for runnable notebooks/scripts demonstrating:

* **Pinch‑out scenario** — control‑point elevation (CPz) varied across a plausible range; compare heads/drawdown and generate **σ(head)** maps across ~O(10²) realisations.
* **Fault scenario** — vary fault displacement (D) and observe head redistribution and connectivity changes across the offset aquitard/aquifer.
* **Inverse modelling (synthetic)** — joint estimation of structure (e.g., CPz) and hydraulic parameters using MCMC/ensemble methods; reproduce posterior plots and prediction envelopes.

> If examples are missing in your clone, also look at the separate `loopflopy_examples` repository.

---

## Troubleshooting

* **Geometry looks wrong** → Check CRS/units and the `extent`/`dz` used when evaluating the geology.
* **Zero‑thickness layers** → Enable sublayering and/or allow pinch‑outs when building layers.


Please open an issue with a minimal script and error text if you’re stuck.

---

## Roadmap

* More examples (including real‑world cases).
* More tests and CI coverage.
* Improved documentation.
* Publish in PyPI and conda-forge.

---

## Contributing

### Project layout

* Source code: `src/loopflopy/`
* Examples: `examples/`
* Tests: `tests/` (soon)

### Dev setup

```bash
conda create -n loopflopy-dev python=3.12 -y
conda activate loopflopy-dev
pip install -e .[dev,test]
pre-commit install   # enable formatting/linting on commit
```

Run tests:

```bash
pytest -q --disable-warnings --maxfail=1
```

### Typical flow

1. Fork and create a feature branch.
2. Add/adjust examples and unit tests if relevant(soon).
3. Run linters/tests locally (soon).
4. Open a PR with a clear description and motivation.

Please also use the issue tracker for bug reports and feature requests.

---

## Citing / acknowledgement

If loopflopy helps your work, please cite the paper describing this workflow and the core tools:

* **Bardot, K.**, Grose, L., Camargo, I., Pirot, G., Siade, A., Pigois, J.-P., Hampton, C., McCallum, J. (2025). *A seamless geo‑flow modelling workflow to tackle structural uncertainty using LoopStructural and MODFLOW*. Environmental Modelling & Software, 192, 106557. https://doi.org/10.1016/j.envsoft.2025.106557
* **LoopStructural**: Grose, L. et al. (2021) *LoopStructural 1.0: time‑aware geological modelling*. GMD.
* **FloPy/MODFLOW 6**: Bakker, M. et al. (2016) Groundwater; Langevin, C.D. et al. (2017) USGS T&M 6‑A55.

A BibTeX snippet (customise as needed):

```bibtex
@article{Bardot2025Loopflopy,
  title={A seamless geo-flow modelling workflow to tackle structural uncertainty using LoopStructural and MODFLOW},
  author={Bardot, Kerry and Grose, Lachlan and Camargo, Itsuo and Pirot, Guillaume and Siade, Adam and Pigois, Jon-Phillippe and Hampton, Clive and McCallum, James},
  journal={Environmental Modelling & Software},
  volume={192},
  pages={106557},
  year={2025},
  doi={10.1016/j.envsoft.2025.106557}
}
```

---

## Contact

Using Loopflopy? I'd love to hear from you—happy to help. [kerry.bardot@uwa.edu.au](mailto:kerry.bardot@uwa.edu.au)

---

## License

This project is released under an open‑source license (see `LICENSE` if present). If no license file is present, please open an issue so we can clarify permitted use.
