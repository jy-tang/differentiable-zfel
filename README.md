# differentiable-zfel

Python code for 1D free-electron laser (FEL) simulation, developed toward a differentiable FEL modeling workflow.

## Overview

This repository builds on the SLAC `zfel` lineage (Fortran -> MATLAB -> Python) and keeps the core 1D SASE FEL physics model while moving the project toward gradient-based optimization and differentiable accelerator workflows.

Current code supports:

- 1D SASE FEL simulation in SI units
- configurable electron beam and undulator parameters
- optional tapered undulator profile via array-valued `unduK`
- seeded reproducibility through `random_seed`
- outputs for power growth, spectra, and final slice phase-space

## Installation

### Conda environment (recommended)

```bash
conda env create -f environment.yml
conda activate zfel-dev
```

### Local editable install

```bash
pip install -e .
```

## Quickstart

```python
from zfel import sase1d

sase_input = dict(
    npart=512,
    s_steps=200,
    z_steps=200,
    energy=4313.34e6,      # eV
    eSpread=0.0,
    emitN=1.2e-6,          # m-rad
    currentMax=3400,       # A
    beta=26,               # m
    unduPeriod=0.03,       # m
    unduK=3.5,             # scalar or array for tapering
    unduL=70,              # m
    radWavelength=None,    # uses resonance if None
    random_seed=31,
    particle_position=None,
    hist_rule="square-root",
    iopt="sase",
    P0=0,                  # seed power [W]
)

out = sase1d.sase(sase_input)

print("rho:", out["params"]["rho"])
print("final power [W]:", out["power_z"][-1])
print("spectrum shape:", out["spectrum"].shape)
```

## Main API

- `zfel.sase1d.sase(inp_struct)`: run a 1D SASE FEL simulation
- `zfel.sase1d.params_calc(...)`: compute derived FEL and beam parameters
- `zfel.fel.FEL_process_complex(...)`: core field-particle integration loop

See notebooks in `docs/examples/` for end-to-end studies.

## Differentiable Direction

This project is being developed as a differentiable 1D FEL code for optimization and inverse-design tasks. The immediate goal is to keep a validated physics baseline while enabling:

- differentiable objective functions on FEL outputs
- gradient-based tuning of undulator and beam controls
- integration with ML/optimization toolchains

## Origin

The baseline model traces back to work by Zhirong Huang (Fortran), later translated through MATLAB variants and then into Python by Xiao Zhang, with consistency checks against prior implementations.
