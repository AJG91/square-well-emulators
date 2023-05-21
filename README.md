# Square Well Emulators

This repository contains code used to test and generate preliminary results in
*Efficient emulators for scattering using eigenvector continuation* ([arXiv:2301.05093][arXiv]).
It also contains code for the bound-state problem.

## Getting Started

* This project relies on `python=3.10`. It was not tested with different versions.
  To view the entire list of required packages, see `environment.yml`.
* Clone the repository to your local machine.
* Once you have `cd` into this repo, create a virtual environment (assuming you have `conda` installed) via
```bash
conda env create -f environment.yml
```
* Enter the virtual environment with `conda activate square-well-env`
* Install the `emulate` package in the repo root directory using `pip install -e .`
  (you only need the `-e` option if you intend to edit the source code in `emulate/`).


## Example

The two main classes for the KVP-based emulators are `KVP_emulator_scattering` for scattering and `KVP_emulator_bound` for bound states.

The code snippet below shows how the scattering emulator should be used:
```python
from emulator_scattering import KVP_emulator_scattering

# Setup
V_exact = ... # The potential depth where we want to make our prediciton
V_b = ...  # The potential used to train the emulator
ps, ws = ...   # The momentum and integration measure in units of inverse fm, corresponding to the potential mesh
k0 = ...   # The wave number where we are making the prediction. Proportional to the square root of the energy.

# Initialize object.
emu = KVP_emulator_scattering(k=k0, ps=ps, ws=ws)

# Train the emulator
emu.train(psi_b, V_b, V_exact)

# Predict phase shifts at validation parameter values using the emulator
emu_pred = emu.prediction(tau_b, nugget) # Emulator
```

The code snippet below shows how the bound state emulator should be used:
```python
from emulator_bound import KVP_emulator_bound

# Setup
V_exact = ... # The potential depth where we want to make our prediciton
Vr = ...  # The potential used to train the emulator
psi_b = ...  # The wave functions used to build the basis
T_b = ...  # The the kinetic energy for psi_b
ps, ws = ...   # The momentum and integration measure in units of inverse fm, corresponding to the potential mesh

# Initialize object.
KVP_emulator_bound(ps, ws, Vr)

# Predict phase shifts at validation parameter values using the emulator
emu_pred = emu.energy_pred(psi_b, T_b) # Emulator
```

## Citing this work

Please cite this work as follows:

```bibtex
Title: KVP emulators for the square well problem
Author: A.J. Garcia
Date: 05/18/2023
Code version: 2.0
Availabilty: https://github.com/AJG91/square-well-emulators
```

[arxiv]: https://arxiv.org/abs/2007.03635
