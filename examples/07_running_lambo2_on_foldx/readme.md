This folder includes an example in which we optimize the thermal stability two PDBs (DNJA1 and RFAH), measured using `foldx`, using `LaMBO2`.

As a pre-requisite, [we encourage you to set-up `poli` for `foldx`](https://machinelearninglifescience.github.io/poli-docs/using_poli/objective_repository/foldx_stability.html).

We recommend running it inside the environment of `LaMBO2`, which you can find inside the `solvers` folder.

```bash
# From the root of the poli-baselines directory
pip install -e .[lambo2]
python examples/07_running_lambo2_on_foldx/run.py
```