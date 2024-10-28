This folder includes an example in which we optimize the thermal stability of red fluorescent proteins (RFPs), measured using an additive version of RaSP, using `LaMBO2`.

As a pre-requisite, [we encourage you to set-up `poli` for RaSP](https://machinelearninglifescience.github.io/poli-docs/using_poli/objective_repository/RaSP.html).

It includes the following assets:
- Several PDB files for these RFPs, based on the Pareto front found by [LaMBO](https://arxiv.org/abs/2203.12742).
- The pre-computed seed data. It takes [the pool assets in LaMBO](https://github.com/samuelstanton/lambo/blob/main/lambo/assets/fpbase/proxy_rfp_seed_data.csv) and pre-computes the additive `RaSP` score for around 1500 mutations.
- a `simple_observer.py` script, which implements an observer inside `poli`.
- a `run.py` with two examples: using the default and modified hyperparameters for the solver.

We recommend running it inside the environment of `LaMBO2`, which you can find inside the `solvers` folder.

```bash
# From the root of the poli-baselines directory
pip install -e .[lambo2]
python examples/06_running_lambo2_on_rasp/run.py
```