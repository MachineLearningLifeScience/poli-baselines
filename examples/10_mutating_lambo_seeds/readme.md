# Mutating `lambo`'s seeds

In `lambo`, they bootstrap the optimization process by running random mutations of their initial pareto front.

In this experiment, we run a budget of 10k iterations of mutating _that_ pareto front instead. This is to compare against `lambo`'s Bayesian Optimization.

`proxy_rfp_seed_data.csv` shows said seeds, with only the `foldx` sequences. We start by copy-pasting all their `foldx` assets, allowing us to attach an initial wildtype to each sequence.
