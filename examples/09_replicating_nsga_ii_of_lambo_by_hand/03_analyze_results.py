from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

THIS_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    # We will load the history of the optimization
    # from the json files.
    timestamp = "1698413727"
    history_files = list((THIS_DIR / "history" / timestamp).glob("history_*.json"))
    history_files = sorted(history_files)

    all_populations = []
    all_evals = []
    for i, file_ in enumerate(history_files):
        with open(file_, "r") as f:
            history = json.load(f)
            all_populations += history["x"]
            all_evals += history["y"]

    all_evals = -np.array(all_evals)

    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(
        x=all_evals[:, 1],
        y=all_evals[:, 0],
        ax=ax,
        label="All populations",
    )
    ax.set_title(f"All populations (n={len(all_evals)})")

    with open(THIS_DIR / "history" / timestamp / "wildtype_scores.json") as fp:
        wildtype_scores = json.load(fp)

    wildtype_scores = np.array(wildtype_scores["y"])
    sns.scatterplot(
        x=wildtype_scores[:, 1],
        y=wildtype_scores[:, 0],
        ax=ax,
        label="Wildtype",
        c="red",
        marker="x",
    )
    ax.set_xlabel("SASA")
    ax.set_ylabel("Stability")

    plt.show()
