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
    timestamp = "1694695415"
    history_files = list((THIS_DIR / "history" / timestamp).glob("history_*.json"))
    history_files = sorted(history_files)

    all_populations = []
    all_evals = []
    for i, file_ in enumerate(history_files):
        with open(file_, "r") as f:
            history = json.load(f)
            all_populations += history["x"]
            all_evals += history["y"]
    
    all_evals = np.array(all_evals)

    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(
        x=all_evals[:, 1],
        y=all_evals[:, 0],
        ax=ax,
        label="All populations",
    )
    # sns.scatterplot(
    #     x=y0[:, 1], y=y0[:, 0], ax=ax, label="Wildtype", c="red", marker="x"
    # )
    ax.set_xlabel("SASA")
    ax.set_ylabel("Stability")
    # fig.savefig(history_dir / "all_populations.png")
    # plt.close()

    plt.show()
