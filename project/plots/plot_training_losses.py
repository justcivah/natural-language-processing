import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "figure.titlesize": 18,
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "grid.color": "#e4e4e4",
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = {
    "train": "#2166ac",
    "eval":  "#d6604d",
    "chrfpp":  "#4fd64d",
}

groups = {
    "t5-efficient-mini": [
        "models/google/t5-efficient-mini/student_a/losses.csv",
        "models/google/t5-efficient-mini/student_b/losses.csv",
        "models/google/t5-efficient-mini/student_ab/losses.csv",
    ],
    "t5-efficient-small": [
        "models/google/t5-efficient-small/student_a/losses.csv",
        "models/google/t5-efficient-small/student_b/losses.csv",
        "models/google/t5-efficient-small/student_ab/losses.csv",
    ],
    "t5-efficient-base": [
        "models/google/t5-efficient-base/student_a/losses.csv",
        "models/google/t5-efficient-base/student_b/losses.csv",
        "models/google/t5-efficient-base/student_ab/losses.csv",
    ],
    "mt5-small": [
        "models/google/mt5-small/student_a/losses.csv",
        "models/google/mt5-small/student_b/losses.csv",
        "models/google/mt5-small/student_ab/losses.csv",
    ],
    "mt5-base": [
        "models/google/mt5-base/student_a/losses.csv",
        "models/google/mt5-base/student_b/losses.csv",
        "models/google/mt5-base/student_ab/losses.csv",
    ],
}

domain_titles = ["Student A", "Student B", "Student AB"]

for model_name, paths in groups.items():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    for i, path in enumerate(paths):
        df = pd.read_csv(path)
        ax = axes[i]
        ax_r = ax.twinx()

        ax.plot(df["epoch"], df["loss"],
                color=COLORS["train"], linewidth=1.8, label="Train loss", zorder=3)
        ax.plot(df["epoch"], df["eval_loss"],
                color=COLORS["eval"], linewidth=1.8, linestyle="--", label="Eval loss", zorder=3)
        ax_r.plot(df["epoch"], df["eval_chrfpp"],
                color=COLORS["chrfpp"], linewidth=1.8, linestyle="-.", label="chrF++", zorder=3)

        ax.set_title(domain_titles[i], pad=6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss") if i == 0 else ax.set_ylabel("")
        ax_r.set_ylabel("chrF++" if i == 2 else "")
        ax_r.spines["top"].set_visible(False)
        ax_r.spines["right"].set_visible(True)
        ax_r.tick_params(axis="y", which="both", right=True, labelright=True)

    fig.suptitle(model_name, fontweight="bold", y=1.01)
    plt.tight_layout()
    output_name = f"plots/{model_name}_losses.pdf"
    plt.savefig(output_name, format="pdf")
    plt.close(fig)