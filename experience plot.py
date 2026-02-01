import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV_PATH = "mnist_bartlett_experiment.csv"  
df = pd.read_csv(CSV_PATH)

widths = sorted(df["width"].unique())
wds = sorted(df["weight_decay"].unique())

# marqueurs par largeur
marker_map = {16: "o", 64: "s", 256: "^", 1024: "D"}
for w in widths:
    marker_map.setdefault(w, "o")

# couleurs par weight_decay
cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_map = {wd: cycle[i % len(cycle)] for i, wd in enumerate(wds)}

def add_dual_legend(ax):
    wd_handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=color_map[wd], markeredgecolor=color_map[wd],
               markersize=8, label=f"wd={wd:g}")
        for wd in wds
    ]
    width_handles = [
        Line2D([0], [0], marker=marker_map[w], linestyle="None",
               color="black", markersize=8, label=f"width={w}")
        for w in widths
    ]
    leg1 = ax.legend(handles=wd_handles, title="Weight decay (couleur)",
                     loc="upper right", frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=width_handles, title="Largeur (marqueur)",
              loc="lower left", frameon=True)

# ---------------------------
# Plot 1: test error vs weight norm
# ---------------------------
plt.figure()
ax = plt.gca()

for wd in wds:
    for w in widths:
        sub = df[(df["weight_decay"] == wd) & (df["width"] == w)]
        if len(sub) == 0:
            continue
        ax.scatter(sub["weight_norm"], sub["test_err01"],
                   marker=marker_map[w],
                   color=color_map[wd],
                   alpha=0.75, s=55)

ax.set_title("Generalization vs weight norm (MNIST 3 vs 5)")
ax.set_xlabel("Weight norm (||W1||_F + ||W2||_F)")
ax.set_ylabel("Test error (0-1)")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
add_dual_legend(ax)
plt.tight_layout()
plt.savefig("plot1_test_error_vs_weight_norm.png", dpi=250)

# ---------------------------
# Plot 2: test error vs train margin q10
# ---------------------------
plt.figure()
ax = plt.gca()

for wd in wds:
    for w in widths:
        sub = df[(df["weight_decay"] == wd) & (df["width"] == w)]
        if len(sub) == 0:
            continue
        ax.scatter(sub["train_margin_q10"], sub["test_err01"],
                   marker=marker_map[w],
                   color=color_map[wd],
                   alpha=0.75, s=55)

ax.set_title("Generalization vs margin (MNIST 3 vs 5)")
ax.set_xlabel("Train margin q10 (10th percentile of y*f(x))")
ax.set_ylabel("Test error (0-1)")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
add_dual_legend(ax)
plt.tight_layout()
plt.savefig("plot2_test_error_vs_train_margin_q10.png", dpi=250)

print("Saved:")
print(" - plot1_test_error_vs_weight_norm.png")
print(" - plot2_test_error_vs_train_margin_q10.png")
