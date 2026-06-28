"""
Quantum circuit execution cost per epoch vs training set size.
Plots the ALIGNMENT-STEP cost (kernel evaluations driven by each method's
gradient objective) -- the quantity that distinguishes the methods.

Complexity classes:
  FullKTA   : O(N²)       -- full N×N kernel matrix per gradient step
  GreedyKTA : O(N·M)      -- Nyström (M landmarks) for uncertainty scoring
  QuackKTA  : O(N)        -- centroid-to-full-dataset kernel vector, C classes
  RandomKTA : O(k²)       -- random mini-batch of k points, constant w.r.t. N
  CCKA      : O(C·K)      -- centroid-to-sub-centroid kernel vector only
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Hyperparameters matching the experimental setup ────────────────────────
C      = 2    # number of classes (binary)
M      = 10   # Nyström landmark points (GreedyKTA)
inner  = 10   # inner gradient steps per outer epoch (CCKA / QuackKTA)

N = np.linspace(10, 10_000, 100)

k = N / 4     # random mini-batch size = N/4 (scales with dataset)
K = N / 4     # sub-centroids per class = N/4 (scales with dataset)

# ── Per-epoch alignment-step circuit executions ────────────────────────────
cost = {
    "FullKTA":    N * (N + 1) / 2,
    "GreedyKTA":  N * M + M * M,
    "QuackKTA":   2 * inner * C * N,
    "RandomKTA":  k * (k + 1) / 2,           # k = N/4  -> O(N²/16)
    "CCKA (ours)":2 * inner * C * K,          # K = N/4  -> O(C·N/4)
}

# Complexity label for the legend
complexity = {
    "FullKTA":     r"$\mathcal{O}(N^2)$",
    "GreedyKTA":   r"$\mathcal{O}(N \cdot M)$, $M{=}10$",
    "QuackKTA":    r"$\mathcal{O}(N)$",
    "RandomKTA":   r"$\mathcal{O}(N^2/16)$, $k{=}N/4$",
    "CCKA (ours)": r"$\mathcal{O}(C \cdot N/4)$, $K{=}N/4$",
}

colors = {
    "FullKTA":     "#185FA5",
    "GreedyKTA":   "#BA7517",
    "QuackKTA":    "#0F6E56",
    "RandomKTA":   "#639922",
    "CCKA (ours)": "#D85A30",
}

lws = {m: 2.8 if m == "CCKA (ours)" else 1.8 for m in cost}
lss = {m: "-"  if m == "CCKA (ours)" else "--" for m in cost}

# ── Figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

for method, y in cost.items():
    label = f"{method}  {complexity[method]}"
    ax.plot(N, y,
            color=colors[method],
            lw=lws[method],
            ls=lss[method],
            label=label,
            zorder=5 if method == "CCKA (ours)" else 3)

ax.set_yscale("log")
ax.set_xlabel("Training samples  $N$", fontsize=12)
ax.set_ylabel("Quantum circuit executions\nper epoch (alignment step)", fontsize=11)
ax.set_title("Alignment-step cost vs training set size", fontsize=12, pad=8)

ax.set_xlim(0, 10_000)
ax.set_ylim(1e1, 1e8)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(True, which="both", alpha=0.25, lw=0.5)
ax.grid(True, which="major", alpha=0.35, lw=0.8)

# Annotate the two extreme methods at N=10000
ax.annotate("FullKTA",
            xy=(9800, cost["FullKTA"][-1]),
            xytext=(7200, 2e7),
            fontsize=8.5, color=colors["FullKTA"],
            arrowprops=dict(arrowstyle="->", color=colors["FullKTA"], lw=1.0),
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec=colors["FullKTA"], alpha=0.85))

ax.annotate("CCKA (ours)",
            xy=(9800, cost["CCKA (ours)"][-1]),
            xytext=(6500, 35),
            fontsize=8.5, color=colors["CCKA (ours)"],
            arrowprops=dict(arrowstyle="->", color=colors["CCKA (ours)"], lw=1.0),
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec=colors["CCKA (ours)"], alpha=0.85))

ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95,
          edgecolor="lightgray", handlelength=2.2)

plt.tight_layout()
plt.savefig("complexity_comparison.pdf", dpi=300, bbox_inches="tight")
plt.savefig("complexity_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: complexity_comparison.pdf / .png")