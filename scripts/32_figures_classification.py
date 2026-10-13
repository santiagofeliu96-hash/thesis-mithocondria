"""Figuras del capítulo de clasificación de instancias (conjunto de desarrollo salvo indicación).

  feature_correlation_heatmap_modl.png      fig:res2-correlacion
  modl_boxplots_morfo_3clases.png           fig:res2-boxplots-morfo
  modl_boxplots_intensidad_3clases.png      fig:res2-boxplots-intensidad
  modl_scatter_real_fp.png                  fig:res2-scatter-fp
  modl_scatter_indiv_cluster.png            fig:res2-scatter-ic
  pr_frontier_{modl,omnipose}.png           fig:res2-frontera
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from tesislib.cascade import make_model, operating_point
from tesislib.features import FEATURE_GROUPS, FINAL_FEATURES, INT_FEATURES

CLASSES = ["individual", "cluster", "spurious"]
COL = {"individual": "#1f77b4", "cluster": "#d62728", "spurious": "#7f7f7f"}
DISCRETE = {"n_holes", "n_branch_points", "n_endpoints", "euler_number"}
TARGET_COLORS = ["#7a1f1f", "#c23b3b", "#e07b39", "#d9a441"]


def out(name):
    return os.path.join(C.FIGURES, name)


def dev_of(tool):
    df = pd.read_csv(os.path.join(C.TABLES, f"all_features_{tool}.csv"))
    return df[~df.film.isin(C.TEST_FILMS)]


def heatmap(dev):
    groups = ["size", "shape", "topology", "branching"]
    feats = [f for g in groups for f in FEATURE_GROUPS[g]]
    corr = dev[feats].corr().abs()
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    im = ax.imshow(corr.values, vmin=0, vmax=1, cmap="magma")
    labels = [f + "  [*]" if f in FINAL_FEATURES else f for f in feats]
    ax.set_xticks(range(len(feats))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(labels, fontsize=7)
    for i in range(len(feats)):
        for j in range(len(feats)):
            if i != j and corr.values[i, j] > 0.9:
                ax.plot(j, i, ".", color="cyan", markersize=3)
    for b in np.cumsum([len(FEATURE_GROUPS[g]) for g in groups])[:-1]:
        ax.axhline(b - 0.5, color="white", lw=0.8); ax.axvline(b - 0.5, color="white", lw=0.8)
    ax.set_title(f"MoDL (n={len(dev)})", fontsize=10)
    fig.suptitle("Correlación |Pearson| de los 21 descriptores candidatos\n"
                 "[*] = conservado en el conjunto reducido (10)   ·   punto cyan = |r| > 0.9", fontsize=11)
    fig.colorbar(im, ax=[ax], fraction=0.025, pad=0.02).set_label("|r|")
    fig.savefig(out("feature_correlation_heatmap_modl.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


def boxplots(dev, feats, kind, name):
    dev = dev[dev.op_label.isin(CLASSES)]
    groups = {c: dev[dev.op_label == c] for c in CLASSES}
    ncol = 5
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.2 * nrow))
    rng = np.random.default_rng(C.SEED)
    for ax, f in zip(axes.ravel(), feats):
        data = [groups[c][f].dropna().values for c in CLASSES]
        bp = ax.boxplot(data, positions=[1, 2, 3], widths=0.6, showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], CLASSES):
            patch.set_facecolor(COL[c]); patch.set_alpha(0.35)
        for pos, c, v in zip([1, 2, 3], CLASSES, data):
            ax.scatter(pos + rng.uniform(-0.14, 0.14, len(v)), v, s=6, color=COL[c], alpha=0.5,
                       edgecolors="none")
        ax.set_title(f, fontsize=9)
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["ind", "clu", "FP"], fontsize=8)
    for ax in axes.ravel()[len(feats):]:
        ax.set_axis_off()
    n = {c: len(groups[c]) for c in CLASSES}
    fig.suptitle(f"MoDL: {kind} por clase  (ind={n['individual']}, clu={n['cluster']}, FP={n['spurious']})",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(out(name), dpi=130, bbox_inches="tight"); plt.close(fig)


def scatter(df, fa, fb, ypos, classes, colors, title, name, auc):
    X = df[[fa, fb]].to_numpy(dtype=float)
    y = ypos.astype(int).to_numpy()
    model = make_model().fit(X, y)
    x0, x1, y0, y1 = X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max()
    mx, my = 0.03 * (x1 - x0 + 1e-9), 0.05 * (y1 - y0 + 1e-9)
    xx, yy = np.meshgrid(np.linspace(x0 - mx, x1 + mx, 300), np.linspace(y0 - my, y1 + my, 300))
    zz = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.contourf(xx, yy, zz, levels=[0, 0.5, 1], colors=list(colors), alpha=0.10)
    ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.2)
    rng = np.random.default_rng(C.SEED)
    for lab, cname, col, mk in [(0, classes[0], colors[0], "o"), (1, classes[1], colors[1], "^")]:
        s = y == lab
        xj = X[s, 0] + (rng.uniform(-0.18, 0.18, s.sum()) if fa in DISCRETE else 0)
        yj = X[s, 1] + (rng.uniform(-0.18, 0.18, s.sum()) if fb in DISCRETE else 0)
        ax.scatter(xj, yj, s=26, marker=mk, c=col, edgecolors="k", linewidths=0.3, alpha=0.8, label=cname)
    ax.set_xlabel(fa); ax.set_ylabel(fb)
    ax.set_title(f"{title}  (AUC = {auc:.3f})", fontsize=11); ax.legend(loc="best", fontsize=9)
    fig.tight_layout(); fig.savefig(out(name), dpi=150, bbox_inches="tight"); plt.close(fig)


def pr_frontier(tool, display):
    df = pd.read_csv(os.path.join(C.TABLES, f"cascade_{tool}_all_thresholds.csv"))
    n_ind = int((dev_of(tool).op_label == "individual").sum())
    cg = np.linspace(0, 1, 101)
    front = np.array([df.precision[df.recall >= c].max() if np.any(df.recall >= c) else np.nan for c in cg])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(cg, front, "-", color="#2f6db0", lw=2.5, label="frontera alcanzable")
    for T, col in zip(C.TARGETS, TARGET_COLORS):
        row = operating_point(df, T)
        if row is None:
            continue
        ax.plot(row.recall, row.precision, "o", color=col, ms=9, zorder=5, markeredgecolor="k",
                label=f"precision objetivo = {T:.2f}")
        ax.annotate(f"{int(row.individuales)} ind", (row.recall, row.precision), textcoords="offset points",
                    xytext=(8, 6), fontsize=9, color=col, fontweight="bold")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title(f"{display}  (individuales totales = {n_ind})", fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0.5, 1.02); ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout(); fig.savefig(out(f"pr_frontier_{tool}.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


def auc_of(tool, stage, fa, fb):
    r = pd.read_csv(os.path.join(C.TABLES, f"{tool}_pairs_stage{stage}.csv"))
    m = r[((r.feat_a == fa) & (r.feat_b == fb)) | ((r.feat_a == fb) & (r.feat_b == fa))]
    return float(m.auc_cv.iloc[0])


def main():
    os.makedirs(C.FIGURES, exist_ok=True)
    dev = dev_of("modl")
    heatmap(dev)
    boxplots(dev, FINAL_FEATURES, "descriptores morfológicos", "modl_boxplots_morfo_3clases.png")
    boxplots(dev, INT_FEATURES + ["prob_mean"], "intensidad, contraste y confianza",
             "modl_boxplots_intensidad_3clases.png")
    d2 = dev[dev.op_label.isin(["individual", "cluster"])]
    scatter(d2, "area", "n_endpoints", d2.op_label == "cluster", ("individual", "cluster"),
            ("#1f77b4", "#d62728"), "MoDL — individual vs cluster: area + n_endpoints",
            "modl_scatter_indiv_cluster.png", auc_of("modl", 2, "area", "n_endpoints"))
    d1 = dev.dropna(subset=["op_label"])
    scatter(d1, "snr", "prob_mean", d1.op_label == "spurious", ("real", "falso positivo"),
            ("#2ca02c", "#7f7f7f"), "MoDL — real vs falso positivo: snr + prob_mean",
            "modl_scatter_real_fp.png", auc_of("modl", 1, "snr", "prob_mean"))
    pr_frontier("modl", "MoDL")
    pr_frontier("omnipose", "Omnipose")
    print("Figuras de clasificación ->", C.FIGURES)


if __name__ == "__main__":
    main()
