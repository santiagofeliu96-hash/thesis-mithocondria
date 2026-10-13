"""Figuras de superposición GT vs predicción (píxel e instancia)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import config as C
from tesislib import io
from tesislib.metrics import match_instances

PIXEL_CMAP = ListedColormap(["black", "white", "green", "red"])   # fondo, TP, FN, FP


def pixel_superposition(gt_bin, pred_bin):
    gt, pr = np.asarray(gt_bin) > 0, np.asarray(pred_bin) > 0
    sup = np.zeros(gt.shape, dtype=np.uint8)
    sup[gt & pr] = 1
    sup[gt & ~pr] = 2
    sup[~gt & pr] = 3
    return sup


def instance_overlay(gt_lab, pr_lab, tau=C.TAU_REF):
    """RGB: TP (predicción) verde, FP rojo, FN (GT) azul."""
    m = match_instances(gt_lab, pr_lab, tau)
    ov = np.zeros((*gt_lab.shape, 3), dtype=np.float32)
    for p in m["tp_pr_ids"]:
        ov[pr_lab == p] = [0, 1, 0]
    for p in m["fp_pr_ids"]:
        ov[pr_lab == p] = [1, 0, 0]
    for g in m["fn_gt_ids"]:
        ov[gt_lab == g] = [0, 0, 1]
    return ov


def _finish(fig, handles, labels, title, path):
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=13)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.06)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[fig]", path)


def grid_pixel(get_bw, title, path, frames=C.GRID9):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for ax, s in zip(axes.ravel(), frames):
        ax.imshow(pixel_superposition(io.load_gt(s), get_bw(s)), cmap=PIXEL_CMAP,
                  vmin=0, vmax=3, interpolation="nearest")
        ax.set_title(C.short(s)); ax.set_axis_off()
    colors = ["black", "white", "green", "red"]
    _finish(fig, [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors],
            ["Fondo", "TP", "FN", "FP"], title, path)


def grid_instance(get_lab, title, path, frames=C.GRID9):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for ax, s in zip(axes.ravel(), frames):
        ax.imshow(io.load_raw_norm(s), cmap="gray")
        ax.imshow(instance_overlay(io.load_gt(s), get_lab(s)), alpha=0.45)
        ax.set_title(C.short(s)); ax.axis("off")
    colors = ["green", "red", "blue"]
    _finish(fig, [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors],
            ["TP", "FP", "FN"], title, path)
