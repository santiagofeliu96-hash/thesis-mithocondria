"""Figuras del capítulo de evaluación de segmentadores.

  ilastik_v2_3classes_segmentation.png        (fig:resultado_ilastik_grilla, píxel)
  modl_superposicion_pixel_thr0.35.png        (fig:modl-instance, píxel)
  modl_curva_pixel_vs_umbral.png              (fig:res-modl-barrido)
  cellpose_instancia_mejor_config_global.png  (fig:cellpose-instance-matching)
  omnipose_cyto2_omni_best_instance.png       (fig:cyto2_instance)
  omnipose_bact_fluor_omni_best_instance.png  (fig:bact_fluor_instance)

Las grillas usan los 9 fotogramas originales (filas P25 / P24 / P10).
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
from tesislib import io
from tesislib.metrics import ilastik_to_binary
from tesislib.plots import grid_instance, grid_pixel


def fig(name):
    return os.path.join(C.FIGURES, name)


def modl_curve():
    md = pd.read_csv(os.path.join(C.TABLES, "tab_modl.csv"))
    md["thr"] = md.config.str.replace("thr=", "").astype(float)
    md = md.sort_values("thr")
    f, ax = plt.subplots(figsize=(7, 5))
    for col, lab in [("px_precision", "precision"), ("px_recall", "recall"),
                     ("px_f1", "F1-score"), ("px_iou", "IoU")]:
        ax.plot(md.thr, md[col], "o-", ms=4, label=lab)
    best = md.loc[md.px_f1.idxmax()]
    ax.axvline(best.thr, ls="--", color="gray", lw=1)
    ax.set_xlabel("valor de corte (threshold)"); ax.set_ylabel("métrica (promedio 15 fotogramas)")
    ax.set_title("MoDL: métricas de píxel vs. valor de corte (mejor F1 = %.3f en %.2f)"
                 % (best.px_f1, best.thr), fontsize=10)
    ax.grid(alpha=0.3); ax.legend()
    f.tight_layout(); f.savefig(fig("modl_curva_pixel_vs_umbral.png"), dpi=150); plt.close(f)
    print("[fig]", fig("modl_curva_pixel_vs_umbral.png"))


def main():
    os.makedirs(C.FIGURES, exist_ok=True)
    grid_pixel(lambda s: ilastik_to_binary(io.load_ilastik("v2_3classes", s), [1]),
               "Ilastik (v2_3classes) — superposicion pixel", fig("ilastik_v2_3classes_segmentation.png"))
    grid_pixel(lambda s: (io.load_modl_prob(s) > 0.35).astype(np.uint8),
               "MoDL (threshold=0.35) — superposicion pixel", fig("modl_superposicion_pixel_thr0.35.png"))
    grid_instance(lambda s: io.load_cellpose("0.4_0.0_0", s),
                  "Cellpose-SAM (ft=0.4, ct=0, niter=0) — instancia",
                  fig("cellpose_instancia_mejor_config_global.png"))
    grid_instance(lambda s: io.load_omnipose("cyto2_omni_mt-2_ft0.4", s),
                  "Omnipose (cyto2, mt=-2, ft=0.4) — instancia", fig("omnipose_cyto2_omni_best_instance.png"))
    grid_instance(lambda s: io.load_omnipose("bact_fluor_omni_mt0_ft0.4", s),
                  "Omnipose (bact_fluor, mt=0, ft=0.4) — instancia",
                  fig("omnipose_bact_fluor_omni_best_instance.png"))
    modl_curve()


if __name__ == "__main__":
    main()
