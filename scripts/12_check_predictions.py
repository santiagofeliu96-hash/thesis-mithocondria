"""Compara las predicciones re-inferidas (MoDL, Ilastik) con las guardadas de las corridas originales.

Salida: outputs/tables/check_predictions.csv
  MoDL:    max |Δprob| y fracción de píxeles con distinta binarización a 0.35 / 0.45.
  Ilastik: fracción de píxeles con la misma clase.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from tesislib import io


def main():
    if not (os.path.isdir(C.MODL_PROB_NEW) and os.path.isdir(C.ILASTIK_NEW)):
        print("No hay predicciones re-inferidas (outputs/predictions): se omite la comparación.")
        return
    rows = []
    for s in C.ALL_FRAMES:
        new, old = io.load_modl_prob(s), io.load_modl_prob(s, stored=True)
        rows.append({"tool": "MoDL", "project": "U-RNet+", "frame": C.short(s),
                     "max_abs_diff": float(np.abs(new - old).max()),
                     "frac_equal": float(np.mean((new > 0.35) == (old > 0.35))),
                     "frac_equal_045": float(np.mean((new > 0.45) == (old > 0.45)))})
    for proj in C.ILASTIK_PROJECT_NAMES:
        for s in C.EVAL15:
            new, old = io.load_ilastik(proj, s), io.load_ilastik(proj, s, stored=True)
            rows.append({"tool": "Ilastik", "project": proj, "frame": C.short(s),
                         "frac_equal": float(np.mean(new == old))})
    df = pd.DataFrame(rows)
    os.makedirs(C.TABLES, exist_ok=True)
    df.to_csv(os.path.join(C.TABLES, "check_predictions.csv"), index=False)
    print(df.groupby(["tool", "project"]).agg(
        n=("frame", "size"), min_frac_equal=("frac_equal", "min"),
        max_abs_diff=("max_abs_diff", "max")).to_string())


if __name__ == "__main__":
    main()
