"""Acuerdo inter-anotador (tab:acuerdo-interanotador) sobre P10T93, P24T61 y P25T34.

Anotador 1 = ground truth del corpus (data/ground_truth); anotador 2 = anotación independiente
(data/interannotator/annotator_Gua, seg_0000X.tif emparejadas por fotograma vía index.txt).

Píxel: IoU del foreground. Instancia: emparejamiento uno a uno (húngaro) a tau = 0.5;
AP0.5 = TP/(TP+FP+FN) y F1 = 2TP/(2TP+FP+FN).

Salida: outputs/tables/tab_acuerdo_interanotador.csv
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from tesislib import io
from tesislib.metrics import match_instances

ANNOTATOR_2 = "annotator_Gua"


def load_index():
    idx = {}
    with open(os.path.join(C.INTERANNOT_DIR, "index.txt"), encoding="utf-8") as f:
        for line in f:
            emb, frame = line.strip().split("\t")
            idx[int(re.search(r"embedding_(\d+)", emb).group(1))] = os.path.splitext(frame)[0]
    return idx


def main():
    rows = []
    for i, stem in sorted(load_index().items()):
        a = io.load_gt(stem)
        b = tifffile.imread(os.path.join(C.INTERANNOT_DIR, ANNOTATOR_2, f"seg_{i:05d}.tif"))
        fa, fb = a > 0, b > 0
        m = match_instances(a, b, C.TAU_REF)
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        rows.append({"fotograma": C.short(stem), "n1": tp + fn, "n2": tp + fp, "coincidentes": tp,
                     "px_iou": np.sum(fa & fb) / np.sum(fa | fb),
                     "ap50": tp / (tp + fp + fn), "f1_inst": 2 * tp / (2 * tp + fp + fn)})
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"fotograma": "Total / Promedio",
                       **df[["n1", "n2", "coincidentes"]].sum().to_dict(),
                       **df[["px_iou", "ap50", "f1_inst"]].mean().to_dict()}
    tot = df.iloc[-1]
    distintas = tot.n1 + tot.n2 - tot.coincidentes
    os.makedirs(C.TABLES, exist_ok=True)
    df.to_csv(os.path.join(C.TABLES, "tab_acuerdo_interanotador.csv"), index=False)
    print(df.to_string(index=False, float_format=lambda v: "%.3f" % v))
    print(f"\nInstancias distintas: {distintas:.0f}; coincidentes: {tot.coincidentes:.0f} "
          f"({100 * tot.coincidentes / distintas:.1f}%)")


if __name__ == "__main__":
    main()
