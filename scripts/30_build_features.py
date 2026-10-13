"""Descriptores + etiqueta operacional de cada máscara predicha por MoDL y Omnipose (21 fotogramas).

- MoDL: probabilidades re-inferidas binarizadas a 0.45 + componentes conexas.
- Omnipose: instancias nativas de cyto2_omni (mt=-2, ft=0.4), con la confianza `cellprob`
  exportada desde Colab (merge por frame, label).

Salidas: outputs/tables/all_features_{modl,omnipose}.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from tesislib import io
from tesislib.features import intensity_rows, morphology_rows
from tesislib.labels import gt_cluster_set, operational_labels

CELLPROB_COLS = ["cellprob_mean", "cellprob_median", "cellprob_p10", "cellprob_min", "cellprob_std"]


def build(tool, gt_clusters):
    rows = []
    for s in C.ALL_FRAMES:
        gt, raw = io.load_gt(s), io.load_raw(s)
        if tool == "modl":
            prob = io.load_modl_prob(s)
            pred = measure.label(prob > C.MODL_CLASSIF_THR, connectivity=C.CONNECTIVITY)
        else:
            prob, pred = None, io.load_omnipose(C.OMNI_CLASSIF_CFG, s)
        op = operational_labels(gt, pred, s, gt_clusters)
        inten = intensity_rows(raw, pred, prob)
        for r in morphology_rows(pred):
            r.update(inten.get(r["label"], {}))
            r.update({"frame": s, "film": C.film(s),
                      "source": "orig" if s in C.ORIG_FRAMES else "ext",
                      "op_label": op.get(r["label"])})
            rows.append(r)
    df = pd.DataFrame(rows)
    if tool == "omnipose":
        d = os.path.join(C.STORED, "omnipose", "confidence")
        conf = pd.concat([pd.read_csv(os.path.join(d, n)) for n in sorted(os.listdir(d))], ignore_index=True)
        df = df.merge(conf[["frame", "label"] + CELLPROB_COLS], on=["frame", "label"], how="left")
        assert df.cellprob_mean.notna().all(), "faltan valores de cellprob"
    df.to_csv(os.path.join(C.TABLES, f"all_features_{tool}.csv"), index=False)
    conj = np.where(df.film.isin(C.TEST_FILMS), "test", "desarrollo")
    print(f"\n=== {tool}: {len(df)} máscaras ===")
    print(df.groupby([conj, "op_label"]).size().unstack(fill_value=0).to_string())


def main():
    os.makedirs(C.TABLES, exist_ok=True)
    clusters = gt_cluster_set()
    for tool in ["modl", "omnipose"]:
        build(tool, clusters)


if __name__ == "__main__":
    main()
