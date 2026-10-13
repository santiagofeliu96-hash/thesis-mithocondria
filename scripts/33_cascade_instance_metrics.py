"""AP0.5 y tasa de error de MoDL (thr = 0.45) en el test (P36), antes y después de la cascada.

Escenarios:
  sin_filtro   todas las componentes conexas de prob > 0.45 (= evaluación de segmentadores)
  cascada_pi   solo las máscaras que la cascada retiene con los umbrales (t1, t2) elegidos en
               desarrollo para la precision objetivo pi (modelos ajustados sobre todo el desarrollo,
               igual que 31_classify_cascade.py)

Se mide contra dos referencias:
  gt_completo     todas las instancias GT (individuales + clusters) -> comparable con la tesis
  gt_individual   solo las instancias GT individuales (la población que la cascada busca retener)

Las máscaras < 16 px no tienen descriptores y nunca son retenidas por la cascada.

Salidas: outputs/tables/tab_cascade_instance_P36.csv (por fotograma y promedio)
"""
import os
import sys

import numpy as np
import pandas as pd
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from tesislib import io
from tesislib.cascade import make_model
from tesislib.labels import load_manual_labels
from tesislib.metrics import instance_metrics

TARGETS = [0.90, 0.85]


def cascade_keep(feats, op):
    """{(frame, label)} retenidas en test para cada pi, con modelos ajustados en desarrollo."""
    dev = feats[~feats.film.isin(C.TEST_FILMS)]
    tst = feats[feats.film.isin(C.TEST_FILMS)]
    r1 = pd.read_csv(os.path.join(C.TABLES, "modl_pairs_stage1.csv")).iloc[0]
    r2 = pd.read_csv(os.path.join(C.TABLES, "modl_pairs_stage2.csv")).iloc[0]
    pair1, pair2 = [r1.feat_a, r1.feat_b], [r2.feat_a, r2.feat_b]
    real = dev[dev.op_label.isin(["individual", "cluster"])]
    m1 = make_model().fit(dev[pair1].to_numpy(), (dev.op_label == "spurious").astype(int))
    m2 = make_model().fit(real[pair2].to_numpy(), (real.op_label == "cluster").astype(int))
    p_fp = m1.predict_proba(tst[pair1].to_numpy())[:, 1]
    p_clu = m2.predict_proba(tst[pair2].to_numpy())[:, 1]
    out = {}
    for pi in TARGETS:
        row = op[(op.tool == "MoDL") & np.isclose(op.pi, pi)].iloc[0]
        keep = (p_fp <= row.t1) & (p_clu <= row.t2)
        out[pi] = (set(zip(tst.frame[keep], tst.label[keep].astype(int))), row.t1, row.t2)
    return out


def main():
    feats = pd.read_csv(os.path.join(C.TABLES, "all_features_modl.csv"))
    op = pd.read_csv(os.path.join(C.TABLES, "tab_res2_operacion.csv"))
    keep = cascade_keep(feats, op)
    manual = load_manual_labels().drop_duplicates(["frame", "label"])
    test_frames = [s for s in C.ALL_FRAMES if C.film(s) in C.TEST_FILMS]

    rows = []
    for s in test_frames:
        gt = io.load_gt(s)
        indiv_ids = manual[(manual.frame == s) & (manual.clase == "individual")].label.astype(int)
        gt_ind = np.where(np.isin(gt, indiv_ids), gt, 0)
        pred = measure.label(io.load_modl_prob(s) > C.MODL_CLASSIF_THR, connectivity=C.CONNECTIVITY)
        scenarios = {"sin_filtro": pred}
        for pi, (kept, t1, t2) in keep.items():
            ids = [lab for fr, lab in kept if fr == s]
            scenarios[f"cascada_pi{pi:.2f} (t1={t1:.2f}, t2={t2:.2f})"] = np.where(np.isin(pred, ids), pred, 0)
        for name, pr in scenarios.items():
            for ref, g in [("gt_completo", gt), ("gt_individual", gt_ind)]:
                m = instance_metrics(g, pr, C.TAU_REF)
                rows.append({"referencia": ref, "escenario": name, "frame": C.short(s),
                             "n_pred": len(np.unique(pr)) - 1, "n_gt": m["tp"] + m["fn"],
                             "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                             "precision": m["precision"], "recall": m["recall"],
                             "ap50": m["ap"], "error_rate": m["error_rate"]})

    df = pd.DataFrame(rows)
    num = ["n_pred", "n_gt", "tp", "fp", "fn", "precision", "recall", "ap50", "error_rate"]
    prom = df.groupby(["referencia", "escenario"], sort=False)[num].mean().reset_index()
    prom["frame"] = "Promedio"
    df = pd.concat([df, prom], ignore_index=True)
    df.to_csv(os.path.join(C.TABLES, "tab_cascade_instance_P36.csv"), index=False, float_format="%.6f")

    pd.set_option("display.width", 220)
    for ref in ["gt_completo", "gt_individual"]:
        print(f"\n=== Referencia: {ref} ===")
        print(df[df.referencia == ref].drop(columns="referencia")
              .to_string(index=False, float_format=lambda v: "%.3f" % v))


if __name__ == "__main__":
    main()
