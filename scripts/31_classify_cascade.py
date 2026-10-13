"""Clasificación en cascada de las máscaras de MoDL y Omnipose (desarrollo vs test = P36).

Etapa 1 (real vs falso positivo): descriptores de intensidad/contraste/confianza.
Etapa 2 (individual vs cluster): 10 descriptores morfológicos, solo máscaras reales.
En cada etapa se elige el par de descriptores con mayor AUC (CV 5x10). Los umbrales
(t1, t2) se eligen sobre probabilidades fuera de pliegue del desarrollo: máximo recall de
la clase individual con precision >= pi. Para test se reajustan los modelos sobre todo
el desarrollo y se aplican los umbrales congelados.

Salidas (outputs/tables/):
  tab_res2_dist.csv                         tab:res2-dist
  tab_res2_auc.csv  + {tool}_pairs_stage{1,2}.csv   tab:res2-auc (ranking completo)
  cascade_{tool}_all_thresholds.csv         grilla 51x51 (desarrollo)
  tab_res2_operacion.csv                    tab:res2-operacion y tab:res2-validacion
  tab_res2_desglose_modl.csv                desglose de las 154 máscaras de test (MoDL, pi=0.90)
  cascade_test_decisions_modl.csv           decisión por máscara de test (para figuras)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from tesislib.cascade import (STAGE1_FEATURES, counts, make_model, oof_probabilities,
                              operating_point, rank_pairs, sweep)
from tesislib.features import FINAL_FEATURES

TOOL_NAMES = {"modl": "MoDL", "omnipose": "Omnipose"}


def masks(df):
    return ((df.op_label == "individual").values, (df.op_label == "cluster").values,
            (df.op_label == "spurious").values)


def run(tool, dist_rows, auc_rows, op_rows):
    df = pd.read_csv(os.path.join(C.TABLES, f"all_features_{tool}.csv"))
    dev = df[~df.film.isin(C.TEST_FILMS)].reset_index(drop=True)
    tst = df[df.film.isin(C.TEST_FILMS)].reset_index(drop=True)
    for name, d in [("desarrollo", dev), ("test", tst)]:
        vc = d.op_label.value_counts()
        dist_rows.append({"conjunto": name, "tool": TOOL_NAMES[tool], "individual": vc.get("individual", 0),
                          "cluster": vc.get("cluster", 0), "falso_positivo": vc.get("spurious", 0),
                          "total": len(d)})

    is_ind, is_clu, is_fp = masks(dev)
    # --- ranking de pares por etapa ---
    f1 = STAGE1_FEATURES[tool]
    r1 = rank_pairs(dev[f1].to_numpy(), is_fp.astype(int), f1)
    real = dev[is_ind | is_clu].reset_index(drop=True)
    r2 = rank_pairs(real[FINAL_FEATURES].to_numpy(), (real.op_label == "cluster").astype(int).values,
                    FINAL_FEATURES)
    r1.to_csv(os.path.join(C.TABLES, f"{tool}_pairs_stage1.csv"), index=False)
    r2.to_csv(os.path.join(C.TABLES, f"{tool}_pairs_stage2.csv"), index=False)
    for stage, r in [("1. Falso positivo", r1), ("2. Cluster", r2)]:
        for k in range(3):
            auc_rows.append({"tool": TOOL_NAMES[tool], "etapa": stage, "rank": k + 1,
                             "par": f"{r.feat_a[k]} + {r.feat_b[k]}", "auc": r.auc_cv[k]})
    pair1 = (r1.feat_a[0], r1.feat_b[0])
    pair2 = (r2.feat_a[0], r2.feat_b[0])

    # --- umbrales en desarrollo (probabilidades fuera de pliegue) ---
    p_fp, p_clu = oof_probabilities(dev, pair1, pair2, is_ind, is_clu, is_fp)
    table = sweep(p_fp, p_clu, is_ind, is_clu, is_fp)
    table.to_csv(os.path.join(C.TABLES, f"cascade_{tool}_all_thresholds.csv"), index=False)

    # --- modelos finales sobre todo el desarrollo -> test ---
    m1 = make_model().fit(dev[list(pair1)].to_numpy(), is_fp.astype(int))
    m2 = make_model().fit(real[list(pair2)].to_numpy(), (real.op_label == "cluster").astype(int).values)
    tp_fp = m1.predict_proba(tst[list(pair1)].to_numpy())[:, 1]
    tp_clu = m2.predict_proba(tst[list(pair2)].to_numpy())[:, 1]
    ti, tc, tf = masks(tst)

    for target in C.TARGETS:
        row = operating_point(table, target)
        keep1 = tp_fp <= row.t1
        keep = keep1 & (tp_clu <= row.t2)
        ni, nc, ns = counts(keep, ti, tc, tf)
        op_rows.append({"tool": TOOL_NAMES[tool], "pi": target, "t1": row.t1, "t2": row.t2,
                        "dev_precision": row.precision, "dev_recall": row.recall,
                        "dev_individuales": int(row.individuales), "dev_n_ind": int(is_ind.sum()),
                        "test_precision": ni / (ni + nc + ns) if (ni + nc + ns) else np.nan,
                        "test_recall": ni / ti.sum(), "test_individuales": ni, "test_n_ind": int(ti.sum())})
        if tool == "modl" and np.isclose(target, 0.90):
            r1n, r2n = ~keep1, keep1 & ~keep
            desg = pd.DataFrame([{"grupo": g, "individual": int((sel & ti).sum()),
                                  "cluster": int((sel & tc).sum()), "falso_positivo": int((sel & tf).sum()),
                                  "total": int(sel.sum())}
                                 for g, sel in [("retenidas", keep), ("descartadas_etapa1", r1n),
                                                ("descartadas_etapa2", r2n),
                                                ("total", np.ones(len(tst), bool))]])
            desg.to_csv(os.path.join(C.TABLES, "tab_res2_desglose_modl.csv"), index=False)
            dec = tst[["frame", "label", "op_label", "centroid_row", "centroid_col"]].copy()
            dec["p_fp"], dec["p_clu"] = tp_fp, tp_clu
            dec["decision"] = np.where(keep, "retenida", np.where(r1n, "descartada_etapa1", "descartada_etapa2"))
            dec.to_csv(os.path.join(C.TABLES, "cascade_test_decisions_modl.csv"), index=False)

    print(f"\n{TOOL_NAMES[tool]}: etapa 1 {pair1} AUC={r1.auc_cv[0]:.3f} | etapa 2 {pair2} AUC={r2.auc_cv[0]:.3f}")


def main():
    dist, auc, op = [], [], []
    for tool in ["modl", "omnipose"]:
        run(tool, dist, auc, op)
    for rows, name in [(dist, "tab_res2_dist.csv"), (auc, "tab_res2_auc.csv"), (op, "tab_res2_operacion.csv")]:
        pd.DataFrame(rows).to_csv(os.path.join(C.TABLES, name), index=False, float_format="%.6f")
    pd.set_option("display.width", 200)
    print("\n", pd.DataFrame(dist).to_string(index=False))
    print("\n", pd.DataFrame(auc).to_string(index=False, float_format=lambda v: "%.3f" % v))
    print("\n", pd.DataFrame(op).to_string(index=False, float_format=lambda v: "%.3f" % v))
    print("\n", pd.read_csv(os.path.join(C.TABLES, "tab_res2_desglose_modl.csv")).to_string(index=False))


if __name__ == "__main__":
    main()
