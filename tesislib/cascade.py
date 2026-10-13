"""Clasificador en cascada: regresión logística por pares de descriptores y barrido de umbrales."""
import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C

GRID = np.linspace(0, 1, 51)
COMMON_STAGE1 = ["area", "int_mean", "int_std", "int_p90", "int_max", "int_cv",
                 "contrast", "snr", "ratio_bg"]
STAGE1_FEATURES = {"modl": COMMON_STAGE1 + ["prob_mean"],
                   "omnipose": COMMON_STAGE1 + ["cellprob_mean", "cellprob_std", "cellprob_p10"]}


def make_model():
    """z-score + regresión logística L2 (C=1), ponderada por desbalance."""
    return make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=2000))


def rank_pairs(X, y, feats):
    """AUC media (CV estratificada 5x10) para todos los pares de descriptores."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=C.SEED)
    rows = []
    for i, j in itertools.combinations(range(len(feats)), 2):
        auc = cross_val_score(make_model(), X[:, [i, j]], y, cv=cv, scoring="roc_auc").mean()
        rows.append({"feat_a": feats[i], "feat_b": feats[j], "auc_cv": auc})
    return pd.DataFrame(rows).sort_values("auc_cv", ascending=False).reset_index(drop=True)


def oof_probabilities(dev, pair1, pair2, is_ind, is_clu, is_fp):
    """Probabilidades fuera de pliegue P(FP) y P(cluster) sobre el conjunto de desarrollo.

    La etapa 2 se ajusta solo sobre máscaras reales; los FP se puntúan con un modelo
    ajustado sobre todas las reales.
    """
    cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    p_fp = cross_val_predict(make_model(), dev[list(pair1)].to_numpy(), is_fp.astype(int),
                             cv=cv, method="predict_proba")[:, 1]
    real = is_ind | is_clu
    y_clu = is_clu[real].astype(int)
    p_clu = np.empty(len(dev))
    p_clu[real] = cross_val_predict(make_model(), dev.loc[real, list(pair2)].to_numpy(), y_clu,
                                    cv=cv, method="predict_proba")[:, 1]
    m2 = make_model().fit(dev.loc[real, list(pair2)].to_numpy(), y_clu)
    p_clu[~real] = m2.predict_proba(dev.loc[~real, list(pair2)].to_numpy())[:, 1]
    return p_fp, p_clu


def counts(keep, is_ind, is_clu, is_fp):
    return int((keep & is_ind).sum()), int((keep & is_clu).sum()), int((keep & is_fp).sum())


def sweep(p_fp, p_clu, is_ind, is_clu, is_fp):
    """Grilla 51x51 de (t1, t2): precision y recall de la clase individual retenida."""
    n_ind = int(is_ind.sum())
    rows = []
    for t1 in GRID:
        k1 = p_fp <= t1
        for t2 in GRID:
            ni, nc, ns = counts(k1 & (p_clu <= t2), is_ind, is_clu, is_fp)
            tot = ni + nc + ns
            if tot:
                rows.append((round(t1, 2), round(t2, 2), ni / tot, ni / n_ind, ni, nc, ns))
    return pd.DataFrame(rows, columns=["t1", "t2", "precision", "recall", "individuales", "clu", "fp"])


def operating_point(table, target):
    """Máximo recall sujeto a precision >= target (primer máximo en orden t1, t2)."""
    sub = table[table.precision >= target]
    return None if sub.empty else sub.loc[sub.recall.idxmax()]
