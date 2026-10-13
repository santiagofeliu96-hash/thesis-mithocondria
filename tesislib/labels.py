"""Etiquetas: clase manual de cada instancia GT y clase operacional de cada máscara predicha."""
import os

import numpy as np
import pandas as pd

import config as C


def load_manual_labels():
    """DataFrame (frame, label, area_px, clase) con las 874 instancias GT (individual / cluster)."""
    return pd.concat([pd.read_csv(os.path.join(C.LABELS_DIR, n))
                      for n in ("gt_original_labels.csv", "gt_extended_labels.csv")],
                     ignore_index=True)


def gt_cluster_set(manual=None):
    manual = load_manual_labels() if manual is None else manual
    return {(r.frame, int(r.label)) for r in manual.itertuples() if r.clase == "cluster"}


def operational_labels(gt_lab, pred_lab, frame, gt_clusters):
    """{id predicho: 'individual' | 'cluster' | 'spurious'}.

    Una instancia GT es capturada por la máscara predicha que más la cubre, si esa
    cobertura es >= COVER_THR. 0 capturadas -> spurious (falso positivo); >= 2 -> cluster;
    1 -> cluster si esa instancia GT es un cluster biológico anotado, si no individual.
    """
    gt_ids, gt_areas = np.unique(gt_lab, return_counts=True)
    captured = {}
    for g, ga in zip(gt_ids, gt_areas):
        if g == 0:
            continue
        ov = pred_lab[gt_lab == g]
        ov = ov[ov != 0]
        if ov.size == 0:
            continue
        vals, cnt = np.unique(ov, return_counts=True)
        if cnt.max() / ga >= C.COVER_THR:
            captured.setdefault(int(vals[np.argmax(cnt)]), []).append(int(g))
    out = {}
    for pid in np.unique(pred_lab):
        if pid == 0:
            continue
        caps = captured.get(int(pid), [])
        if not caps:
            out[int(pid)] = "spurious"
        elif len(caps) >= 2:
            out[int(pid)] = "cluster"
        else:
            out[int(pid)] = "cluster" if (frame, caps[0]) in gt_clusters else "individual"
    return out
