"""Métricas de segmentación a nivel de píxel y de instancia.

Instancia: emparejamiento uno a uno por algoritmo húngaro sobre 1 - IoU, aceptando
pares con IoU >= tau. AP (convención Cellpose) = TP / (TP + FP + FN);
tasa de error = (FP + FN) / (TP + FN).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.measure import label

import config as C


# --------------------------------------------------------------------------- #
# Píxel
# --------------------------------------------------------------------------- #
def ilastik_to_binary(mask, mito_classes):
    """Simple Segmentation de Ilastik -> binaria con las clases consideradas mitocondria."""
    return np.isin(mask, mito_classes).astype(np.uint8)


def pixel_metrics(gt, pred):
    """precision, recall, F1 e IoU de la clase mitocondria (binarias con > 0)."""
    gt = np.asarray(gt) > 0
    pred = np.asarray(pred) > 0
    tp = int(np.sum(gt & pred))
    fp = int(np.sum(~gt & pred))
    fn = int(np.sum(gt & ~pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {"px_precision": precision, "px_recall": recall, "px_f1": f1, "px_iou": iou}


# --------------------------------------------------------------------------- #
# Instancia
# --------------------------------------------------------------------------- #
def label_binary(bw):
    """Instanciación por componentes conexas (8-conexidad)."""
    return label(np.asarray(bw) > 0, connectivity=C.CONNECTIVITY)


def iou_matrix(gt_lab, pr_lab):
    """Matriz de IoU entre instancias GT (filas) y predichas (columnas)."""
    gt_lab = np.asarray(gt_lab, dtype=np.int64)
    pr_lab = np.asarray(pr_lab, dtype=np.int64)
    gt_ids = np.unique(gt_lab); gt_ids = gt_ids[gt_ids != 0]
    pr_ids = np.unique(pr_lab); pr_ids = pr_ids[pr_ids != 0]
    iou = np.zeros((len(gt_ids), len(pr_ids)), dtype=float)
    if len(gt_ids) == 0 or len(pr_ids) == 0:
        return gt_ids, pr_ids, iou
    gt_area = np.bincount(gt_lab.ravel())[gt_ids]
    pr_area = np.bincount(pr_lab.ravel())[pr_ids]
    both = (gt_lab > 0) & (pr_lab > 0)
    gi = np.searchsorted(gt_ids, gt_lab[both])
    pj = np.searchsorted(pr_ids, pr_lab[both])
    inter = np.zeros((len(gt_ids), len(pr_ids)), dtype=np.int64)
    np.add.at(inter, (gi, pj), 1)
    union = gt_area[:, None] + pr_area[None, :] - inter
    np.divide(inter, union, out=iou, where=union > 0)
    return gt_ids, pr_ids, iou


def match_instances(gt_lab, pr_lab, tau=C.TAU_REF):
    gt_ids, pr_ids, iou = iou_matrix(gt_lab, pr_lab)
    matched_gt, matched_pr = [], []
    if iou.size > 0:
        gi, pj = linear_sum_assignment(1.0 - iou)
        for g, p in zip(gi, pj):
            if iou[g, p] >= tau:
                matched_gt.append(gt_ids[g]); matched_pr.append(pr_ids[p])
    tp = len(matched_pr)
    return {"tp": tp, "fp": len(pr_ids) - tp, "fn": len(gt_ids) - tp,
            "tp_pr_ids": set(matched_pr),
            "fp_pr_ids": set(pr_ids) - set(matched_pr),
            "fn_gt_ids": set(gt_ids) - set(matched_gt)}


def instance_metrics(gt_lab, pr_lab, tau=C.TAU_REF):
    """precision, recall, F1, AP y tasa de error a un tau. None si el GT está vacío."""
    m = match_instances(gt_lab, pr_lab, tau)
    tp, fp, fn = m["tp"], m["fp"], m["fn"]
    if tp + fn == 0:
        return None
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
            "ap": tp / (tp + fp + fn), "error_rate": (fp + fn) / (tp + fn)}
