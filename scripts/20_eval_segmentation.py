"""Evaluación de las cuatro herramientas sobre los 15 fotogramas de evaluación, a nivel de píxel
y de instancia (tau = 0.5), para todas sus configuraciones.

- MoDL:     barrido de umbral 0.05..0.95 sobre las probabilidades RE-INFERIDAS (10_run_modl).
- Ilastik:  4 variantes (proyecto x clases mito) sobre las máscaras RE-INFERIDAS (11_run_ilastik).
- Cellpose-SAM y Omnipose: métricas por imagen calculadas en Colab (CSV guardados; la
  inferencia necesita GPU y no se re-ejecuta). Como chequeo cruzado se recalculan las
  métricas desde las máscaras guardadas disponibles y se comparan con esos CSV.

Las herramientas semánticas se instancian por componentes conexas; las de instancia se
colapsan a binaria para el nivel de píxel.

Salidas (outputs/tables/): seg_raw_pixel.csv, seg_raw_instance.csv, seg_crosscheck.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from tesislib import io
from tesislib.metrics import ilastik_to_binary, instance_metrics, label_binary, pixel_metrics

INST_COLS = ["precision", "recall", "f1", "ap", "error_rate"]
PX_COLS = ["px_precision", "px_recall", "px_f1", "px_iou"]


def inst_row(gt_lab, pr_lab):
    m = instance_metrics(gt_lab, pr_lab, C.TAU_REF)
    return {k: m[k] for k in INST_COLS}


def eval_semantic(rows_px, rows_in, tool, config, frame, gt, bw):
    rows_px.append({"tool": tool, "config": config, "frame": frame, **pixel_metrics(gt, bw)})
    rows_in.append({"tool": tool, "config": config, "frame": frame, **inst_row(gt, label_binary(bw))})


def cellpose_cfg(ft, ct, niter):
    return "ft=%.1f,ct=%g,niter=%d" % (ft, ct, niter)


def omni_cfg(model, mt, ft):
    return "%s,mt=%g,ft=%.1f" % (model, mt, ft)


def read_colab(paths, frames):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["frame"] = df.image_name.str.replace(".tif", "", regex=False)
    return df[df.frame.isin(frames)].copy()


def main():
    rows_px, rows_in, cross = [], [], []
    short_eval = {s: C.short(s) for s in C.EVAL15}

    # ---------------- MoDL ----------------
    for s in C.EVAL15:
        gt, prob = io.load_gt(s), io.load_modl_prob(s)
        for thr in C.MODL_THRESHOLDS:
            eval_semantic(rows_px, rows_in, "MoDL", "thr=%.2f" % thr, C.short(s), gt,
                          (prob > thr).astype(np.uint8))
    print("MoDL listo")

    # ---------------- Ilastik ----------------
    for name, (proj, classes) in C.ILASTIK_VARIANTS.items():
        for s in C.EVAL15:
            bw = ilastik_to_binary(io.load_ilastik(proj, s), classes)
            eval_semantic(rows_px, rows_in, "Ilastik", name, C.short(s), io.load_gt(s), bw)
    print("Ilastik listo")

    # ---------------- Cellpose-SAM (CSV de Colab) ----------------
    d = os.path.join(C.STORED, "cellpose", "csv")
    cp_in = read_colab([os.path.join(d, "orig_resultados_por_imagen.csv"),
                        os.path.join(d, "ext_resultados_por_imagen.csv")], C.EVAL15)
    cp_in = cp_in[np.isclose(cp_in.tau, C.TAU_REF)]
    cp_px = read_colab([os.path.join(d, "orig_resultados_pixel_por_imagen.csv"),
                        os.path.join(d, "ext_resultados_pixel_por_imagen.csv")], C.EVAL15)
    for df, rows, cols in [(cp_in, rows_in, INST_COLS), (cp_px, rows_px, PX_COLS)]:
        df = df.rename(columns={"ap_cellpose": "ap"})
        for r in df.itertuples():
            rows.append({"tool": "Cellpose-SAM",
                         "config": cellpose_cfg(r.flow_threshold, r.cellprob_threshold, r.niter),
                         "frame": short_eval[r.frame], **{c: getattr(r, c) for c in cols}})

    # chequeo cruzado: máscaras guardadas (11 fotogramas originales, 36 configs)
    ref_in = cp_in.set_index(["frame", "flow_threshold", "cellprob_threshold", "niter"])
    ref_px = cp_px.set_index(["frame", "flow_threshold", "cellprob_threshold", "niter"])
    for s in C.ORIG_FRAMES:
        gt = io.load_gt(s)
        for p in sorted(glob.glob(os.path.join(C.STORED, "cellpose", "masks", s, "*_masks.npy"))):
            ft, ct, ni = os.path.basename(p).replace("_masks.npy", "").split("_")
            key = (s, float(ft), float(ct), int(ni))
            pr = io.load_label_npy(p)
            m, px = instance_metrics(gt, pr), pixel_metrics(gt, pr > 0)
            cross.append({"tool": "Cellpose-SAM", "config": cellpose_cfg(*key[1:]), "frame": C.short(s),
                          "ap_recalc": m["ap"], "ap_colab": ref_in.loc[key, "ap_cellpose"],
                          "px_f1_recalc": px["px_f1"], "px_f1_colab": ref_px.loc[key, "px_f1"]})
    print("Cellpose-SAM listo")

    # ---------------- Omnipose (CSV de Colab) ----------------
    d = os.path.join(C.STORED, "omnipose", "csv")
    om_in = read_colab([os.path.join(d, "resultados_por_imagen_first.csv"),
                        os.path.join(d, "resultados_por_imagen_ext.csv")], C.EVAL15)
    om_in = om_in[np.isclose(om_in.tau, C.TAU_REF)]
    om_px = read_colab([os.path.join(d, "resultados_pixel_por_imagen_first.csv"),
                        os.path.join(d, "resultados_pixel_por_imagen_ext.csv")], C.EVAL15)
    for df, rows, cols in [(om_in, rows_in, INST_COLS), (om_px, rows_px, PX_COLS)]:
        df = df.rename(columns={"ap_cellpose": "ap"})
        for r in df.itertuples():
            rows.append({"tool": "Omnipose", "config": omni_cfg(r.model, r.mask_threshold, r.flow_threshold),
                         "frame": short_eval[r.frame], **{c: getattr(r, c) for c in cols}})

    ref_in = om_in.set_index(["frame", "model", "mask_threshold", "flow_threshold"])
    ref_px = om_px.set_index(["frame", "model", "mask_threshold", "flow_threshold"])
    for cfg_dir in sorted(glob.glob(os.path.join(C.STORED, "omnipose", "masks", "*"))):
        cfg = os.path.basename(cfg_dir)
        model, rest = cfg.split("_omni_")
        mt, ft = rest.replace("mt", "").split("_ft")
        model = model + "_omni"
        for s in C.EVAL15:
            p = os.path.join(cfg_dir, s + "_pred.npy")
            key = (s, model, float(mt), float(ft))
            if not os.path.exists(p) or key not in ref_in.index:
                continue
            gt, pr = io.load_gt(s), io.load_omnipose(cfg, s)
            m, px = instance_metrics(gt, pr), pixel_metrics(gt, pr > 0)
            cross.append({"tool": "Omnipose", "config": omni_cfg(*key[1:]), "frame": C.short(s),
                          "ap_recalc": m["ap"], "ap_colab": ref_in.loc[key, "ap_cellpose"],
                          "px_f1_recalc": px["px_f1"], "px_f1_colab": ref_px.loc[key, "px_f1"]})
    print("Omnipose listo")

    os.makedirs(C.TABLES, exist_ok=True)
    px, ins, cr = pd.DataFrame(rows_px), pd.DataFrame(rows_in), pd.DataFrame(cross)
    px.to_csv(os.path.join(C.TABLES, "seg_raw_pixel.csv"), index=False)
    ins.to_csv(os.path.join(C.TABLES, "seg_raw_instance.csv"), index=False)
    cr["ap_absdiff"] = (cr.ap_recalc - cr.ap_colab).abs()
    cr["px_f1_absdiff"] = (cr.px_f1_recalc - cr.px_f1_colab).abs()
    cr.to_csv(os.path.join(C.TABLES, "seg_crosscheck.csv"), index=False)

    print("\nCobertura (fotogramas por herramienta y config):")
    print(ins.groupby("tool").frame.nunique().to_string())
    print("\nChequeo cruzado máscaras guardadas vs CSV de Colab (máx. |Δ|):")
    print(cr.groupby("tool")[["ap_absdiff", "px_f1_absdiff"]].max().to_string())


if __name__ == "__main__":
    main()
