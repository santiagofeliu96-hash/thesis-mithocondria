"""Copia a `data/`, `models/` y `notebooks/` todos los insumos que usa el pipeline.

Los archivos originales SOLO se leen (shutil.copy2). Nunca se sobrescribe un archivo
ya copiado salvo con --force. Al final se escribe `data/manifest.csv` con origen,
destino, tamaño y MD5 de cada archivo copiado.

Uso:  python scripts/00_prepare_data.py [--force] [--skip-model]
"""
import argparse
import csv
import glob
import hashlib
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

TESIS = os.path.dirname(C.ROOT)                                   # .../Facultad/Tesis
CODE = os.path.join(TESIS, "Code")
DS = os.path.join(CODE, "segmetation_investigation", "dataset")
CMP = os.path.join(CODE, "segmetation_investigation", "visualization", "metrics", "comparison")
CCI = os.path.join(CODE, "cluster_classification_investigation")
INTER = os.path.join(CODE, "gt_extension", "interannotator")
DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")

copied = []


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cp(src, dst, force=False):
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if force or not os.path.exists(dst):
        shutil.copy2(src, dst)
    copied.append((src, os.path.relpath(dst, C.ROOT)))


def first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(paths[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-copiar aunque el destino exista")
    ap.add_argument("--skip-model", action="store_true", help="no copiar el modelo MoDL (478 MB)")
    a = ap.parse_args()
    f = a.force
    orig = set(C.ORIG_FRAMES)

    # --- Imágenes crudas y ground truth (21 fotogramas) ---
    for s in C.ALL_FRAMES:
        if s in orig:
            raw = first_existing(os.path.join(DS, "ground_truth", "tif", s + ".tif"),
                                 os.path.join(DS, "raw", "tif", s + ".tif"))
            gt = os.path.join(DS, "ground_truth", s + "_seg.npy")
        else:
            raw = os.path.join(DS, "ground_truth_ext", "tif", s + ".tif")
            gt = os.path.join(DS, "ground_truth_ext", s + "_seg.npy")
        cp(raw, os.path.join(C.RAW_DIR, s + ".tif"), f)
        cp(gt, os.path.join(C.GT_DIR, s + "_seg.npy"), f)

    # --- Etiquetas manuales individual / cluster ---
    for n in ["gt_original_labels.csv", "gt_extended_labels.csv"]:
        cp(os.path.join(CCI, n), os.path.join(C.LABELS_DIR, n), f)

    # --- Acuerdo inter-anotador ---
    for who in ["annotator_Gua"]:   # anotador 1 = ground truth
        for p in sorted(glob.glob(os.path.join(INTER, who, "seg_*.tif"))):
            cp(p, os.path.join(C.INTERANNOT_DIR, who, os.path.basename(p)), f)
    cp(os.path.join(INTER, "embeddings", "index.txt"), os.path.join(C.INTERANNOT_DIR, "index.txt"), f)

    # --- Predicciones guardadas: MoDL (probabilidades de la corrida original) ---
    for s in C.ALL_FRAMES:
        sub = "modl" if s in orig else "modl_ext"
        cp(os.path.join(CMP, sub, "prob", s + ".npy"),
           os.path.join(C.STORED, "modl_prob", s + ".npy"), f)

    # --- Predicciones guardadas: Ilastik (para comparar con la re-inferencia) ---
    for proj in C.ILASTIK_PROJECT_NAMES:
        for s in C.EVAL15:
            if s in orig:
                d = os.path.join(DS, "segmented", "ilastik",
                                 proj if proj == "2mito_3classes" else os.path.join("archive", proj))
            else:
                d = os.path.join(DS, "segmented", "ilastik_ext", proj)
            src = first_existing(os.path.join(d, s + ".tif"), os.path.join(d, s + ".tiff"))
            cp(src, os.path.join(C.STORED, "ilastik", proj, s + ".tif"), f)

    # --- Cellpose-SAM (Colab): CSVs por imagen + máscaras locales disponibles ---
    for sub, tag in [("cellpose", "orig"), ("cellpose_ext", "ext")]:
        for n in ["resultados_por_imagen.csv", "resultados_pixel_por_imagen.csv"]:
            cp(os.path.join(CMP, sub, n), os.path.join(C.STORED, "cellpose", "csv", f"{tag}_{n}"), f)
    for s in C.ORIG_FRAMES:
        for p in glob.glob(os.path.join(DS, "segmented", "cellpose", "base_best", s + ".tif", "*_masks.npy")):
            cp(p, os.path.join(C.STORED, "cellpose", "masks", s, os.path.basename(p)), f)

    # --- Omnipose (Colab): CSVs de la grilla completa (15+ fotogramas) + máscaras top-5 ---
    for n in ["resultados_por_imagen_first.csv", "resultados_por_imagen_ext.csv",
              "resultados_pixel_por_imagen_first.csv", "resultados_pixel_por_imagen_ext.csv"]:
        cp(os.path.join(CMP, "omnipose_eval15", n), os.path.join(C.STORED, "omnipose", "csv", n), f)
    for sub in ["omnipose", "omnipose_ext"]:
        for cfg_dir in glob.glob(os.path.join(DS, "segmented", sub, "top5_ap_cellpose_masks", "*")):
            if not os.path.isdir(cfg_dir):
                continue
            for p in glob.glob(os.path.join(cfg_dir, "*_pred.npy")):
                cp(p, os.path.join(C.STORED, "omnipose", "masks", os.path.basename(cfg_dir),
                                   os.path.basename(p)), f)
    for n in ["omnipose_confidence_cyto2_mt-2_ft0.4.csv", "omnipose_confidence_ext_cyto2_mt-2_ft0.4.csv"]:
        cp(os.path.join(CCI, n), os.path.join(C.STORED, "omnipose", "confidence", n), f)

    # --- Modelos ---
    for proj in C.ILASTIK_PROJECT_NAMES:
        cp(os.path.join(TESIS, "Ilastik", "pixel_classification", proj + ".ilp"),
           os.path.join(C.ILASTIK_PROJECTS, proj + ".ilp"), f)
    if not a.skip_model:
        cp(os.path.join(TESIS, "MoDL-main", "model", "U-RNet+.hdf5"), C.MODL_MODEL, f)

    # --- Notebooks de Colab (referencia; no se ejecutan localmente) ---
    for n in ["Cellpose_evaluation (2).ipynb", "Omnipose_evaluation_eval15.ipynb",
              "omnipose_extract_confidence.ipynb"]:
        p = os.path.join(DOWNLOADS, n)
        if os.path.exists(p):
            cp(p, os.path.join(C.ROOT, "notebooks", n.replace(" (2)", "")), f)
    cp(os.path.join(CODE, "gt_extension", "colab", "Cellpose_evaluation_EXT.ipynb"),
       os.path.join(C.ROOT, "notebooks", "Cellpose_evaluation_EXT.ipynb"), f)

    # --- Manifiesto ---
    with open(os.path.join(C.DATA, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["origen", "destino", "bytes", "md5"])
        for src, dst in copied:
            w.writerow([src, dst, os.path.getsize(src), md5(src)])
    print(f"{len(copied)} archivos copiados/verificados -> data/manifest.csv")


if __name__ == "__main__":
    main()
