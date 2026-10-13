"""Carga de imágenes crudas, ground truth y predicciones (guardadas o re-inferidas)."""
import os

import numpy as np
import tifffile

import config as C


def load_raw(stem):
    """Imagen cruda (uint16, HxW)."""
    img = np.asarray(tifffile.imread(os.path.join(C.RAW_DIR, stem + ".tif")))
    if img.ndim == 3:
        img = img[..., 0] if img.shape[-1] <= 4 else img[0]
    return img


def load_raw_norm(stem):
    """Cruda normalizada a [0, 1] (para fondos de figuras)."""
    img = load_raw(stem).astype(float)
    return (img - img.min()) / (img.max() - img.min()) if img.max() > img.min() else img


def load_gt(stem):
    """Máscara de instancias del ground truth (0 = fondo, 1..N = instancias)."""
    d = np.load(os.path.join(C.GT_DIR, stem + "_seg.npy"), allow_pickle=True).item()
    return np.asarray(d["masks"])


def load_label_npy(path):
    """Máscara etiquetada desde .npy (array plano o dict de Cellpose con 'masks')."""
    d = np.load(path, allow_pickle=True)
    try:
        d = d.item()
        return np.asarray(d["masks"] if "masks" in d else next(iter(d.values())))
    except (ValueError, AttributeError):
        return np.asarray(d)


def _pick(new_path, stored_path, stored):
    """Predicción re-inferida si existe (y no se pide la guardada); si no, la guardada."""
    return stored_path if stored or not os.path.exists(new_path) else new_path


def load_modl_prob(stem, stored=False):
    """Mapa de probabilidad de MoDL: re-inferido (10_run_modl.py) o, si no existe, el guardado."""
    p = _pick(os.path.join(C.MODL_PROB_NEW, stem + ".npy"),
              os.path.join(C.STORED, "modl_prob", stem + ".npy"), stored)
    return np.load(p)


def load_ilastik(project, stem, stored=False):
    """Simple Segmentation de Ilastik (1 = mito, 2 = fondo, 3 = tercera clase): re-inferida
    (11_run_ilastik.py) o, si no existe, la guardada."""
    p = _pick(os.path.join(C.ILASTIK_NEW, project, stem + ".tif"),
              os.path.join(C.STORED, "ilastik", project, stem + ".tif"), stored)
    return np.asarray(tifffile.imread(p)).squeeze()


def load_omnipose(cfg, stem):
    return load_label_npy(os.path.join(C.STORED, "omnipose", "masks", cfg, stem + "_pred.npy"))


def load_cellpose(cfg, stem):
    """cfg con el formato de archivo de Colab, p. ej. '0.4_0.0_0'."""
    return load_label_npy(os.path.join(C.STORED, "cellpose", "masks", stem, cfg + "_masks.npy"))
