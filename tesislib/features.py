"""Descriptores por instancia: morfológicos (21 candidatos) y de intensidad/contraste/confianza."""
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import convolve
from skimage import measure
from skimage.morphology import skeletonize

MIN_AREA_PX = 16   # coherente con el umbral de existencia del criterio de anotación
RING = 6           # radio (px) del anillo de fondo local

FEATURE_GROUPS = {
    "size": ["area", "perimeter", "perimeter_crofton", "equiv_diameter", "major_axis",
             "minor_axis", "feret_diameter_max", "convex_area", "filled_area"],
    "shape": ["eccentricity", "extent", "solidity", "form_factor", "roundness",
              "aspect_ratio", "convexity"],
    "topology": ["euler_number", "n_holes"],
    "branching": ["skeleton_length", "n_branch_points", "n_endpoints"],
}
ALL_FEATURES = [f for fs in FEATURE_GROUPS.values() for f in fs]

# 10 descriptores retenidos tras eliminar redundancia (|r| > 0.9)
FINAL_FEATURES = ["area", "solidity", "extent", "eccentricity", "form_factor", "aspect_ratio",
                  "convexity", "n_holes", "n_branch_points", "n_endpoints"]

INT_FEATURES = ["int_mean", "int_median", "int_std", "int_p90", "int_max", "int_cv",
                "contrast", "snr", "ratio_bg"]


def _skeleton_stats(mask_bool):
    """Longitud del esqueleto, puntos de ramificación (>=3 vecinos) y extremos (1 vecino)."""
    skel = skeletonize(mask_bool)
    length = int(skel.sum())
    if length == 0:
        return {"skeleton_length": 0, "n_branch_points": 0, "n_endpoints": 0}
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0) * skel
    return {"skeleton_length": length,
            "n_branch_points": int(np.sum(neighbors >= 3)),
            "n_endpoints": int(np.sum(neighbors == 1))}


def morphology_rows(label_img, min_area=MIN_AREA_PX):
    """Una fila por instancia (área >= min_area) con los 21 descriptores morfológicos."""
    rows = []
    for p in measure.regionprops(label_img):
        if p.area < min_area:
            continue
        area = float(p.area)
        perim = float(p.perimeter) if p.perimeter > 0 else np.nan
        major, minor = float(p.axis_major_length), float(p.axis_minor_length)
        has_perim = bool(perim) and not np.isnan(perim)
        euler = int(p.euler_number)
        row = {
            "label": int(p.label),
            "centroid_row": float(p.centroid[0]), "centroid_col": float(p.centroid[1]),
            "area": area, "perimeter": perim, "perimeter_crofton": float(p.perimeter_crofton),
            "equiv_diameter": float(p.equivalent_diameter_area),
            "major_axis": major, "minor_axis": minor,
            "feret_diameter_max": float(p.feret_diameter_max),
            "convex_area": float(p.area_convex), "filled_area": float(p.area_filled),
            "eccentricity": float(p.eccentricity), "extent": float(p.extent),
            "solidity": float(p.solidity),
            "form_factor": 4 * np.pi * area / perim ** 2 if has_perim else np.nan,
            "roundness": 4 * area / (np.pi * major ** 2) if major > 0 else np.nan,
            "aspect_ratio": major / minor if minor > 0 else np.nan,
            "convexity": float(measure.perimeter(p.image_convex)) / perim if has_perim else np.nan,
            "euler_number": euler,
            "n_holes": 1 - euler,
        }
        row.update(_skeleton_stats(p.image))
        rows.append(row)
    return rows


def intensity_rows(raw, pred_lab, prob=None):
    """{label: descriptores de intensidad} con contraste contra un anillo de fondo local."""
    raw = raw.astype(float)
    fg_dil = ndi.binary_dilation(pred_lab > 0, iterations=2)   # excluye el halo de señal
    rows = {}
    for pid in np.unique(pred_lab):
        if pid == 0:
            continue
        m = pred_lab == pid
        vals = raw[m]
        ring = ndi.binary_dilation(m, iterations=RING) & ~fg_dil
        bg = raw[ring]
        bg_mean = float(bg.mean()) if bg.size else float(raw[~fg_dil].mean())
        bg_std = float(bg.std()) if bg.size else float(raw[~fg_dil].std())
        mu = float(vals.mean())
        r = {"int_mean": mu, "int_median": float(np.median(vals)), "int_std": float(vals.std()),
             "int_p90": float(np.percentile(vals, 90)), "int_max": float(vals.max()),
             "int_cv": float(vals.std() / mu) if mu else np.nan,
             "contrast": mu - bg_mean,
             "snr": (mu - bg_mean) / bg_std if bg_std > 0 else np.nan,
             "ratio_bg": mu / bg_mean if bg_mean > 0 else np.nan}
        if prob is not None:
            r["prob_mean"] = float(prob[m].mean())
        rows[int(pid)] = r
    return rows
