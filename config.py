"""Configuración central: rutas, conjuntos de fotogramas y parámetros fijos.

Todas las rutas son relativas a esta carpeta, salvo `ORIGINAL_*`, que apuntan al
código/datos originales de la tesis y solo se leen (en `00_prepare_data.py`).
"""
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# Carpetas de esta base de código
# --------------------------------------------------------------------------- #
DATA = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA, "raw")                      # <stem>.tif (uint16, 512x512)
GT_DIR = os.path.join(DATA, "ground_truth")              # <stem>_seg.npy (campo "masks")
LABELS_DIR = os.path.join(DATA, "labels")                # etiquetas manuales individual/cluster
INTERANNOT_DIR = os.path.join(DATA, "interannotator")
STORED = os.path.join(DATA, "stored_predictions")        # predicciones guardadas (Colab / corridas viejas)

MODELS = os.path.join(ROOT, "models")
MODL_MODEL = os.path.join(MODELS, "U-RNet+.hdf5")
ILASTIK_PROJECTS = os.path.join(MODELS, "ilastik")

OUTPUTS = os.path.join(ROOT, "outputs")
PRED_DIR = os.path.join(OUTPUTS, "predictions")          # predicciones re-generadas en esta base
TABLES = os.path.join(OUTPUTS, "tables")
FIGURES = os.path.join(OUTPUTS, "figures")

# Predicciones re-inferidas (MoDL, Ilastik)
MODL_PROB_NEW = os.path.join(PRED_DIR, "modl", "prob")
ILASTIK_NEW = os.path.join(PRED_DIR, "ilastik")

# Ejecutables / entornos externos
ILASTIK_EXE = r"C:\Program Files\ilastik-1.4.2\ilastik.exe"

# --------------------------------------------------------------------------- #
# Fotogramas del corpus (convención ImageFV1000{zoom}00{película}_C002T{tiempo})
# --------------------------------------------------------------------------- #
ORIG_FRAMES = [  # anotación original (11)
    "ImageFV10006.50006_C002T010",
    "ImageFV10006.50010_C002T056", "ImageFV10006.50010_C002T093", "ImageFV10006.50010_C002T134",
    "ImageFV10006.50024_C002T023", "ImageFV10006.50024_C002T038", "ImageFV10006.50024_C002T061",
    "ImageFV10006.50024_C002T186",
    "ImageFV10006.50025_C002T020", "ImageFV10006.50025_C002T034", "ImageFV10006.50025_C002T111",
]
EXT_FRAMES = [  # extensión del corpus (10)
    "ImageFV10006.50007_C002T050", "ImageFV10006.50007_C002T164",
    "ImageFV10006.50008_C002T047", "ImageFV10006.50008_C002T121",
    "ImageFV10006.50028_C002T027", "ImageFV10006.50028_C002T177",
    "ImageFV10006.50036_C002T086", "ImageFV10006.50036_C002T118",
    "ImageFV10006.50037_C002T017", "ImageFV10006.50037_C002T100",
]
ALL_FRAMES = ORIG_FRAMES + EXT_FRAMES  # 21 fotogramas, 9 películas

# Películas usadas para entrenar Ilastik -> excluidas de la evaluación de segmentadores
ILASTIK_TRAIN_FILMS = {"P28", "P36", "P37"}

# Para ordenar tablas por película y tiempo
def _order_key(stem):
    m = re.search(r"\.500(\d\d)_C\d+T(\d+)", stem)
    return int(m.group(1)), int(m.group(2))


def short(stem):
    """ImageFV10006.50025_C002T020 -> P25T20."""
    m = re.search(r"\.(\d+)_C\d+T(\d+)", stem)
    return "P%dT%d" % (int(m.group(1)) % 100, int(m.group(2)))


def film(stem):
    """ImageFV10006.50025_C002T020 -> P25 (con dos dígitos: P06)."""
    return "P" + re.search(r"500(\d\d)", stem).group(1)


ALL_FRAMES = sorted(ALL_FRAMES, key=_order_key)
EVAL15 = [s for s in ALL_FRAMES if film(s) not in {"P28", "P36", "P37"}]  # Cap. Evaluación

# Fotogramas de la grilla 3x3 de las figuras (filas = películas P25 / P24 / P10)
GRID9 = ["ImageFV10006.50025_C002T020", "ImageFV10006.50025_C002T034", "ImageFV10006.50025_C002T111",
         "ImageFV10006.50024_C002T023", "ImageFV10006.50024_C002T061", "ImageFV10006.50024_C002T186",
         "ImageFV10006.50010_C002T056", "ImageFV10006.50010_C002T093", "ImageFV10006.50010_C002T134"]

# --------------------------------------------------------------------------- #
# Parámetros de evaluación de segmentadores
# --------------------------------------------------------------------------- #
TAU_REF = 0.5
TAUS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
CONNECTIVITY = 2                     # 8-conexidad en componentes conexas
MODL_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]   # 0.05..0.95

# Variantes de Ilastik: nombre -> (proyecto .ilp, clases que cuentan como mitocondria)
ILASTIK_VARIANTS = {
    "v2_3classes/1mito": ("v2_3classes", [1]),
    "2mito_3classes/1mito": ("2mito_3classes", [1]),
    "v2_2classes/1mito": ("v2_2classes", [1]),
    "2mito_3classes/2mito": ("2mito_3classes", [1, 3]),
}
ILASTIK_PROJECT_NAMES = ["v2_2classes", "v2_3classes", "2mito_3classes"]

# --------------------------------------------------------------------------- #
# Parámetros de la clasificación en cascada
# --------------------------------------------------------------------------- #
MODL_CLASSIF_THR = 0.45        # mejor umbral de instancia de MoDL
OMNI_CLASSIF_CFG = "cyto2_omni_mt-2_ft0.4"
COVER_THR = 0.5                # cobertura mínima para "capturar" una instancia GT
TEST_FILMS = {"P36"}           # conjunto de test (por película)
TARGETS = [0.99, 0.95, 0.90, 0.85]
SEED = 0
