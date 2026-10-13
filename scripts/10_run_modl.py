"""Inferencia de MoDL (U-RNet+, pesos preentrenados) sobre los 21 fotogramas del corpus.

Requiere el entorno con TensorFlow 2.5 (MoDL-main/venvtest2). Guarda el mapa de
probabilidad crudo por fotograma en outputs/predictions/modl/prob/<stem>.npy; la
binarización se hace después (barrido de umbral).

Preprocesado idéntico a MoDL-main/MoDL_seg/predict_masks.py: carga en escala de grises
con keras `load_img` + `img_to_array(...).astype(uint8)`, luego /255 y resta de la media
por imagen.

Uso:  <venvtest2>/Scripts/python.exe scripts/10_run_modl.py [--cpu]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

import tensorflow as tf  # noqa: E402
from tensorflow.keras.models import load_model  # noqa: E402
from tensorflow.keras.preprocessing.image import img_to_array, load_img  # noqa: E402

SIZE = 512


def load_input(path):
    img = load_img(path, color_mode="grayscale")
    if img.size != (SIZE, SIZE):
        img = img.resize((SIZE, SIZE))
    gray = img_to_array(img)[:, :, 0].astype(np.uint8)
    x = gray.astype("float32") / 255.0
    return (x - x.mean())[:, :, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="forzar CPU")
    a = ap.parse_args()
    if a.cpu:
        tf.config.set_visible_devices([], "GPU")

    stems = C.ALL_FRAMES
    batch = np.stack([load_input(os.path.join(C.RAW_DIR, s + ".tif")) for s in stems]).astype("float32")
    model = load_model(C.MODL_MODEL, compile=False)
    probs = model.predict(batch, batch_size=8, verbose=1)

    os.makedirs(C.MODL_PROB_NEW, exist_ok=True)
    for s, p in zip(stems, probs):
        np.save(os.path.join(C.MODL_PROB_NEW, s + ".npy"), p[:, :, 0].astype("float32"))
    print(f"{len(stems)} mapas de probabilidad -> {C.MODL_PROB_NEW}")


if __name__ == "__main__":
    main()
