"""Inferencia de Ilastik (Pixel Classification) en modo headless sobre los 15 fotogramas de evaluación.

Usa las copias de los proyectos en models/ilastik/ (con el Random Forest ya entrenado
y cacheado) en modo --readonly, y exporta la "Simple Segmentation"
(1 = mito, 2 = fondo, 3 = ruido / mito_blur) a outputs/predictions/ilastik/<proyecto>/.

Uso:  python scripts/11_run_ilastik.py
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


def main():
    inputs = [os.path.join(C.RAW_DIR, s + ".tif") for s in C.EVAL15]
    for proj in C.ILASTIK_PROJECT_NAMES:
        out_dir = os.path.join(C.ILASTIK_NEW, proj)
        os.makedirs(out_dir, exist_ok=True)
        cmd = [C.ILASTIK_EXE, "--headless", "--readonly",
               f"--project={os.path.join(C.ILASTIK_PROJECTS, proj + '.ilp')}",
               "--export_source=Simple Segmentation", "--output_format=tif",
               f"--output_filename_format={out_dir}/{{nickname}}.tif"] + inputs
        print(f"[ilastik] {proj}: {len(inputs)} fotogramas ...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:])
            raise SystemExit(f"ilastik falló en {proj}")
        print(f"[ilastik] {proj}: OK -> {out_dir}")


if __name__ == "__main__":
    main()
