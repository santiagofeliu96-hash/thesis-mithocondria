# Código reproducible de la tesis

*Segmentación y clasificación de mitocondrias en imágenes de microscopía de fluorescencia* (Santiago Feliu).

Esta carpeta reúne solo el código y los datos necesarios para regenerar las tablas y figuras de
resultados de la tesis, y los valida automáticamente contra los números del `.tex`. Los datos se
copiaron desde `Code/` sin modificar nada del original (ver `data/manifest.csv`).

## Cómo reproducir

```powershell
powershell -ExecutionPolicy Bypass -File run_all.ps1            # todo, con re-inferencia (~10 min)
powershell -ExecutionPolicy Bypass -File run_all.ps1 -SkipInference -SkipPrepare   # solo análisis
```

La carpeta es autocontenida: `data/`, `models/` y los entornos tienen todo lo necesario. La única
dependencia externa es la instalación de Ilastik (solo para `11_run_ilastik.py`; sin ella, usar
`-SkipInference`). `00_prepare_data.py` solo se usó para armar `data/` desde las carpetas originales y
`run_all.ps1` lo saltea si `data/manifest.csv` ya existe.

### Modelo de MoDL (no incluido en el repositorio)

`models/U-RNet+.hdf5` pesa 478 MB y supera el límite de GitHub, por lo que no se versiona. Son los pesos
preentrenados oficiales de MoDL (licencia GPL-3.0, <https://github.com/OBPNPW2024/MoDL>):

1. Descargar el modelo desde Zenodo: <https://zenodo.org/records/10889134>.
2. Descomprimirlo y copiar `U-RNet+.hdf5` a `models/U-RNet+.hdf5`.

Solo lo necesita `10_run_modl.py`. Sin el modelo se puede correr todo el resto con
`run_all.ps1 -SkipInference` (usa las probabilidades de MoDL guardadas en `data/stored_predictions/`).

### Entornos

Se usan dos entornos virtuales, porque MoDL necesita TensorFlow 2.5 (solo Python ≤ 3.8):

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt          # análisis: todos los scripts salvo 10

py -3.8 -m venv .venv_modl
.venv_modl\Scripts\python.exe -m pip install -r requirements_modl.txt # solo 10_run_modl.py
```

`run_all.ps1` usa `.venv` y `.venv_modl` si existen; si no, cae en los entornos originales
(`../Code/venv` y `../MoDL-main/venvtest2`). Ambos requirements tienen las versiones exactas con las que
se generaron y validaron los resultados.

Además, `11_run_ilastik.py` usa `C:\Program Files\ilastik-1.4.2` en modo headless (no es un paquete de pip).

El resultado de la validación queda en `outputs/validation_report.md`.

## Qué se re-ejecuta y qué no

| Herramienta | Inferencia | Evaluación |
|---|---|---|
| MoDL | **se re-ejecuta** (`10_run_modl.py`, 21 fotogramas) | desde las probabilidades nuevas |
| Ilastik | **se re-ejecuta** (`11_run_ilastik.py`, 3 proyectos × 15 fotogramas) | desde las máscaras nuevas |
| Cellpose-SAM | no (necesita GPU; corrió en Colab, `notebooks/Cellpose_evaluation*.ipynb`) | desde los CSV por imagen de Colab |
| Omnipose | no (corrió en Colab, `notebooks/Omnipose_evaluation_eval15.ipynb`) | desde los CSV de Colab; máscaras top-5 para figuras y para la cascada |

La re-inferencia reproduce las predicciones originales: Ilastik idéntico píxel a píxel y MoDL con una
diferencia máxima de probabilidad de ~2·10⁻⁶ (ningún píxel cambia al binarizar).

## Estructura

```
config.py              rutas, listas de fotogramas, parámetros fijos (tau, umbrales, test = P36, semilla)
expected_thesis.json   valores transcriptos de la tesis (entrada de 99_validate.py)
tesislib/              io (carga), metrics (píxel + instancia), features, labels (etiqueta operacional),
                       cascade (regresión logística + barrido de umbrales), plots
scripts/               pasos numerados del pipeline (ver tabla)
data/                  crudas, GT, etiquetas manuales, anotaciones inter-anotador, predicciones guardadas
models/                U-RNet+.hdf5 (MoDL) y copias de los proyectos .ilp de Ilastik
notebooks/             notebooks de Colab (referencia; no se ejecutan localmente)
outputs/               predictions/, tables/, figures/, validation_report.md
```

## Mapa tesis → script → salida

| Tesis | Script | Salida (`outputs/`) |
|---|---|---|
| tab:corpus-composicion, tab:res2-particion | `01_corpus_stats.py` | `tables/tab_corpus_composicion.csv`, `tab_res2_particion.csv` |
| tab:acuerdo-interanotador | `02_interannotator.py` | `tables/tab_acuerdo_interanotador.csv` |
| tab:ilastik-validos | `20` + `21_tables_segmentation.py` | `tables/tab_ilastik.csv` |
| tab:cellpose | `20` + `21` | `tables/tab_cellpose.csv` |
| tab:omnipose | `20` + `21` | `tables/tab_omnipose.csv` |
| tab:modl | `20` + `21` | `tables/tab_modl.csv` |
| tab:comp-pixel, tab:comp-instancia, tab:comp-por-imagen | `21` | `tables/tab_comp_*.csv` |
| fig:resultado_ilastik_grilla, fig:modl-instance, fig:res-modl-barrido, fig:cellpose-instance-matching, fig:cyto2_instance, fig:bact_fluor_instance | `22_figures_segmentation.py` | `figures/*.png` |
| tab:res2-dist, tab:res2-auc, tab:res2-operacion, tab:res2-validacion, desglose del test | `30_build_features.py` + `31_classify_cascade.py` | `tables/tab_res2_*.csv` |
| fig:res2-correlacion, fig:res2-boxplots-*, fig:res2-scatter, fig:res2-frontera | `32_figures_classification.py` | `figures/*.png` |

## Observaciones de la validación

- Todo coincide salvo **filas inferiores de tres tablas top-10** (tab:cellpose fila ft=0.6/ct=0/niter=0;
  tab:omnipose filas bact_fluor 7-10; tab:modl filas thr 0.50/0.15/0.55/0.60). La salida original del
  script que generó esas tablas (registrada en la sesión del 2026-09-29) da los mismos valores que este
  código, así que son errores de transcripción en la tesis. Las configuraciones ganadoras y las filas
  superiores son correctas. Valores correctos en `outputs/validation_report.md`.
- Las máscaras locales de Cellpose-SAM (`data/stored_predictions/cellpose/masks`, corrida de 2025)
  **no** son las que produjeron los CSV de Colab usados en las tablas: solo coinciden 129 de 396 pares
  (fotograma × configuración), casi todos configuraciones que dan máscara vacía. La figura
  fig:cellpose-instance-matching (generada con esas máscaras) no corresponde exactamente a tab:cellpose.
- Figuras sin script (retocadas a mano o diagramas): `*_grande.png`, `*_super_big.png`,
  `Resultados-obtenidos.png`, `modelo-cascada.png`, `clases_existentes.png`, `clases_cluster_biologico.png`
  y las figuras del marco teórico. `31_classify_cascade.py` exporta `cascade_test_decisions_modl.csv`
  (decisión por máscara de test), que es la base de `Resultados-obtenidos.png`.
- En el texto de la tesis (sec. de evaluación en test) dice "el clasificador en cascada **remueve** 97";
  son las 97 máscaras **retenidas** (86 individuales, 8 clusters, 3 FP).
- La selección de la configuración se hace sobre los mismos 15 fotogramas en que se reporta (como en la
  tesis); queda señalado en un `\todo` del `.tex`.
