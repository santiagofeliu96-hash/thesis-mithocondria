# Reporte de validación contra la tesis

Generado por `scripts/99_validate.py`. Estados: **OK** = mismo valor redondeado a los decimales de la tesis; **~1** = difiere en una unidad del último decimal; **FALLA** = otra diferencia.

| Tabla / dato | Valores | OK | ~1 | FALLA |
|---|---|---|---|---|
| tab:corpus-composicion | 66 | 66 | 0 | 0 |
| tab:acuerdo-interanotador | 24 | 24 | 0 | 0 |
| tab:res2-particion | 44 | 44 | 0 | 0 |
| tab:ilastik-validos | 16 | 16 | 0 | 0 |
| tab:cellpose | 40 | 37 | 2 | 1 |
| tab:omnipose | 40 | 24 | 1 | 15 |
| tab:modl | 40 | 24 | 1 | 15 |
| texto MoDL: rango F1 [0.662, 0.757] | 2 | 2 | 0 | 0 |
| tab:comp-pixel | 12 | 12 | 0 | 0 |
| tab:comp-instancia | 12 | 12 | 0 | 0 |
| tab:comp-por-imagen | 68 | 68 | 0 | 0 |
| tab:res2-dist | 16 | 16 | 0 | 0 |
| tab:res2-auc | 24 | 24 | 0 | 0 |
| tab:res2-operacion (desarrollo) | 32 | 32 | 0 | 0 |
| tab:res2-validacion (test P36) | 32 | 32 | 0 | 0 |
| texto/fig:res2-visualizacion: umbrales MoDL pi=0.90 | 2 | 2 | 0 | 0 |
| texto: desglose de las 154 máscaras de test (MoDL, pi=0.90) | 16 | 16 | 0 | 0 |
| **Total** | 486 | 451 | 4 | 31 |

## Diferencias

### tab:cellpose (`tab_cellpose.csv`)

> La salida original de tablas_res1.py (sesión del 2026-09-29, registrada en el transcript) da exactamente los valores recalculados: las filas señaladas están mal transcriptas en la tesis. Las filas superiores y la configuración ganadora coinciden.

| Fila | Columna | Tesis | Recalculado | Estado |
|---|---|---|---|---|
| ft=0.6,ct=0,niter=0 | precision | 0.387 | 0.388 | ~1 |
| ft=0.6,ct=0,niter=0 | recall | 0.384 | 0.385 | ~1 |
| ft=0.6,ct=0,niter=0 | error_rate | 1.121 | 1.119 | FALLA |

### tab:omnipose (`tab_omnipose.csv`)

> La salida original de tablas_res1.py (sesión del 2026-09-29, registrada en el transcript) da exactamente los valores recalculados: las filas señaladas están mal transcriptas en la tesis. Las filas superiores y la configuración ganadora coinciden. Además, con los valores correctos la fila 'bact_fluor_omni, mt=1, ft=0.0' no está en el top 10 (sí 'bact_fluor_omni, mt=-4, ft=0.4', AP=0.200).

| Fila | Columna | Tesis | Recalculado | Estado |
|---|---|---|---|---|
| bact_fluor_omni,mt=1,ft=0.4 | precision | 0.235 | 0.231 | FALLA |
| bact_fluor_omni,mt=1,ft=0.4 | recall | 0.615 | 0.610 | FALLA |
| bact_fluor_omni,mt=1,ft=0.4 | ap | 0.210 | 0.208 | FALLA |
| bact_fluor_omni,mt=1,ft=0.4 | error_rate | 3.810 | 3.570 | FALLA |
| bact_fluor_omni,mt=-1,ft=0.4 | precision | 0.228 | 0.226 | FALLA |
| bact_fluor_omni,mt=-1,ft=0.4 | recall | 0.695 | 0.712 | FALLA |
| bact_fluor_omni,mt=-1,ft=0.4 | ap | 0.208 | 0.207 | ~1 |
| bact_fluor_omni,mt=-1,ft=0.4 | error_rate | 3.890 | 4.026 | FALLA |
| bact_fluor_omni,mt=-2,ft=0.4 | precision | 0.205 | 0.198 | FALLA |
| bact_fluor_omni,mt=-2,ft=0.4 | recall | 0.728 | 0.730 | FALLA |
| bact_fluor_omni,mt=-2,ft=0.4 | ap | 0.189 | 0.184 | FALLA |
| bact_fluor_omni,mt=-2,ft=0.4 | error_rate | 4.250 | 4.527 | FALLA |
| bact_fluor_omni,mt=1,ft=0.0 | precision | 0.210 | 0.196 | FALLA |
| bact_fluor_omni,mt=1,ft=0.0 | recall | 0.617 | 0.612 | FALLA |
| bact_fluor_omni,mt=1,ft=0.0 | ap | 0.185 | 0.179 | FALLA |
| bact_fluor_omni,mt=1,ft=0.0 | error_rate | 4.310 | 4.496 | FALLA |

### tab:modl (`tab_modl.csv`)

> La salida original de tablas_res1.py (sesión del 2026-09-29, registrada en el transcript) da exactamente los valores recalculados: las filas señaladas están mal transcriptas en la tesis. Las filas superiores y la configuración ganadora coinciden. El orden correcto del top 10 tras thr=0.20 es 0.50, 0.55, 0.15, 0.60.

| Fila | Columna | Tesis | Recalculado | Estado |
|---|---|---|---|---|
| thr=0.50 | px_f1 | 0.751 | 0.754 | FALLA |
| thr=0.50 | px_precision | 0.763 | 0.761 | FALLA |
| thr=0.50 | px_recall | 0.789 | 0.793 | FALLA |
| thr=0.50 | px_iou | 0.621 | 0.622 | ~1 |
| thr=0.15 | px_f1 | 0.748 | 0.751 | FALLA |
| thr=0.15 | px_precision | 0.693 | 0.696 | FALLA |
| thr=0.15 | px_recall | 0.875 | 0.873 | FALLA |
| thr=0.15 | px_iou | 0.621 | 0.625 | FALLA |
| thr=0.55 | px_f1 | 0.746 | 0.751 | FALLA |
| thr=0.55 | px_precision | 0.772 | 0.768 | FALLA |
| thr=0.55 | px_recall | 0.772 | 0.780 | FALLA |
| thr=0.55 | px_iou | 0.614 | 0.618 | FALLA |
| thr=0.60 | px_f1 | 0.739 | 0.749 | FALLA |
| thr=0.60 | px_precision | 0.783 | 0.775 | FALLA |
| thr=0.60 | px_recall | 0.751 | 0.767 | FALLA |
| thr=0.60 | px_iou | 0.605 | 0.614 | FALLA |


## Re-inferencia vs. predicciones originales

| tool | project | fotogramas | min_frac_igual | max_abs_diff |
|---|---|---|---|---|
| Ilastik | 2mito_3classes | 15 | 1 | nan |
| Ilastik | v2_2classes | 15 | 1 | nan |
| Ilastik | v2_3classes | 15 | 1 | nan |
| MoDL | U-RNet+ | 21 | 1 | 2.44e-06 |

Máscaras guardadas recomputadas vs. CSV de Colab (fotograma x config):

| tool | pares | coinciden | max_dif_ap |
|---|---|---|---|
| Cellpose-SAM | 396 | 129 | 0.28 |
| Omnipose | 75 | 75 | 4.84e-13 |
