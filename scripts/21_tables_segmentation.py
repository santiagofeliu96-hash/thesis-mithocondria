"""Tablas del capítulo "Evaluación de las herramientas de segmentación" (15 fotogramas).

Lee seg_raw_pixel.csv / seg_raw_instance.csv (20_eval_segmentation.py) y promedia por
configuración sobre los 15 fotogramas. La mejor configuración global de cada herramienta
es la que maximiza el F1 de píxel o el AP de instancia promedio (selección independiente
por nivel).

Salidas (outputs/tables/):
  tab_ilastik.csv          tab:ilastik-validos     (métricas de píxel, 4 variantes)
  tab_cellpose.csv         tab:cellpose            (instancia, 48 configs, ordenadas por AP)
  tab_omnipose.csv         tab:omnipose            (instancia, 20 configs, ordenadas por AP)
  tab_modl.csv             tab:modl                (píxel, 19 umbrales, ordenados por F1)
  tab_comp_pixel.csv       tab:comp-pixel
  tab_comp_instancia.csv   tab:comp-instancia
  tab_comp_por_imagen.csv  tab:comp-por-imagen     (AP por fotograma + promedio y desvío)
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

TOOLS = ["Ilastik", "Cellpose-SAM", "Omnipose", "MoDL"]


def mean_by_config(df, tool, cols):
    d = df[df.tool == tool]
    g = d.groupby("config")[cols].mean()
    g["n_frames"] = d.groupby("config").frame.nunique()
    return g


def save(df, name, sort_by=None):
    if sort_by:
        df = df.sort_values(sort_by, ascending=False, kind="mergesort")
    df.to_csv(os.path.join(C.TABLES, name), float_format="%.6f")
    return df


def main():
    px = pd.read_csv(os.path.join(C.TABLES, "seg_raw_pixel.csv"))
    ins = pd.read_csv(os.path.join(C.TABLES, "seg_raw_instance.csv"))
    pxc = ["px_f1", "px_precision", "px_recall", "px_iou"]
    inc = ["precision", "recall", "ap", "error_rate", "f1"]

    n_frames = ins.groupby("tool").frame.nunique()
    assert (n_frames == 15).all(), f"faltan fotogramas: {n_frames.to_dict()}"

    ila = save(mean_by_config(px, "Ilastik", pxc), "tab_ilastik.csv", "px_f1")
    cp = save(mean_by_config(ins, "Cellpose-SAM", inc), "tab_cellpose.csv", "ap")
    om = save(mean_by_config(ins, "Omnipose", inc), "tab_omnipose.csv", "ap")
    md = save(mean_by_config(px, "MoDL", pxc), "tab_modl.csv", "px_f1")

    # --- comparación homogénea: mejor config por nivel ---
    comp_px, comp_in, best_in = [], [], {}
    for t in TOOLS:
        g = mean_by_config(px, t, pxc).sort_values("px_f1", ascending=False, kind="mergesort")
        comp_px.append({"tool": t, "config": g.index[0], "px_f1": g.px_f1.iloc[0], "px_iou": g.px_iou.iloc[0]})
        g = mean_by_config(ins, t, inc).sort_values("ap", ascending=False, kind="mergesort")
        best_in[t] = g.index[0]
        comp_in.append({"tool": t, "config": g.index[0], "ap": g.ap.iloc[0],
                        "error_rate": g.error_rate.iloc[0], "f1": g.f1.iloc[0]})
    comp_px = pd.DataFrame(comp_px).set_index("tool").sort_values("px_f1", ascending=False)
    comp_in = pd.DataFrame(comp_in).set_index("tool").sort_values("ap", ascending=False)
    save(comp_px, "tab_comp_pixel.csv")
    save(comp_in, "tab_comp_instancia.csv")

    # --- AP por fotograma con la mejor config de instancia ---
    order = [C.short(s) for s in C.EVAL15]
    per = pd.DataFrame({t: ins[(ins.tool == t) & (ins.config == best_in[t])].set_index("frame").ap
                        for t in TOOLS}).reindex(order)
    per.loc["Promedio"] = per.loc[order].mean()
    per.loc["Desv. estándar"] = per.loc[order].std()     # desvío muestral (ddof = 1)
    save(per, "tab_comp_por_imagen.csv")

    pd.set_option("display.width", 160)
    fmt = lambda v: "%.3f" % v  # noqa: E731
    print("== Ilastik (píxel) ==\n", ila.to_string(float_format=fmt))
    print("\n== Cellpose-SAM top 10 (instancia) ==\n", cp.head(10).to_string(float_format=fmt))
    print("\n== Omnipose top 10 (instancia) ==\n", om.head(10).to_string(float_format=fmt))
    print("\n== MoDL top 10 (píxel) ==\n", md.head(10).to_string(float_format=fmt))
    print("   rango F1: [%.3f, %.3f]" % (md.px_f1.min(), md.px_f1.max()))
    print("\n== Comparación píxel ==\n", comp_px.to_string(float_format=fmt))
    print("\n== Comparación instancia ==\n", comp_in.to_string(float_format=fmt))
    print("\n== AP por fotograma ==\n", per.to_string(float_format=fmt))


if __name__ == "__main__":
    main()
