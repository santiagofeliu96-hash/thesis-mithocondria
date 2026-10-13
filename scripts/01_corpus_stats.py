"""Composición del corpus anotado (tab:corpus-composicion) y partición por película (tab:res2-particion).

Las instancias se cuentan sobre las máscaras GT (fuente autoritativa) y la clase
individual / cluster se toma de las etiquetas manuales, unidas por (frame, label)
(las instancias "ramificadas" ya están contadas como individuales).

Salidas: outputs/tables/tab_corpus_composicion.csv, tab_res2_particion.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from tesislib import io
from tesislib.labels import load_manual_labels


def main():
    manual = load_manual_labels().drop_duplicates(["frame", "label"]).set_index(["frame", "label"]).clase
    rows = []
    for s in C.ALL_FRAMES:
        ids = np.unique(io.load_gt(s)); ids = ids[ids != 0]
        clases = [manual.get((s, int(i))) for i in ids]
        missing = sum(c is None for c in clases)
        assert missing == 0, f"{s}: {missing} instancias GT sin etiqueta manual"
        rows.append({"pelicula": C.film(s), "fotograma": C.short(s), "instancias": len(ids),
                     "individuales": clases.count("individual"), "clusters": clases.count("cluster")})
    comp = pd.DataFrame(rows)
    tot = comp[["instancias", "individuales", "clusters"]].sum()
    comp.loc[len(comp)] = {"pelicula": "Total", "fotograma": f"{len(rows)} fotogramas", **tot.to_dict()}
    os.makedirs(C.TABLES, exist_ok=True)
    comp.to_csv(os.path.join(C.TABLES, "tab_corpus_composicion.csv"), index=False)

    per = comp.iloc[:-1]
    part = per.groupby("pelicula").agg(fotogramas=("fotograma", "size"), individuales=("individuales", "sum"),
                                       clusters=("clusters", "sum"), instancias=("instancias", "sum"))
    part["conjunto"] = np.where(part.index.isin(C.TEST_FILMS), "Test", "Desarrollo")
    for name in ["Desarrollo", "Test"]:
        sub = part[part.conjunto == name]
        part.loc[name] = [sub.fotogramas.sum(), sub.individuales.sum(), sub.clusters.sum(),
                          sub.instancias.sum(), ""]
    part.to_csv(os.path.join(C.TABLES, "tab_res2_particion.csv"))

    print(comp.to_string(index=False))
    print("\n", part.to_string())
    print("\nIndividuales %.2f%% / clusters %.2f%%" % (100 * tot.individuales / tot.instancias,
                                                       100 * tot.clusters / tot.instancias))


if __name__ == "__main__":
    main()
