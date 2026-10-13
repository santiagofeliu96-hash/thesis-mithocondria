"""Compara las tablas regeneradas (outputs/tables) con los valores de la tesis (expected_thesis.json).

Cada valor numérico se redondea a los decimales con que figura en la tesis:
  OK      redondeo idéntico
  ~1      difiere en una unidad del último decimal (redondeo en el borde)
  FALLA   cualquier otra diferencia
Los conteos y textos (configuración ganadora, par de descriptores) deben ser idénticos.

Salida: outputs/validation_report.md (+ resumen por consola). Exit code 1 si hay FALLAS.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


def decimals(s):
    return len(s.split(".")[1]) if "." in s else 0


def compare(actual, expected):
    """-> (estado, valor real formateado)."""
    try:
        e = float(expected)
        a = float(actual)
    except (TypeError, ValueError):
        return ("OK" if str(actual) == expected else "FALLA"), str(actual)
    d = decimals(expected)
    a_r = round(a + 0.0, d)
    shown = f"{a:.{d}f}" if d else f"{a:g}"
    if abs(a_r - e) < 0.5 * 10 ** -d:
        return "OK", shown
    if abs(a_r - e) < 1.5 * 10 ** -d:
        return "~1", shown
    return "FALLA", shown


def key_match(series, value):
    try:
        return (series.astype(float) - float(value)).abs() < 1e-9
    except ValueError:
        return series.astype(str) == value


def md_table(df):
    """DataFrame -> tabla Markdown (sin depender de tabulate)."""
    fmt = lambda v: f"{v:.3g}" if isinstance(v, float) else str(v)  # noqa: E731
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def run_check(chk):
    df = pd.read_csv(os.path.join(C.TABLES, chk["table"]))
    results = []
    if "agg" in chk:
        for spec, exp in chk["agg"].items():
            col, fn = spec.split(":")
            results.append((spec, col, exp, *compare(getattr(df[col], fn)(), exp)))
        return results
    key_cols = chk["key_col"] if isinstance(chk["key_col"], list) else [chk["key_col"]]
    for key, exps in chk["rows"].items():
        parts = key.split("|")
        sel = pd.Series(True, index=df.index)
        for kc, kv in zip(key_cols, parts):
            sel &= key_match(df[kc], kv)
        row = df[sel]
        for col, exp in zip(chk["cols"], exps):
            if len(row) != 1:
                results.append((key, col, exp, "FALLA", f"<{len(row)} filas>"))
            else:
                results.append((key, col, exp, *compare(row.iloc[0][col], exp)))
    return results


def main():
    with open(os.path.join(C.ROOT, "expected_thesis.json"), encoding="utf-8") as f:
        checks = json.load(f)["checks"]
    lines = ["# Reporte de validación contra la tesis", "",
             "Generado por `scripts/99_validate.py`. Estados: **OK** = mismo valor redondeado a los "
             "decimales de la tesis; **~1** = difiere en una unidad del último decimal; **FALLA** = otra diferencia.", "",
             "| Tabla / dato | Valores | OK | ~1 | FALLA |", "|---|---|---|---|---|"]
    detail, total = [], {"OK": 0, "~1": 0, "FALLA": 0}
    for chk in checks:
        res = run_check(chk)
        cnt = {s: sum(r[3] == s for r in res) for s in total}
        for s in total:
            total[s] += cnt[s]
        lines.append(f"| {chk['label']} | {len(res)} | {cnt['OK']} | {cnt['~1']} | {cnt['FALLA']} |")
        bad = [r for r in res if r[3] != "OK"]
        if bad:
            detail += [f"### {chk['label']} (`{chk['table']}`)", ""]
            if chk.get("nota"):
                detail += [f"> {chk['nota']}", ""]
            detail += [
                       "| Fila | Columna | Tesis | Recalculado | Estado |", "|---|---|---|---|---|"]
            detail += [f"| {k} | {c} | {e} | {a} | {s} |" for k, c, e, s, a in bad] + [""]
    lines.append(f"| **Total** | {sum(total.values())} | {total['OK']} | {total['~1']} | {total['FALLA']} |")
    lines += ["", "## Diferencias", ""] + (detail or ["Ninguna."])

    # --- verificación de predicciones re-inferidas y chequeo cruzado ---
    lines += ["", "## Re-inferencia vs. predicciones originales", ""]
    p = os.path.join(C.TABLES, "check_predictions.csv")
    if os.path.exists(p):
        cp = pd.read_csv(p)
        g = cp.groupby(["tool", "project"]).agg(fotogramas=("frame", "size"), min_frac_igual=("frac_equal", "min"),
                                                max_abs_diff=("max_abs_diff", "max")).reset_index()
        lines += [md_table(g)]
    p = os.path.join(C.TABLES, "seg_crosscheck.csv")
    if os.path.exists(p):
        cr = pd.read_csv(p)
        cr["coincide"] = (cr.ap_absdiff < 1e-6) & (cr.px_f1_absdiff < 1e-6)
        g = cr.groupby("tool").agg(pares=("coincide", "size"), coinciden=("coincide", "sum"),
                                   max_dif_ap=("ap_absdiff", "max")).reset_index()
        lines += ["", "Máscaras guardadas recomputadas vs. CSV de Colab (fotograma x config):", "",
                  md_table(g)]

    report = os.path.join(C.OUTPUTS, "validation_report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:len(checks) + 7]))
    print(f"\nReporte -> {report}")
    sys.exit(1 if total["FALLA"] else 0)


if __name__ == "__main__":
    main()
