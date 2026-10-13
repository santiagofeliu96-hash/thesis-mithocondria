# Regenera todos los resultados de la tesis y los valida contra el .tex.
# Uso (desde esta carpeta):  powershell -ExecutionPolicy Bypass -File run_all.ps1 [-SkipInference] [-SkipPrepare]
param([switch]$SkipInference, [switch]$SkipPrepare)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
# Entornos: los locales (.venv, .venv_modl; ver requirements*.txt) si existen, si no los originales.
function PickPython($local, $fallback) {
    $p = Join-Path $Root "$local\Scripts\python.exe"
    if (Test-Path $p) { return $p }
    return (Join-Path $Root $fallback)
}
$Py = PickPython ".venv" "..\Code\venv\Scripts\python.exe"                  # análisis (Python 3.10)
$PyTF = PickPython ".venv_modl" "..\MoDL-main\venvtest2\Scripts\python.exe"  # MoDL (Python 3.8 + TF 2.5)
Write-Host "Python análisis: $Py`nPython MoDL:     $PyTF"
$env:PYTHONIOENCODING = "utf-8"

function Step($py, $script, $extra = @()) {
    Write-Host "`n===== $script =====" -ForegroundColor Cyan
    & $py (Join-Path $Root "scripts\$script") @extra
    if ($LASTEXITCODE -ne 0 -and $script -ne "99_validate.py") { throw "$script falló ($LASTEXITCODE)" }
}

# 00 copia los insumos desde las carpetas originales: solo corre si data/ todavía no fue armada.
if (-not $SkipPrepare -and -not (Test-Path (Join-Path $Root "data\manifest.csv"))) { Step $Py "00_prepare_data.py" }
Step $Py "01_corpus_stats.py"
Step $Py "02_interannotator.py"
if (-not $SkipInference) {
    Step $PyTF "10_run_modl.py"
    Step $Py "11_run_ilastik.py"
}
Step $Py "12_check_predictions.py"
Step $Py "20_eval_segmentation.py"
Step $Py "21_tables_segmentation.py"
Step $Py "22_figures_segmentation.py"
Step $Py "30_build_features.py"
Step $Py "31_classify_cascade.py"
Step $Py "32_figures_classification.py"
Step $Py "99_validate.py"
