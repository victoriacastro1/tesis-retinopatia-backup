# -*- coding: utf-8 -*-
"""
Intervalo de confianza del 95% (bootstrap percentil) del Quadratic Weighted
Kappa (QWK) de EfficientNet-B0, calculado POR PROCEDENCIA DEL DATO (APTOS 2019,
EyePACS, IDRiD), sobre el mismo conjunto FIJO de validacion del Experimento 1
(744 imagenes, sin Messidor, sin entrenamiento).

Motivo: un revisor de tesis senalo que IDRiD tiene solo 25 imagenes de
validacion y que un QWK puntual de 0.869 alli es potencialmente muy inestable;
pide cuantificar la incertidumbre en vez de solo reportar el punto.

No se reentrena el modelo ni se modifican las predicciones: se usan
EXACTAMENTE los y_true / y_pred ya guardados para EfficientNet-B0.

Fuente (y_true, y_pred, dataset):
  resultados_dl/COMBINED_TRAIN_ONLY_aptos_2019-eyePACS-idrid_efficientnet_b0_img384_bs8_lr0.0001_rerun/predicciones_validacion_efficientnet_b0_por_dataset.csv

El bootstrap se hace POR SEPARADO dentro de cada dataset (remuestreo con
reemplazo sobre las N filas de ese dataset unicamente), con semilla 42 y 2000
remuestreos. En cada remuestreo se intenta calcular QWK; si resulta
indefinido (division 0/0, cuando y_true e y_pred del remuestreo son un unico
valor identico -> sklearn devuelve nan con RuntimeWarning), ese remuestreo se
descarta EXPLICITAMENTE (no se reemplaza por 0) y se cuenta como invalido.
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = r"C:\Users\Victoria\Desktop\data\datos_raw"
OUT_DIR = os.path.join(ROOT, "resultados")
os.makedirs(OUT_DIR, exist_ok=True)

PRED_CSV = os.path.join(
    ROOT, "resultados_dl",
    "COMBINED_TRAIN_ONLY_aptos_2019-eyePACS-idrid_efficientnet_b0_img384_bs8_lr0.0001_rerun",
    "predicciones_validacion_efficientnet_b0_por_dataset.csv",
)

N_BOOT = 2000
SEED = 42

# QWK puntual ya reportado (tabla_estabilidad_por_dataset_efficientnet_b0.csv)
# para verificacion cruzada.
QWK_REPORTADO = {
    "aptos_2019": 0.766,
    "eyePACS": 0.767,
    "idrid": 0.869,
}
NOMBRE_TABLA = {
    "aptos_2019": "APTOS 2019",
    "eyePACS": "EyePACS",
    "idrid": "IDRiD",
}
N_ESPERADO = {"aptos_2019": 105, "eyePACS": 614, "idrid": 25}

# ---------------------------------------------------------------------------
# 1) Cargar y_true / y_pred de EfficientNet-B0 (ya guardados, sin recalcular)
# ---------------------------------------------------------------------------
df = pd.read_csv(PRED_CSV)
assert len(df) == 744, f"Se esperaban 744 filas en total, hay {len(df)}"

inconsistencias = []


def qwk_seguro(y_true, y_pred):
    """Devuelve (valor, es_valido). Invalido si sklearn produce nan (0/0)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        val = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    if isinstance(val, float) and np.isnan(val):
        return np.nan, False
    return float(val), True


# ---------------------------------------------------------------------------
# 2) Bootstrap por dataset (remuestreo independiente dentro de cada dataset)
# ---------------------------------------------------------------------------
filas_reporte = []
filas_num = []

for ds_key in ["aptos_2019", "eyePACS", "idrid"]:
    g = df[df["dataset"] == ds_key].reset_index(drop=True)
    n = len(g)

    if n != N_ESPERADO[ds_key]:
        inconsistencias.append(f"{ds_key}: N={n} (se esperaba {N_ESPERADO[ds_key]})")

    y_true = g["y_true"].to_numpy(dtype=int)
    y_pred = g["y_pred"].to_numpy(dtype=int)

    qwk_puntual, valido_puntual = qwk_seguro(y_true, y_pred)
    if not valido_puntual:
        inconsistencias.append(f"{ds_key}: QWK puntual indefinido (no deberia ocurrir con datos reales)")

    rep = QWK_REPORTADO[ds_key]
    if abs(round(qwk_puntual, 3) - rep) > 0.001:
        inconsistencias.append(
            f"{ds_key}: QWK puntual recalculado={qwk_puntual:.4f} vs reportado={rep:.3f} (no coincide)"
        )

    # Bootstrap independiente para este dataset: RNG propio con la MISMA
    # semilla base (42) pero derivada por dataset para que cada dataset tenga
    # su propia matriz de indices (remuestreo separado dentro de cada uno).
    rng = np.random.RandomState(SEED)
    boot_indices = rng.randint(0, n, size=(N_BOOT, n))

    valores = []
    n_invalidos = 0
    for b in range(N_BOOT):
        idx = boot_indices[b]
        v, ok = qwk_seguro(y_true[idx], y_pred[idx])
        if ok:
            valores.append(v)
        else:
            n_invalidos += 1

    valores = np.array(valores)
    n_validos = len(valores)

    if n_validos > 0:
        lo, hi = np.percentile(valores, [2.5, 97.5])
    else:
        lo, hi = np.nan, np.nan
        inconsistencias.append(f"{ds_key}: 0 remuestreos validos de {N_BOOT}; IC no calculable")

    nombre_tabla = NOMBRE_TABLA[ds_key]
    filas_reporte.append({
        "Dataset": nombre_tabla,
        "N": n,
        "QWK (IC 95%)": f"{qwk_puntual:.3f} ({lo:.3f}–{hi:.3f})" if n_validos > 0 else f"{qwk_puntual:.3f} (IC no calculable)",
    })
    filas_num.append({
        "Dataset": nombre_tabla,
        "N": n,
        "QWK_puntual": round(qwk_puntual, 6),
        "QWK_IC_inf": round(float(lo), 6) if n_validos > 0 else np.nan,
        "QWK_IC_sup": round(float(hi), 6) if n_validos > 0 else np.nan,
        "Remuestreos_validos": n_validos,
        "Remuestreos_invalidos": n_invalidos,
        "Remuestreos_totales": N_BOOT,
    })

    print(f"[{nombre_tabla}] N={n}  QWK puntual={qwk_puntual:.4f} (reportado={rep:.3f})  "
          f"validos={n_validos}/{N_BOOT}  invalidos={n_invalidos}  IC95%=({lo:.3f}, {hi:.3f})" if n_validos > 0
          else f"[{nombre_tabla}] N={n}  QWK puntual={qwk_puntual:.4f}  validos={n_validos}/{N_BOOT}  IC no calculable")

tabla_reporte = pd.DataFrame(filas_reporte)
tabla_num = pd.DataFrame(filas_num)

print("\n" + "=" * 90)
print("TABLA FINAL: QWK por procedencia del dato (EfficientNet-B0), IC 95% bootstrap")
print("=" * 90)
print(tabla_reporte.to_string(index=False))

print("\nInconsistencias:")
if inconsistencias:
    for msg in inconsistencias:
        print(" [INCONSISTENCIA]", msg)
else:
    print(" Ninguna: QWK puntuales coinciden con lo reportado, N correctos por dataset.")

# ---------------------------------------------------------------------------
# 3) Guardar resultados
# ---------------------------------------------------------------------------
tabla_reporte.to_csv(os.path.join(OUT_DIR, "qwk_ic_por_dataset_efficientnet_b0.csv"), index=False, encoding="utf-8-sig")
with pd.ExcelWriter(os.path.join(OUT_DIR, "qwk_ic_por_dataset_efficientnet_b0.xlsx"), engine="openpyxl") as writer:
    tabla_reporte.to_excel(writer, sheet_name="tabla_formateada", index=False)
    tabla_num.to_excel(writer, sheet_name="valores_numericos", index=False)

print("\nGuardado:")
print(" ", os.path.join(OUT_DIR, "qwk_ic_por_dataset_efficientnet_b0.csv"))
print(" ", os.path.join(OUT_DIR, "qwk_ic_por_dataset_efficientnet_b0.xlsx"))
