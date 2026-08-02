# -*- coding: utf-8 -*-
"""
Distancia absoluta del error ordinal (|clase_real - clase_predicha|) para
Random Forest, XGBoost, EfficientNet-B0 e InceptionV3, usando EXCLUSIVAMENTE
las predicciones ya guardadas sobre el mismo conjunto FIJO de validacion del
Experimento 1 (744 imagenes: aptos_2019 + eyePACS + idrid, sin Messidor).

No se reentrena ningun modelo.

Fuentes (y_true / y_pred):
- Random Forest / XGBoost:
    resultados_ml/COMBINED_TRAIN_ONLY/predicciones_validacion_ml_por_dataset.csv
    (columna 'Modelo' filtra cada modelo)
- EfficientNet-B0:
    resultados_dl/COMBINED_TRAIN_ONLY_aptos_2019-eyePACS-idrid_efficientnet_b0_img384_bs8_lr0.0001_rerun/predicciones_validacion_efficientnet_b0_por_dataset.csv
- InceptionV3:
    resultados_dl/COMBINED_TRAIN_ONLY_aptos_2019-eyePACS-idrid_inception_v3_img299_bs8_lr0.0003_no_messidor/figuras_tesis/InceptionV3_predicciones_validacion.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = r"C:\Users\Victoria\Desktop\data\datos_raw"
OUT_DIR = os.path.join(ROOT, "resultados")
os.makedirs(OUT_DIR, exist_ok=True)

ML_PRED_PATH = os.path.join(ROOT, "resultados_ml", "COMBINED_TRAIN_ONLY", "predicciones_validacion_ml_por_dataset.csv")
EFFNET_PRED_PATH = os.path.join(ROOT, "resultados_dl",
    "COMBINED_TRAIN_ONLY_aptos_2019-eyePACS-idrid_efficientnet_b0_img384_bs8_lr0.0001_rerun",
    "predicciones_validacion_efficientnet_b0_por_dataset.csv")
INCEPTION_PRED_PATH = os.path.join(ROOT, "resultados_dl",
    "COMBINED_TRAIN_ONLY_aptos_2019-eyePACS-idrid_inception_v3_img299_bs8_lr0.0003_no_messidor",
    "figuras_tesis", "InceptionV3_predicciones_validacion.csv")

# Valores ya conocidos/reportados para verificacion cruzada
VERIF_LE1 = {
    "Random Forest": 81.85,
    "XGBoost": 83.74,
    "InceptionV3": 91.80,
    "EfficientNet-B0": 90.3,   # aproximado
}
VERIF_LE2 = {
    "EfficientNet-B0": 98.5,   # aproximado
}


def cargar_ml(modelo):
    df = pd.read_csv(ML_PRED_PATH)
    df = df[df["Modelo"] == modelo].reset_index(drop=True)
    return df["y_true"].to_numpy(dtype=int), df["y_pred"].to_numpy(dtype=int)


def cargar_dl(path):
    df = pd.read_csv(path)
    return df["y_true"].to_numpy(dtype=int), df["y_pred"].to_numpy(dtype=int)


modelos = {
    "Random Forest": cargar_ml("RandomForest"),
    "XGBoost": cargar_ml("XGBoost"),
    "EfficientNet-B0": cargar_dl(EFFNET_PRED_PATH),
    "InceptionV3": cargar_dl(INCEPTION_PRED_PATH),
}

filas = []
inconsistencias = []
y_true_ref = None

for nombre, (y_true, y_pred) in modelos.items():
    n = len(y_true)
    if y_true_ref is None:
        y_true_ref = (nombre, y_true)
    elif len(y_true) == len(y_true_ref[1]) and not np.array_equal(y_true, y_true_ref[1]):
        inconsistencias.append(f"{nombre}: y_true no coincide con {y_true_ref[0]} (mismo N pero distinto orden/valores)")

    dist = np.abs(y_true - y_pred)
    pct = {d: float((dist == d).mean() * 100) for d in range(5)}
    le1 = float((dist <= 1).mean() * 100)
    le2 = float((dist <= 2).mean() * 100)
    mae = float(dist.mean())

    errores = dist[dist > 0]  # solo predicciones incorrectas
    pct_adyacentes_entre_errores = float((errores == 1).mean() * 100) if len(errores) > 0 else float("nan")

    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    filas.append({
        "Modelo": nombre,
        "N": n,
        "QWK": qwk,
        "d=0 (%)": pct[0],
        "d=1 (%)": pct[1],
        "d=2 (%)": pct[2],
        "d=3 (%)": pct[3],
        "d=4 (%)": pct[4],
        "Distancia ≤1 (%)": le1,
        "Distancia ≤2 (%)": le2,
        "% errores adyacentes (entre incorrectas)": pct_adyacentes_entre_errores,
        "Error absoluto medio": mae,
    })

    # Verificaciones
    suma = pct[0] + pct[1] + pct[2] + pct[3] + pct[4]
    if abs(suma - 100.0) > 0.05:
        inconsistencias.append(f"{nombre}: suma d0..d4 = {suma:.4f}% (esperado ~100%)")

    if n != 744:
        inconsistencias.append(f"{nombre}: N={n} (se esperaban 744)")

    ref_le1 = VERIF_LE1.get(nombre)
    if ref_le1 is not None:
        diff = abs(le1 - ref_le1)
        tol = 0.1 if nombre in ("Random Forest", "XGBoost", "InceptionV3") else 0.15  # EfficientNet-B0 es "≈"
        if diff > tol:
            inconsistencias.append(
                f"{nombre}: distancia<=1 calculada={le1:.2f}% vs valor ya conocido={ref_le1}% (diff={diff:.3f}pp)"
            )
        else:
            print(f"[OK] {nombre}: distancia<=1 calculada={le1:.2f}% coincide con valor ya conocido={ref_le1}% (diff={diff:.3f}pp)")

    ref_le2 = VERIF_LE2.get(nombre)
    if ref_le2 is not None:
        diff = abs(le2 - ref_le2)
        if diff > 0.15:
            inconsistencias.append(
                f"{nombre}: distancia<=2 calculada={le2:.2f}% vs valor ya conocido={ref_le2}% (diff={diff:.3f}pp)"
            )
        else:
            print(f"[OK] {nombre}: distancia<=2 calculada={le2:.2f}% coincide con valor ya conocido={ref_le2}% (diff={diff:.3f}pp)")

tabla = pd.DataFrame(filas)

print("\n" + "=" * 130)
print("TABLA: Distancia absoluta del error ordinal + QWK (744 imagenes, conjunto fijo de validacion)")
print("=" * 130)
with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 200, "display.max_columns", 20):
    print(tabla.to_string(index=False))

print("\nInconsistencias:")
if inconsistencias:
    for msg in inconsistencias:
        print(" [INCONSISTENCIA]", msg)
else:
    print(" Ninguna: todos los valores ya conocidos coinciden dentro de tolerancia de redondeo.")

csv_path = os.path.join(OUT_DIR, "distancia_error_ordinal_4modelos.csv")
xlsx_path = os.path.join(OUT_DIR, "distancia_error_ordinal_4modelos.xlsx")
tabla.to_csv(csv_path, index=False, encoding="utf-8-sig")
tabla.to_excel(xlsx_path, index=False)
print("\nGuardado:")
print(" ", csv_path)
print(" ", xlsx_path)
