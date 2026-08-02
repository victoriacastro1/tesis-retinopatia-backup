# -*- coding: utf-8 -*-
"""
Calcula la distancia absoluta del error ordinal (|clase_real - clase_predicha|)
para RandomForest, XGBoost, ResNet34 e InceptionV3, usando EXCLUSIVAMENTE las
predicciones ya guardadas en disco sobre el conjunto fijo de validacion (744
imagenes, aptos_2019 + eyePACS + idrid, SIN Messidor) reportado en la tesis.

Fuentes (y_true / y_pred) verificadas previamente:
- RandomForest / XGBoost:
    resultados_ml/COMBINED_TRAIN_ONLY/predicciones_validacion_ml_por_dataset.csv
    (columna 'Modelo' filtra cada modelo; columnas 'y_true', 'y_pred')
- ResNet34:
    resultados_dl/COMBINED_TRAIN_ONLY_aptos_2019-eyePACS-idrid_resnet34_img384_bs8_lr0.0003_no_messidor/figuras_tesis/ResNet34_predicciones_validacion.csv
- InceptionV3:
    resultados_dl/COMBINED_TRAIN_ONLY_aptos_2019-eyePACS-idrid_inception_v3_img299_bs8_lr0.0003_no_messidor/figuras_tesis/InceptionV3_predicciones_validacion.csv

Se verifico que las 744 imagenes son EXACTAMENTE las mismas (mismo orden e
identidad) entre ML y DL, y que corresponden al split fijo generado con
StratifiedShuffleSplit(seed=42, test_size=0.15) sobre el df_all combinado
(aptos_2019 + eyePACS + idrid, tope de balanceo 991/clase), tal como lo exige
el notebook DL FINAL (assert len(df_va) == 744; assert 'messidor' not in DATASETS).
"""
import os
import numpy as np
import pandas as pd

ROOT = r"C:\Users\Victoria\Desktop\data\datos_raw"
OUT_DIR = os.path.join(ROOT, "resultados")
os.makedirs(OUT_DIR, exist_ok=True)

ML_PRED_PATH = os.path.join(ROOT, "resultados_ml", "COMBINED_TRAIN_ONLY", "predicciones_validacion_ml_por_dataset.csv")
RESNET_PRED_PATH = os.path.join(ROOT, "resultados_dl",
    "COMBINED_TRAIN_ONLY_aptos_2019-eyePACS-idrid_resnet34_img384_bs8_lr0.0003_no_messidor",
    "figuras_tesis", "ResNet34_predicciones_validacion.csv")
INCEPTION_PRED_PATH = os.path.join(ROOT, "resultados_dl",
    "COMBINED_TRAIN_ONLY_aptos_2019-eyePACS-idrid_inception_v3_img299_bs8_lr0.0003_no_messidor",
    "figuras_tesis", "InceptionV3_predicciones_validacion.csv")

# Reportes de clasificacion ya guardados (para verificar accuracy vs d=0)
RF_REPORT = os.path.join(ROOT, "resultados_ml", "COMBINED_TRAIN_ONLY", "RandomForest_classification_report.csv")
XGB_REPORT = os.path.join(ROOT, "resultados_ml", "COMBINED_TRAIN_ONLY", "XGBoost_classification_report.csv")
RESNET_REPORT = os.path.join(ROOT, "resultados_dl",
    "COMBINED_TRAIN_ONLY_aptos_2019-eyePACS-idrid_resnet34_img384_bs8_lr0.0003_no_messidor",
    "figuras_tesis", "ResNet34_classification_report_val.csv")
INCEPTION_REPORT = os.path.join(ROOT, "resultados_dl",
    "COMBINED_TRAIN_ONLY_aptos_2019-eyePACS-idrid_inception_v3_img299_bs8_lr0.0003_no_messidor",
    "figuras_tesis", "InceptionV3_classification_report_val.csv")


def cargar_ml(modelo):
    df = pd.read_csv(ML_PRED_PATH)
    df = df[df["Modelo"] == modelo].reset_index(drop=True)
    return df["y_true"].to_numpy(dtype=int), df["y_pred"].to_numpy(dtype=int)


def cargar_dl(path):
    df = pd.read_csv(path)
    return df["y_true"].to_numpy(dtype=int), df["y_pred"].to_numpy(dtype=int)


def accuracy_reportada(path):
    rep = pd.read_csv(path, index_col=0)
    return float(rep.loc["accuracy"].iloc[0])


modelos = {
    "Random Forest": cargar_ml("RandomForest"),
    "XGBoost": cargar_ml("XGBoost"),
    "ResNet34": cargar_dl(RESNET_PRED_PATH),
    "InceptionV3": cargar_dl(INCEPTION_PRED_PATH),
}

accuracy_reportes = {
    "Random Forest": accuracy_reportada(RF_REPORT),
    "XGBoost": accuracy_reportada(XGB_REPORT),
    "ResNet34": accuracy_reportada(RESNET_REPORT),
    "InceptionV3": accuracy_reportada(INCEPTION_REPORT),
}

filas = []
inconsistencias = []

for nombre, (y_true, y_pred) in modelos.items():
    n = len(y_true)
    dist = np.abs(y_true - y_pred)

    pct = {d: float((dist == d).mean() * 100) for d in range(5)}
    error_le1 = float((dist <= 1).mean() * 100)
    mae = float(dist.mean())

    filas.append({
        "Modelo": nombre,
        "N": n,
        "d=0 (%)": pct[0],
        "d=1 (%)": pct[1],
        "d=2 (%)": pct[2],
        "d=3 (%)": pct[3],
        "d=4 (%)": pct[4],
        "Error <=1 (%)": error_le1,
        "Error absoluto medio": mae,
    })

    # --- Verificaciones ---
    suma_pct = pct[0] + pct[1] + pct[2] + pct[3] + pct[4]
    if abs(suma_pct - 100.0) > 0.05:
        inconsistencias.append(f"{nombre}: suma de porcentajes d0..d4 = {suma_pct:.4f}% (esperado ~100%)")

    acc_reportado = accuracy_reportes[nombre]
    acc_calculado = pct[0] / 100.0
    if abs(acc_reportado - acc_calculado) > 0.005:
        inconsistencias.append(
            f"{nombre}: accuracy reportado={acc_reportado:.4f} vs d=0 calculado={acc_calculado:.4f} (diferencia > 0.005)"
        )

    if n != 744:
        inconsistencias.append(f"{nombre}: N={n} (se esperaban 744 imagenes de validacion)")

tabla = pd.DataFrame(filas)

# Verificacion cruzada: mismas 744 imagenes / mismo orden para todos los modelos
y_true_rf, _ = modelos["Random Forest"]
y_true_xgb, _ = modelos["XGBoost"]
y_true_resnet, _ = modelos["ResNet34"]
y_true_incep, _ = modelos["InceptionV3"]

if not (np.array_equal(y_true_rf, y_true_xgb) and
        np.array_equal(y_true_rf, y_true_resnet) and
        np.array_equal(y_true_rf, y_true_incep)):
    inconsistencias.append("Los y_true (clase real, en el mismo orden) NO coinciden exactamente entre los 4 modelos.")

print("=" * 90)
print("TABLA: Distancia absoluta del error ordinal (conjunto fijo de validacion, 744 imagenes)")
print("=" * 90)
with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 140):
    print(tabla.to_string(index=False))

print("\nVerificaciones:")
if inconsistencias:
    for msg in inconsistencias:
        print(f"  [INCONSISTENCIA] {msg}")
else:
    print("  OK: sumas ~100%, d=0 coincide con accuracy reportado (tolerancia 0.5pp), N=744 y mismas imagenes/orden en los 4 modelos.")

csv_path = os.path.join(OUT_DIR, "distancia_error_ordinal.csv")
xlsx_path = os.path.join(OUT_DIR, "distancia_error_ordinal.xlsx")
tabla.to_csv(csv_path, index=False)
tabla.to_excel(xlsx_path, index=False)
print(f"\nGuardado: {csv_path}")
print(f"Guardado: {xlsx_path}")
