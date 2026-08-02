# -*- coding: utf-8 -*-
"""
Metricas globales (Accuracy, Precision macro, Recall macro, F1 macro,
AUC-ROC macro OvR, QWK) con intervalos de confianza del 95% por bootstrap
no parametrico, para los 8 modelos finales de la tesis, sobre el conjunto
FIJO de validacion del Experimento 1 (744 imagenes: aptos_2019 + eyePACS +
idrid, SIN Messidor).

No se reentrena ningun modelo. Las predicciones (y_pred) y probabilidades /
puntajes (y_prob o decision_function) se leen desde la cache generada por
inferencia (NO entrenamiento) con los checkpoints/modelos ya guardados:

  - scratch_build_ml_proba.py  -> resultados/cache_probas/{RandomForest,
    LogisticRegression, LinearSVC, XGBoost}_proba.npz
  - scratch_build_dl_proba.py  -> resultados/cache_probas/{MobileNetV3,
    EfficientNet-B0, ResNet34, InceptionV3}_proba.npz

Cada .npz contiene: y_true, y_pred_saved (el y_pred ya reportado en la tesis),
y_pred_recomputado (obtenido al re-ejecutar inferencia con el modelo/checkpoint
ya entrenado), y_score (probabilidades o decision_function), score_type.
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score,
)
from sklearn.preprocessing import label_binarize

ROOT = r"C:\Users\Victoria\Desktop\data\datos_raw"
CACHE_DIR = os.path.join(ROOT, "resultados", "cache_probas")
OUT_DIR = os.path.join(ROOT, "resultados")
os.makedirs(OUT_DIR, exist_ok=True)

N_CLASSES = 5
CLASSES = np.arange(N_CLASSES)
N_BOOT = 2000
SEED = 42
N_OBS = 744

MODELOS = [
    ("LinearSVC", "LinearSVC"),
    ("Regresión Logística", "LogisticRegression"),
    ("Random Forest", "RandomForest"),
    ("XGBoost", "XGBoost"),
    ("MobileNetV3", "MobileNetV3"),
    ("EfficientNet-B0", "EfficientNet-B0"),
    ("ResNet34", "ResNet34"),
    ("InceptionV3", "InceptionV3"),
]

# Valores publicados en la tesis para verificacion cruzada (donde existen)
# Fuente: resultados_ml/COMBINED_TRAIN_ONLY/models_comparison_summary.csv
#         resultados_dl/summary_dl_no_messidor_aptos_2019-eyePACS-idrid.csv
REPORTADO = {
    "LinearSVC": dict(accuracy=0.293011, qwk=0.207752, auc_macro=0.558044),
    "LogisticRegression": dict(accuracy=0.297043, qwk=0.229557, auc_macro=0.619828),
    "RandomForest": dict(accuracy=0.686828, qwk=0.682337, auc_macro=0.891412),
    "XGBoost": dict(accuracy=0.680108, qwk=0.702713, auc_macro=0.890618),
    "MobileNetV3": dict(accuracy=0.579, qwk=0.798, auc_macro=0.868),
    "ResNet34": dict(accuracy=0.573, qwk=0.800, auc_macro=0.862),
    "InceptionV3": dict(accuracy=0.558, qwk=0.802, auc_macro=0.855),
    # EfficientNet-B0: no hay fila consolidada de AUC/QWK global en resultados_dl;
    # solo accuracy en classification_report_val.csv (0.549731). Se deja como
    # inconsistencia/ausencia de referencia y se reporta en el resumen final.
    "EfficientNet-B0": dict(accuracy=0.549731, qwk=None, auc_macro=None),
}

# ---------------------------------------------------------------------------
# 1) Cargar y_true / y_pred / y_score de cada modelo, verificar consistencia
# ---------------------------------------------------------------------------
datos = {}
y_true_ref = None
inconsistencias = []

for nombre, key in MODELOS:
    path = os.path.join(CACHE_DIR, f"{key}_proba.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Falta cache de probabilidades para {nombre}: {path}")
    d = np.load(path, allow_pickle=True)
    y_true = d["y_true"].astype(int)
    y_pred = d["y_pred_saved"].astype(int)
    y_pred_recomp = d["y_pred_recomputado"].astype(int)
    y_score = d["y_score"]
    score_type = str(d["score_type"])

    if len(y_true) != N_OBS:
        inconsistencias.append(f"{nombre}: N={len(y_true)} (se esperaban {N_OBS})")

    if y_true_ref is None:
        y_true_ref = y_true
    elif not np.array_equal(y_true, y_true_ref):
        inconsistencias.append(f"{nombre}: y_true no coincide en orden/valores con el resto de modelos")

    coincidencia = float((y_pred == y_pred_recomp).mean())
    if coincidencia < 1.0:
        inconsistencias.append(
            f"{nombre}: y_pred recomputado por inferencia difiere del y_pred guardado en "
            f"{(1-coincidencia)*100:.2f}% de las 744 imagenes (posible no-determinismo de "
            f"inferencia GPU/CPU); se usa el y_pred YA GUARDADO/reportado, no el recomputado."
        )

    datos[key] = dict(nombre=nombre, y_true=y_true, y_pred=y_pred, y_score=y_score, score_type=score_type)

print(f"[OK] y_true identico entre los {len(MODELOS)} modelos: {y_true_ref is not None}")


# ---------------------------------------------------------------------------
# 2) Funciones de metrica puntual
# ---------------------------------------------------------------------------
def metricas_puntuales(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)

    auc = np.nan
    try:
        y_true_bin = label_binarize(y_true, classes=CLASSES)
        # AUC solo valido si estan presentes las >=2 clases requeridas por cada columna OvR
        if y_true_bin.sum(axis=0).min() > 0 and (y_true_bin.sum(axis=0) < len(y_true)).all():
            auc = roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr")
    except Exception:
        auc = np.nan

    qwk = np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)
            val = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        if not np.isnan(val):
            qwk = val
    except Exception:
        qwk = np.nan

    return dict(accuracy=acc, precision_macro=prec, recall_macro=rec, f1_macro=f1,
                auc_macro=auc, qwk=qwk)


# ---------------------------------------------------------------------------
# 3) Bootstrap: MISMOS indices para todos los modelos (misma particion)
# ---------------------------------------------------------------------------
rng = np.random.RandomState(SEED)
boot_indices = rng.randint(0, N_OBS, size=(N_BOOT, N_OBS))
np.save(os.path.join(CACHE_DIR, "boot_indices.npy"), boot_indices)

metricas_nombres = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "auc_macro", "qwk"]

resultados_puntuales = {}
resultados_boot = {}  # key -> {metrica: array de valores validos}
resultados_boot_validos = {}  # key -> {metrica: n validos}

for nombre, key in MODELOS:
    y_true = datos[key]["y_true"]
    y_pred = datos[key]["y_pred"]
    y_score = datos[key]["y_score"]

    punt = metricas_puntuales(y_true, y_pred, y_score)
    resultados_puntuales[key] = punt

    boot_vals = {m: [] for m in metricas_nombres}
    for b in range(N_BOOT):
        idx = boot_indices[b]
        yt, yp, ys = y_true[idx], y_pred[idx], y_score[idx]
        m = metricas_puntuales(yt, yp, ys)
        for met in metricas_nombres:
            v = m[met]
            if not (isinstance(v, float) and np.isnan(v)):
                boot_vals[met].append(v)

    resultados_boot[key] = {met: np.array(vals) for met, vals in boot_vals.items()}
    resultados_boot_validos[key] = {met: len(vals) for met, vals in boot_vals.items()}
    print(f"[{nombre}] bootstrap listo. Validos -> AUC: {resultados_boot_validos[key]['auc_macro']}/{N_BOOT}  "
          f"QWK: {resultados_boot_validos[key]['qwk']}/{N_BOOT}")


# ---------------------------------------------------------------------------
# 4) Verificacion cruzada contra valores reportados en la tesis
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("VERIFICACION vs valores reportados en la tesis (tolerancia redondeo)")
print("=" * 100)
for nombre, key in MODELOS:
    rep = REPORTADO.get(key, {})
    punt = resultados_puntuales[key]
    for metname, tol in [("accuracy", 0.005), ("qwk", 0.02), ("auc_macro", 0.02)]:
        rep_val = rep.get(metname)
        if rep_val is None:
            inconsistencias.append(f"{nombre}: no hay valor de referencia reportado para '{metname}' "
                                    f"(no existe fila consolidada global en resultados_dl para EfficientNet-B0); "
                                    f"valor recalculado = {punt[metname]:.4f} (no se puede verificar, no se inventa referencia).")
            continue
        calc_val = punt[metname]
        if np.isnan(calc_val) or abs(calc_val - rep_val) > tol:
            inconsistencias.append(
                f"{nombre}: {metname} recalculado={calc_val:.4f} vs reportado={rep_val:.4f} "
                f"(diferencia > {tol})"
            )

for msg in inconsistencias:
    print(" [INCONSISTENCIA]", msg)
if not inconsistencias:
    print(" OK: sin inconsistencias.")


# ---------------------------------------------------------------------------
# 5) Construir tabla final (texto formateado + version numerica)
# ---------------------------------------------------------------------------
def ci_percentil(vals):
    if len(vals) == 0:
        return (np.nan, np.nan)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (lo, hi)


filas_texto = []
filas_num = []

col_metricas = [
    ("Accuracy", "accuracy"),
    ("Precisión macro", "precision_macro"),
    ("Recall macro", "recall_macro"),
    ("F1 macro", "f1_macro"),
    ("AUC-ROC macro", "auc_macro"),
    ("QWK", "qwk"),
]

for nombre, key in MODELOS:
    punt = resultados_puntuales[key]
    fila_txt = {"Modelo": nombre, "N": N_OBS}
    fila_num = {"Modelo": nombre, "N": N_OBS}

    for label, metkey in col_metricas:
        valor = punt[metkey]
        boot_vals = resultados_boot[key][metkey]
        lo, hi = ci_percentil(boot_vals)

        fila_txt[f"{label} (IC 95%)"] = f"{valor:.3f} ({lo:.3f}–{hi:.3f})"

        fila_num[f"{label}_valor"] = round(float(valor), 6) if not np.isnan(valor) else np.nan
        fila_num[f"{label}_IC_inf"] = round(float(lo), 6) if not np.isnan(lo) else np.nan
        fila_num[f"{label}_IC_sup"] = round(float(hi), 6) if not np.isnan(hi) else np.nan
        fila_num[f"{label}_n_validos"] = resultados_boot_validos[key][metkey]

    filas_texto.append(fila_txt)
    filas_num.append(fila_num)

tabla_texto = pd.DataFrame(filas_texto)
tabla_num = pd.DataFrame(filas_num)

# ---------------------------------------------------------------------------
# 6) Regla de negrita: valor puntual mas alto por metrica (empate a 3 decimales)
# ---------------------------------------------------------------------------
celdas_negrita = []  # lista de (Modelo, Metrica)
for label, metkey in col_metricas:
    valores_redondeados = {}
    for nombre, key in MODELOS:
        v = resultados_puntuales[key][metkey]
        if not np.isnan(v):
            valores_redondeados[nombre] = round(float(v), 3)
    if not valores_redondeados:
        continue
    maximo = max(valores_redondeados.values())
    ganadores = [nombre for nombre, v in valores_redondeados.items() if v == maximo]
    for g in ganadores:
        celdas_negrita.append((g, label))

print("\nCeldas en negrita (Modelo, Métrica):")
for c in celdas_negrita:
    print("  ", c)

# ---------------------------------------------------------------------------
# 7) Guardar resultados
# ---------------------------------------------------------------------------
tabla_texto.to_csv(os.path.join(OUT_DIR, "metricas_globales_con_ic.csv"), index=False, encoding="utf-8-sig")
with pd.ExcelWriter(os.path.join(OUT_DIR, "metricas_globales_con_ic.xlsx"), engine="openpyxl") as writer:
    tabla_texto.to_excel(writer, sheet_name="tabla_formateada", index=False)
    tabla_num.to_excel(writer, sheet_name="valores_numericos", index=False)
    pd.DataFrame(celdas_negrita, columns=["Modelo", "Métrica"]).to_excel(writer, sheet_name="negrita", index=False)

print("\n" + "=" * 100)
print("TABLA FINAL")
print("=" * 100)
with pd.option_context("display.width", 220, "display.max_columns", 20):
    print(tabla_texto.to_string(index=False))

print("\nGuardado:")
print(" ", os.path.join(OUT_DIR, "metricas_globales_con_ic.csv"))
print(" ", os.path.join(OUT_DIR, "metricas_globales_con_ic.xlsx"))
