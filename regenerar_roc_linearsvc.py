# -*- coding: utf-8 -*-
"""
Regenera UNICAMENTE la curva ROC multiclase de LinearSVC, para que sea
coherente con el AUC-ROC macro definitivo (0.580) reportado en la tesis.

No se reentrena el modelo, no se recalculan predicciones, no se toca la
particion de datos ni ningun resultado de los demas modelos.

y_true / y_score provienen de la cache ya generada (inferencia, no
entrenamiento) sobre las 744 imagenes de la particion fija de validacion:
  resultados/cache_probas/LinearSVC_proba.npz
    - y_true       : etiquetas reales (744,), identicas a las usadas en la
                     tabla global (verificado igual a la columna 'y_true' de
                     resultados_ml/COMBINED_TRAIN_ONLY/predicciones_validacion_ml_por_dataset.csv,
                     Modelo="LinearSVC").
    - y_score      : modelo_linearsvc.decision_function(X_validacion), NO
                     predict() ni clases predichas. LinearSVC no tiene
                     predict_proba, por eso se usa decision_function (score
                     continuo real de margen por clase), tal como pide la
                     correccion.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

ROOT = r"C:\Users\Victoria\Desktop\data\datos_raw"
CACHE_PATH = os.path.join(ROOT, "resultados", "cache_probas", "LinearSVC_proba.npz")
OUT_PATH = os.path.join(ROOT, "resultados_ml", "COMBINED_TRAIN_ONLY", "LinearSVC_roc_curves.png")

N_CLASSES = 5
CLASSES = np.arange(N_CLASSES)
class_names = ["No RD", "Leve", "Moderado", "Severo", "PRD"]

# ---------------------------------------------------------------------------
# 1) Cargar y_true / y_score (decision_function) ya generados por inferencia
# ---------------------------------------------------------------------------
d = np.load(CACHE_PATH, allow_pickle=True)
y_true = d["y_true"].astype(int)
y_score = d["y_score"]              # decision_function(X_validacion), (744, 5)
score_type = str(d["score_type"])

assert score_type == "decision_function", f"Se esperaba decision_function, se encontro: {score_type}"
assert len(y_true) == 744, f"Se esperaban 744 observaciones, hay {len(y_true)}"
assert y_score.shape == (744, 5), f"y_score debe ser (744,5), es {y_score.shape}"

# Verificacion de correspondencia y_true <-> predicciones ya reportadas en la tabla global
y_pred_saved = d["y_pred_saved"].astype(int)
import pandas as pd
tabla_global = pd.read_csv(os.path.join(ROOT, "resultados_ml", "COMBINED_TRAIN_ONLY",
                                         "predicciones_validacion_ml_por_dataset.csv"))
sub = tabla_global[tabla_global["Modelo"] == "LinearSVC"].reset_index(drop=True)
assert len(sub) == 744, f"Tabla global tiene {len(sub)} filas para LinearSVC, se esperaban 744"
assert np.array_equal(sub["y_true"].to_numpy(dtype=int), y_true), \
    "El orden/valores de y_true NO coincide con la tabla global"
assert np.array_equal(sub["y_pred"].to_numpy(dtype=int), y_pred_saved), \
    "y_pred no coincide con la tabla global"
print("[OK] y_true y y_pred coinciden exactamente (mismo orden) con la tabla global de 744 imagenes.")

# ---------------------------------------------------------------------------
# 2) Binarizar (One-vs-Rest) y calcular AUC por clase + macro
# ---------------------------------------------------------------------------
y_true_bin = label_binarize(y_true, classes=CLASSES)

auc_por_clase = {}
fig, ax = plt.subplots(figsize=(7, 6), dpi=200)

for k in range(N_CLASSES):
    fpr, tpr, _ = roc_curve(y_true_bin[:, k], y_score[:, k])
    cls_auc = auc(fpr, tpr)
    auc_por_clase[k] = cls_auc
    ax.plot(fpr, tpr, lw=1.8, label=f"Clase {k} - {class_names[k]} (AUC={cls_auc:.3f})")

auc_macro = roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr")

ax.plot([0, 1], [0, 1], "--", lw=1, color="gray", label="Referencia (azar)")
ax.set_xlabel("Tasa de Falsos Positivos")
ax.set_ylabel("Tasa de Verdaderos Positivos")
ax.set_title("Curva ROC Multiclase - LinearSVC")
ax.legend(loc="lower right", fontsize=8)
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 3) Reporte y verificacion final
# ---------------------------------------------------------------------------
print("\nAUC por clase (One-vs-Rest, decision_function):")
for k in range(N_CLASSES):
    print(f"  Clase {k} ({class_names[k]}): AUC = {auc_por_clase[k]:.4f}")

print(f"\nAUC-ROC macro (no ponderado) = {auc_macro:.4f}")
diff = abs(auc_macro - 0.580)
print(f"Diferencia vs valor reportado en la tabla definitiva (0.580): {diff:.4f}")
if diff <= 0.001:
    print("[OK] Coincide con el AUC-ROC macro definitivo de la tabla (0.580).")
else:
    print("[AVISO] La diferencia supera 0.001; revisar.")

print(f"\nFigura guardada en: {OUT_PATH}")
