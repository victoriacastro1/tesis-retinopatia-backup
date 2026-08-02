# -*- coding: utf-8 -*-
"""
Inferencia (NO entrenamiento) con los modelos ML ya guardados (.pkl) sobre las
744 imagenes fijas de validacion del Experimento 1, para obtener y_prob
(necesario para AUC-ROC macro OvR), reutilizando exactamente la funcion de
extraccion de caracteristicas del notebook ML FINAL.
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import joblib

ROOT = r"C:\Users\Victoria\Desktop\data\datos_raw"
OUT_DIR = os.path.join(ROOT, "resultados", "cache_probas")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Funciones EXACTAS de entrenamiento_ML_combined_qwk_FINAL.ipynb (celda 4) ----
def crop_fundus_rgb(im_pil: Image.Image) -> Image.Image:
    arr = np.array(im_pil)
    g = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    m = (g > 5).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return im_pil
    cnt = max(cnts, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    cx, cy, r = int(cx), int(cy), int(r * 0.98)
    x0, x1 = max(0, cx - r), min(arr.shape[1], cx + r)
    y0, y1 = max(0, cy - r), min(arr.shape[0], cy + r)
    arr = arr[y0:y1, x0:x1]
    return Image.fromarray(arr)

def clahe_green(im_pil: Image.Image) -> Image.Image:
    arr = np.array(im_pil)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr[..., 1] = clahe.apply(arr[..., 1])
    return Image.fromarray(arr)

def extract_features_from_image(im_path: str) -> np.ndarray:
    try:
        im = Image.open(im_path).convert("RGB")
        im = crop_fundus_rgb(im)
        im = clahe_green(im)
        arr = np.array(im, dtype=np.float32) / 255.0

        hist_r = cv2.calcHist([arr[:, :, 0]], [0], None, [32], [0, 1])
        hist_g = cv2.calcHist([arr[:, :, 1]], [0], None, [32], [0, 1])
        hist_b = cv2.calcHist([arr[:, :, 2]], [0], None, [32], [0, 1])

        stats_r = [arr[:, :, 0].mean(), arr[:, :, 0].std(), arr[:, :, 0].min(), arr[:, :, 0].max()]
        stats_g = [arr[:, :, 1].mean(), arr[:, :, 1].std(), arr[:, :, 1].min(), arr[:, :, 1].max()]
        stats_b = [arr[:, :, 2].mean(), arr[:, :, 2].std(), arr[:, :, 2].min(), arr[:, :, 2].max()]

        gray = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edge_detection = cv2.Canny(gray, 50, 150)
        edge_ratio = edge_detection.sum() / (gray.size * 255)

        features = np.concatenate([
            hist_r.flatten(), hist_g.flatten(), hist_b.flatten(),
            stats_r, stats_g, stats_b,
            [edge_ratio],
        ])
        return features
    except Exception as e:
        print(f"Error extrayendo features de {im_path}: {e}")
        return np.zeros(109)


# ---- Cargar 744 imagenes de validacion en el orden canonico (ResNet34 CSV) ----
resnet_csv = os.path.join(ROOT, "resultados_dl",
    "COMBINED_TRAIN_ONLY_aptos_2019-eyePACS-idrid_resnet34_img384_bs8_lr0.0003_no_messidor",
    "figuras_tesis", "ResNet34_predicciones_validacion.csv")
ref = pd.read_csv(resnet_csv)
assert len(ref) == 744

ml_csv = os.path.join(ROOT, "resultados_ml", "COMBINED_TRAIN_ONLY", "predicciones_validacion_ml_por_dataset.csv")
ml_all = pd.read_csv(ml_csv)

print("Extrayendo features (109-d) para 744 imagenes de validacion...")
paths = ref["path"].tolist()
X_va = np.array([extract_features_from_image(p) for p in paths], dtype=np.float64)
y_true = ref["y_true"].to_numpy(dtype=int)
print("X_va:", X_va.shape)

np.savez_compressed(os.path.join(OUT_DIR, "X_va_744.npz"), X_va=X_va, y_true=y_true, paths=np.array(paths))

modelos = {
    "Random Forest": "RandomForest",
    "Regresión Logística": "LogisticRegression",
    "LinearSVC": "LinearSVC",
    "XGBoost": "XGBoost",
}

for nombre, key in modelos.items():
    model_path = os.path.join(ROOT, "resultados_ml", "COMBINED_TRAIN_ONLY", f"{key}_model.pkl")
    model = joblib.load(model_path)

    y_pred_new = model.predict(X_va)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_va)
        score_type = "predict_proba"
    else:
        y_score = model.decision_function(X_va)
        score_type = "decision_function"

    # y_pred/y_true guardados originalmente para este modelo
    sub = ml_all[ml_all["Modelo"] == key].reset_index(drop=True)
    y_pred_saved = sub["y_pred"].to_numpy(dtype=int)
    y_true_saved = sub["y_true"].to_numpy(dtype=int)

    match_pred = (y_pred_new == y_pred_saved).mean()
    match_true = (y_true_saved == y_true).all()

    print(f"[{nombre}] score_type={score_type} shape={y_score.shape} "
          f"coincidencia y_pred recomputado vs guardado={match_pred*100:.2f}%  y_true igual={match_true}")

    np.savez_compressed(
        os.path.join(OUT_DIR, f"{key}_proba.npz"),
        y_true=y_true_saved,
        y_pred_saved=y_pred_saved,
        y_pred_recomputado=y_pred_new,
        y_score=y_score,
        score_type=score_type,
        classes=getattr(model, "classes_", np.arange(5)),
    )

print("\nListo. Cache guardada en:", OUT_DIR)
