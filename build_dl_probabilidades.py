# -*- coding: utf-8 -*-
"""
Inferencia (NO entrenamiento) con los checkpoints DL ya entrenados (best.pth)
sobre las 744 imagenes fijas de validacion del Experimento 1, para obtener
y_prob (softmax) necesario para AUC-ROC macro OvR. Reutiliza EXACTAMENTE el
preprocesamiento y la arquitectura de entrenamiento_DL_efficientnet_b0_ordinal_mixup_plateau_FINAL.ipynb.
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

ROOT = r"C:\Users\Victoria\Desktop\data\datos_raw"
OUT_DIR = os.path.join(ROOT, "resultados", "cache_probas")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = 5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---- Funciones EXACTAS del notebook DL FINAL (celda 8) ----
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


# ---- build_model EXACTO del notebook DL FINAL (celda 18) ----
def build_model(arch: str, n_classes: int):
    if arch == "resnet34":
        m = models.resnet34(weights=None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    elif arch == "inception_v3":
        # Igual que el notebook original: construir con pesos preentrenados de
        # ImageNet para que transform_input=True (el checkpoint fue entrenado
        # con esa normalización interna); luego se sobreescriben todos los
        # pesos al cargar el state_dict fine-tuneado.
        m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        m.aux_logits = False
    else:
        raise ValueError(f"Arquitectura no soportada: {arch}")
    return m


class CSVImages(Dataset):
    def __init__(self, df, tfm, do_fov_crop=True, do_clahe=True):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm
        self.do_fov_crop = do_fov_crop
        self.do_clahe = do_clahe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(r["path"]).convert("RGB")
        if self.do_fov_crop:
            im = crop_fundus_rgb(im)
        if self.do_clahe:
            im = clahe_green(im)
        im = self.tfm(im)
        return im, int(r["level"])


@torch.no_grad()
def eval_collect(model, loader, device):
    model.eval()
    all_prob, all_pred, all_true = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
        all_prob.append(prob.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())
        all_true.append(y.numpy())
    y_prob = np.concatenate(all_prob, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    return y_true, y_pred, y_prob


MODELOS_DL = [
    {
        "nombre": "MobileNetV3",
        "arch": "mobilenet_v3_large",
        "img_size": 384,
        "ckpt": os.path.join(ROOT, "resultados_dl",
            "COMBINED_TRAIN_ONLY_aptos_2019-eyePACS-idrid_mobilenet_v3_large_img384_bs8_lr0.0003_no_messidor", "best.pth"),
        "pred_csv": os.path.join(ROOT, "resultados_dl",
            "COMBINED_TRAIN_ONLY_aptos_2019-eyePACS-idrid_mobilenet_v3_large_img384_bs8_lr0.0003_no_messidor",
            "figuras_tesis", "MobileNetV3_predicciones_validacion.csv"),
    },
    {
        "nombre": "EfficientNet-B0",
        "arch": "efficientnet_b0",
        "img_size": 384,
        "ckpt": os.path.join(ROOT, "resultados_dl",
            "COMBINED_TRAIN_ONLY_aptos_2019-eyePACS-idrid_efficientnet_b0_img384_bs8_lr0.0001_rerun", "best.pth"),
        "pred_csv": os.path.join(ROOT, "resultados_dl",
            "COMBINED_TRAIN_ONLY_aptos_2019-eyePACS-idrid_efficientnet_b0_img384_bs8_lr0.0001_rerun",
            "predicciones_validacion_efficientnet_b0_por_dataset.csv"),
    },
    {
        "nombre": "ResNet34",
        "arch": "resnet34",
        "img_size": 384,
        "ckpt": os.path.join(ROOT, "resultados_dl",
            "COMBINED_TRAIN_ONLY_aptos_2019-eyePACS-idrid_resnet34_img384_bs8_lr0.0003_no_messidor", "best.pth"),
        "pred_csv": os.path.join(ROOT, "resultados_dl",
            "COMBINED_TRAIN_ONLY_aptos_2019-eyePACS-idrid_resnet34_img384_bs8_lr0.0003_no_messidor",
            "figuras_tesis", "ResNet34_predicciones_validacion.csv"),
    },
    {
        "nombre": "InceptionV3",
        "arch": "inception_v3",
        "img_size": 299,
        "ckpt": os.path.join(ROOT, "resultados_dl",
            "COMBINED_TRAIN_ONLY_aptos_2019-eyePACS-idrid_inception_v3_img299_bs8_lr0.0003_no_messidor", "best.pth"),
        "pred_csv": os.path.join(ROOT, "resultados_dl",
            "COMBINED_TRAIN_ONLY_aptos_2019-eyePACS-idrid_inception_v3_img299_bs8_lr0.0003_no_messidor",
            "figuras_tesis", "InceptionV3_predicciones_validacion.csv"),
    },
]

print("DEVICE:", DEVICE)

for cfg in MODELOS_DL:
    nombre = cfg["nombre"]
    print(f"\n=== {nombre} ===")
    df_pred_saved = pd.read_csv(cfg["pred_csv"])
    assert len(df_pred_saved) == 744, f"{nombre}: N={len(df_pred_saved)}"

    tfm_va = transforms.Compose([
        transforms.Resize(cfg["img_size"], interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(cfg["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    dl_va = DataLoader(
        CSVImages(df_pred_saved, tfm_va),
        batch_size=16, shuffle=False, num_workers=0, pin_memory=True,
    )

    model = build_model(cfg["arch"], N_CLASSES).to(DEVICE)
    state = torch.load(cfg["ckpt"], map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    y_true, y_pred_new, y_prob = eval_collect(model, dl_va, DEVICE)

    y_pred_saved = df_pred_saved["y_pred"].to_numpy(dtype=int)
    y_true_saved = df_pred_saved["y_true"].to_numpy(dtype=int)

    match_pred = (y_pred_new == y_pred_saved).mean()
    match_true = (y_true == y_true_saved).all()
    print(f"  N={len(y_true)}  coincidencia y_pred recomputado vs guardado={match_pred*100:.2f}%  y_true igual={match_true}")

    np.savez_compressed(
        os.path.join(OUT_DIR, f"{nombre}_proba.npz"),
        y_true=y_true_saved,
        y_pred_saved=y_pred_saved,
        y_pred_recomputado=y_pred_new,
        y_score=y_prob,
        score_type="softmax_proba",
        classes=np.arange(N_CLASSES),
    )

    del model
    torch.cuda.empty_cache()

print("\nListo. Cache DL guardada en:", OUT_DIR)
