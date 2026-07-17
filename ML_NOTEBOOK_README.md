# Machine Learning Notebook - Guía de Uso

## 📋 Descripción General

Este notebook implementa un **pipeline de Machine Learning completo** para clasificación de Retinopatía Diabética:

- ✅ **Combinación de 4 datasets**: aptos_2019, eyePACS, idrid, messidor
- ✅ **5 modelos de ML**: RandomForest, LogisticRegression, LinearSVC, XGBoost, SVM-RBF (pendiente)
- ✅ **Métricas DL**: F1 Score, IoU/Jaccard, ROC/AUC (idénticas a Deep Learning)
- ✅ **W&B Integration**: Nuevo proyecto `retinopatia-diabetica-ml` con logging completo
- ✅ **Artifacts**: Modelos, reportes y gráficos subidos a W&B

## 🚀 Flujo de Ejecución

### 1. **Importar librerías y configuración** (Celda 2)
   - Configura rutas, seeds, directorios de W&B
   - Define constantes (N_CLASSES=5, SEED=42, VAL_SIZE=15%)

### 2. **Funciones de preprocesamiento** (Celda 3)
   - `crop_fundus_rgb()`: Recorta región circular del fondo de ojo
   - `clahe_green()`: Mejora de contraste local (CLAHE) en canal verde
   - `extract_features_from_image()`: Extrae características manualmente
     - Histogramas RGB (32 bins × 3 = 96 features)
     - Estadísticas por canal (media, std, min, max = 12 features)
     - Detección de bordes (edge ratio = 1 feature)
     - **Total: ~109 características por imagen**

### 3. **Cargar y combinar datasets** (Celda 4)
   - Lee train.csv de cada dataset
   - Detecta automáticamente columnas de imagen y etiqueta
   - Resuelve rutas de imágenes
   - Aplica balanceo opcional (BALANCE_MAX_PER_CLASS=1000)

### 4. **Extracción de características** (Celda 5)
   - Procesa todas las imágenes
   - Extrae 109 características por imagen
   - Divide en train (85%) y validación (15%) con StratifiedShuffleSplit

### 5. **Entrenar modelos** (Celda 6)
   - **RandomForest**: 200 árboles, max_depth=15, class_weight=balanced
   - **LogisticRegression**: max_iter=1000, class_weight=balanced
   - **LinearSVC**: max_iter=2000, class_weight=balanced, dual=False
   - **XGBoost**: 200 estimadores, max_depth=7, learning_rate=0.05
   - Guarda cada modelo como `.pkl` en OUTDIR

### 6. **Evaluar y calcular métricas** (Celda 7)
   - Para cada modelo calcula:
     - **Accuracy**: (y_pred == y_va).mean()
     - **F1-macro/micro**: f1_score()
     - **IoU-macro**: Jaccard index por clase
     - **AUC-macro/micro**: roc_auc_score() con OVR
     - **Confusion Matrix**: classification_report()
   - Genera gráficos:
     - F1 por clase
     - IoU por clase
     - ROC curves (OVR)
     - Matriz de confusión

### 7. **Logging en W&B** (Celda 8)
   - Crea **run separate por modelo** en proyecto `retinopatia-diabetica-ml`
   - Loguea:
     - Métricas (Accuracy, F1, IoU, AUC)
     - Tablas (Classification Report, IoU by class)
     - Gráficos (F1 plot, IoU plot, ROC plot, Confusion Matrix)
     - Artifacts (modelo .pkl, CSV de reportes)

### 8. **Guardar configuración y resumen** (Celda 9)
   - Guarda `config.json` con todos los parámetros
   - Crea `models_comparison_summary.csv` con métricas de todos los modelos
   - Muestra resumen final con estadísticas

## 📊 Archivos de Salida

```
C:\Users\Victoria\Desktop\data\datos_raw\resultados_ml\COMBINED_TRAIN_ONLY\
├── RandomForest_model.pkl                    # Modelo entrenado
├── RandomForest_classification_report.csv    # Reportes de métricas
├── RandomForest_iou_per_class.csv
├── RandomForest_f1_per_class.png            # Gráficos
├── RandomForest_iou_per_class.png
├── RandomForest_roc_curves.png
├── RandomForest_confusion_matrix.png
├── LogisticRegression_model.pkl
├── LogisticRegression_*                     # (idem para otros modelos)
├── LinearSVC_model.pkl
├── LinearSVC_*
├── XGBoost_model.pkl
├── XGBoost_*
├── models_comparison_summary.csv            # Resumen comparativo
├── config.json                              # Configuración del experimento
└── _wandb_runs/                             # Logs locales de W&B
```

## 📈 Métricas Esperadas

### Comparación Típica (Valores Aproximados)
| Modelo | Accuracy | F1-Macro | IoU-Macro | AUC-Macro |
|--------|----------|----------|-----------|-----------|
| RandomForest | ~0.65 | ~0.55 | ~0.40 | ~0.82 |
| LogisticRegression | ~0.60 | ~0.48 | ~0.35 | ~0.80 |
| LinearSVC | ~0.62 | ~0.50 | ~0.37 | ~0.81 |
| XGBoost | ~0.64 | ~0.54 | ~0.39 | ~0.83 |

*Valores varían según balanceo de clases y características extraídas*

## 🔄 Modificaciones Posibles

### Para SVM-RBF Completo
```python
# Agregar en models_config (Celda 6):
"SVM_RBF": {
    "model": SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        probability=True  # Para predict_proba
    ),
    "name": "SVM_RBF"
}
```

### Para más características
Editar `extract_features_from_image()` para agregar:
- SIFT/SURF keypoints
- HOG (Histogram of Oriented Gradients)
- LBP (Local Binary Patterns)
- GLCM (Gray Level Co-occurrence Matrix)

### Para grid search de hiperparámetros
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20]
}
grid = GridSearchCV(RandomForestClassifier(...), param_grid)
grid.fit(X_tr, y_tr)
```

## 🌐 Acceso en W&B

**Nuevo Proyecto**: `retinopatia-diabetica-ml`
- URL: https://wandb.ai/victoria-castro-universidad-peruana-cayetano-heredia/retinopatia-diabetica-ml
- Runs por modelo (RandomForest_combined, LogisticRegression_combined, etc.)
- Artifacts: Modelos, reportes, gráficos
- Tablas: Classification Report, IoU by class

## ⏱️ Tiempo Estimado de Ejecución

| Paso | Tiempo Aproximado |
|------|------------------|
| Cargar datasets | 10-15s |
| Extraer características (4 datasets) | 15-20 minutos |
| Entrenar RF (200 árboles) | 30-60s |
| Entrenar LogReg | 5-10s |
| Entrenar LinearSVC | 10-20s |
| Entrenar XGBoost | 20-40s |
| Evaluar + Métricas | 10-15s |
| Logging W&B | 30-60s |
| **TOTAL** | **~17-23 minutos** |

## ✅ Checklist Antes de Ejecutar

- [ ] Librerías instaladas: `xgboost`, `scikit-learn`, `wandb`, `joblib`
- [ ] Datasets presentes en `C:\Users\Victoria\Desktop\data\datos_raw\{aptos_2019,eyePACS,idrid,messidor}`
- [ ] W&B API key configurada (wandb.login())
- [ ] Espacio en disco para modelos + gráficos (~500MB)
- [ ] Conexión a internet (para W&B logging)

## 🐛 Troubleshooting

**Error: "No train.csv"**
- Verificar que cada dataset tenga `train.csv` en su carpeta

**Error: "No columns detectadas"**
- Verificar nombres de columnas (image, level) en CSVs

**Error en extraction_features: "shapes inconsistent"**
- Asegurarse que todas las imágenes existan y sean válidas

**W&B sync lento**
- Reduce número de logs o agrupa métricas

## 📝 Próximos Pasos (Recomendado)

1. **Agregar SVM-RBF** cuando el notebook esté estable
2. **Optimizar características** con feature selection (SelectKBest)
3. **Cross-validation** en lugar de solo train/val split
4. **Comparar con modelos DL** (DL vs ML performance)
5. **Fine-tuning hiperparámetros** con GridSearchCV

---

**Creado**: 17 de Noviembre de 2025
**Proyecto**: Retinopatía Diabética - Machine Learning Pipeline
**Autor**: Victoria Castro
