# IDVerify TFM

Detección de documentos de identidad falsificados mediante una red neuronal multi-tarea
(clasificación binaria bonafide/attack + segmentación de regiones alteradas), desarrollado
como Trabajo de Fin de Máster.

El sistema combina un encoder convolucional con un decoder U-Net, optimizado mediante
Nested Cross-Validation (10 outer folds × 5 inner folds) y búsqueda de hiperparámetros
multi-objetivo con Optuna (Pareto front sobre PR-AUC y Dice).

## Estructura de ramas

Cada rama contiene el pipeline completo con un encoder distinto, manteniendo idénticos el
decoder, las pérdidas, el protocolo de validación y la búsqueda de hiperparámetros:

| Rama | Encoder |
|---|---|
| `master` | Patel CNN (entrenado desde cero) |
| `experiment_efficientnet-b4` | EfficientNet-B4 (preentrenado en ImageNet) |
| `experiment_vit-b16` | ViT-B/16 (preentrenado en ImageNet) — próximamente |

## Datasets

Este repositorio **no incluye los datasets** por su tamaño y por las licencias de uso de
cada uno. Hay que descargarlos por separado:

### FantasyID (entrenamiento)

- Página del proyecto: https://www.idiap.ch/paper/fantasyid/
- Paper: https://arxiv.org/pdf/2507.20808
- Licencia: CC 4.0 para `train`/`test`; licencia académica no comercial (equivalente a
  HQ-WMCA) para `val`. Revisar el README del propio dataset antes de su uso. Cita
  obligatoria del paper original.

Tras descargarlo, descomprimir de forma que quede una carpeta `FantasyID/` con la
estructura `train/`, `test/`, `train.csv`, `test.csv` en la raíz del proyecto, o indicar
la ruta mediante la variable de entorno `DATASET_ROOT`.

### SIDTD (evaluación cross-dataset)

- Descarga: http://datasets.cvc.uab.es/SIDTD/

Tras descargarlo, descomprimir en una carpeta `Dataset_SIDTD/` en la raíz del proyecto, o
indicar la ruta mediante la variable de entorno `SIDTD_ROOT`.

## Instalación

```bash
pip install -r requirements.txt
```

Para GPU (CUDA), instalar antes `torch`/`torchvision` con el índice correspondiente (ver
comentarios en `requirements.txt`).

## Uso

```bash
# Pipeline completo: Nested CV + HPO + blind test (30 seeds x 4 variantes)
python main.py

# Evaluación cross-dataset sobre SIDTD (requiere modelos ya entrenados)
python evaluate_sidtd.py

# Recuperar un modelo individual a partir de los CSVs de resultados
python train_best_model.py
```

Variables de entorno relevantes (ver `config.py` para la lista completa):

- `DATASET_ROOT` — ruta a FantasyID (por defecto `./FantasyID`)
- `SIDTD_ROOT` — ruta a SIDTD (por defecto `./Dataset_SIDTD`)
- `EXPORT_DIR` — carpeta de resultados (por defecto `./exports_hpo_pareto_nested`)
