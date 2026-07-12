# IDVerify TFM

Detección de documentos de identidad falsificados generados con IA mediante una red
neuronal multi-tarea (clasificación binaria bonafide/attack + segmentación de regiones
alteradas), desarrollado como Trabajo de Fin de Máster.

El sistema combina un encoder intercambiable (CNN propia tipo Patel, o EfficientNet-B4 /
ViT-B/16 preentrenados con ImageNet-1K, según la rama) con un decoder U-Net común,
optimizado mediante Nested
Cross-Validation (10 outer folds × 5 inner folds) y búsqueda de hiperparámetros
multi-objetivo con Optuna (Pareto front sobre PR-AUC y Dice).

## Estructura de ramas

Cada rama contiene el pipeline completo con un encoder distinto, manteniendo idénticos el
decoder, las pérdidas, el protocolo de validación y la búsqueda de hiperparámetros:

| Rama | Encoder |
|---|---|
| `master` | Patel CNN (entrenado desde cero) |
| `experiment_efficientnet-b4` | EfficientNet-B4 (preentrenado en ImageNet-1K) |
| `experiment_vit-b16` | ViT-B/16 (preentrenado en ImageNet-1K) |

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

- Descarga (subconjunto de templates, el único usado en este proyecto):
  http://datasets.cvc.uab.es/SIDTD/templates.zip
  (el repositorio también ofrece `clips.zip`, `clips_cropped.zip` y `videos.zip`, no usados aquí)
- Código de generación (referencia): https://github.com/Oriolrt/SIDTD_Dataset
- Licencia: CC-BY-4.0. Cita obligatoria del paper original.

Tras descargarlo, descomprimir en una carpeta `Dataset_SIDTD/` en la raíz del proyecto, o
indicar la ruta mediante la variable de entorno `SIDTD_ROOT`.

## Instalación

```bash
pip install -r requirements.txt
```

Para GPU (CUDA), instalar antes `torch`/`torchvision` con el índice correspondiente (ver
comentarios en `requirements.txt`).

## Uso

### Ejecución local (verificación rápida)

```bash
# Smoke test con parámetros mínimos (pocos folds/epochs, subconjunto de datos):
# comprueba que el pipeline completo funciona antes de lanzarlo en el HPC
python main_local.py
```

`main_local.py` fija `EXPORT_DIR=./exports_local_test` y valores reducidos de
`N_OUTER`/`N_INNER`/`N_TRIALS`/épocas, pensado solo para verificación, no para obtener
resultados válidos.

### Ejecución completa (pensada para el HPC)

```bash
# Pipeline completo: Nested CV + HPO + blind test (30 seeds x 4 variantes)
python main.py

# Evaluación cross-dataset sobre SIDTD (requiere modelos ya entrenados)
python evaluate_sidtd.py

# Recuperar un modelo individual a partir de los CSVs de resultados
python train_best_model.py
```

### Lanzamiento en el HPC (SLURM)

Cada rama incluye su propio script de entrenamiento y de evaluación cross-dataset, cada uno
con su `EXPORT_DIR` fijado explícitamente para no mezclar resultados entre codificadores:

| Rama | Entrenamiento | Evaluación cross-dataset |
|---|---|---|
| `master` | `sbatch run_pareto.slurm` | `sbatch evaluate_sidtd_pareto.slurm` |
| `experiment_efficientnet-b4` | `sbatch run_effnet.slurm` | `sbatch evaluate_sidtd_effnet.slurm` |
| `experiment_vit-b16` | `sbatch run_vit.slurm` | `sbatch evaluate_sidtd_vit.slurm` |

Todos los jobs usan `--requeue` y reanudación automática (checkpointing) para tolerar
interrupciones por prioridad en el clúster.

Variables de entorno relevantes (ver `config.py` para la lista completa):

- `DATASET_ROOT` — ruta a FantasyID (por defecto `./FantasyID`)
- `SIDTD_ROOT` — ruta a SIDTD (por defecto `./Dataset_SIDTD`)
- `EXPORT_DIR` — carpeta de resultados (por defecto `./exports_<fecha>_<hora>` si no se
  fija explícitamente; cada script de SLURM la fija a una carpeta dedicada por codificador)

## Demo interactiva

La rama `demo` contiene una aplicación web (FastAPI + React) para probar el sistema de
forma interactiva con los tres codificadores entrenados. Desplegada en
[https://huggingface.co/spaces/INESFA/idverify-demo](https://huggingface.co/spaces/INESFA/idverify-demo).

## Nota sobre el historial de commits

Los commits de este repositorio pueden aparecer firmados con dos identidades de git
distintas (`uo29067` y `mfernandez345`), ambas cuentas personales de la autora, sin
relación con otros colaboradores.
