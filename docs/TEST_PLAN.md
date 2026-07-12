# Plan de Pruebas — Demo DocVerify

**Proyecto:** DocVerify — Detección de Documentos de Identidad Falsificados  
**TFM:** Máster Universitario en Ingeniería Informática — Universidad de Oviedo  
**Autora:** Inés Fernández Álvarez (uo29067)  
**Tutor:** Hernán Díaz Rodríguez  
**Versión:** 1.0 — Julio 2026

---

## 1. Introducción

### 1.1 Objetivo

Este documento describe la estrategia, los tipos de prueba, los casos de prueba y los criterios de aceptación aplicados al backend de la demo DocVerify. Su finalidad es verificar que el sistema cumple los requisitos funcionales y no funcionales definidos en `REQUIREMENTS.md` y garantizar la calidad del código antes del despliegue.

### 1.2 Alcance

| Incluido | Excluido |
|---|---|
| Backend FastAPI (`demo_app/app.py`) | Modelos de ML (entrenamiento y métricas — cubiertos en el TFM) |
| Funciones de utilidad (preprocesado, heatmap, parseo de regiones) | Frontend React (pruebas manuales en navegador) |
| Endpoints REST (`/analyze`, `/dnis`, `/dnis/{id}/meta`) | Infraestructura de despliegue (HF Spaces, Netlify) |
| Validación de entradas y manejo de errores | Pruebas de carga o estrés |

### 1.3 Herramientas

| Herramienta | Versión | Uso |
|---|---|---|
| pytest | 9.1.1 | Framework de pruebas |
| httpx2 | 2.5.0 | Cliente HTTP para TestClient de FastAPI |
| FastAPI TestClient | — | Pruebas de integración de endpoints |
| unittest.mock | stdlib | Mock de modelos PyTorch |
| PIL / numpy | — | Generación de imágenes sintéticas |

### 1.4 Ejecución

```bash
cd demo/demo_app
python -m pytest          # todos los tests
python -m pytest -v       # salida detallada
python -m pytest tests/test_utils.py   # solo unitarios
python -m pytest tests/test_api.py     # solo integración
```

---

## 2. Estrategia de pruebas

### 2.1 Niveles de prueba

#### Pruebas unitarias (`tests/test_utils.py`)

Verifican funciones individuales de forma aislada, sin red ni modelos reales. Cada función se prueba con entradas válidas, valores límite y entradas inválidas.

**Funciones cubiertas:**

| Función | Descripción |
|---|---|
| `_preprocess` | Decodificación de imagen y conversión a tensor PyTorch |
| `_normalize01_robust` | Normalización por percentiles |
| `_simple_colormap` | Mapeado de valores escalares a color RGB |
| `_heatmap_and_overlay` | Generación de heatmap y overlay a partir de logits de máscara |
| `_norm_provenance` | Normalización de etiquetas de procedencia de región |
| `_parse_region_generic` | Extracción de coordenadas y atributos de una región JSON |
| `_iter_regions_any_schema` | Soporte de múltiples esquemas JSON de anotación |
| `_safe_doc_id` | Validación de identificadores de documento |

#### Pruebas de integración (`tests/test_api.py`)

Verifican el comportamiento completo de los endpoints HTTP usando `TestClient`. Los modelos ML se sustituyen por un `MockDocVerifyModel` que devuelve tensores de ceros con las formas correctas, lo que permite ejecutar las pruebas sin cargar los pesos reales (89 MB).

**Endpoints cubiertos:**

| Endpoint | Método | Pruebas |
|---|---|---|
| `/dnis` | GET | Respuesta, formato, contenido |
| `/dnis/{id}/meta` | GET | Documento existente, no existente, IDs inválidos |
| `/analyze` | POST | Encoders válidos, errores de validación, formato de respuesta |

### 2.2 Estrategia de mock

Los modelos PyTorch reales (`patel_outer9.pt`, `efficientnet_outer7.pt`) no se cargan en los tests para:
- Evitar dependencias de ficheros binarios de 89 MB
- Reducir el tiempo de ejecución de la suite (< 3 s frente a > 30 s con modelos reales)
- Permitir verificar la lógica de la API independientemente de los pesos

El `MockDocVerifyModel` implementa `forward(x) → {"cls": zeros(B,1), "mask": zeros(B,1,H,W)}`, produciendo `sigmoid(0) = 0.5` como probabilidad de falsificación — valor determinista que permite assertions exactas.

La inyección se realiza en `conftest.py` mediante `unittest.mock.patch` sobre la función `load_models`, que es invocada por el gestor de ciclo de vida (`lifespan`) de FastAPI al arrancar el `TestClient`.

---

## 3. Casos de prueba

### 3.1 Pruebas unitarias — `_preprocess`

| ID | Descripción | Entrada | Resultado esperado |
|---|---|---|---|
| U-PRE-01 | Devuelve tensor y array numpy | Bytes JPEG válidos | `(torch.Tensor, np.ndarray)` |
| U-PRE-02 | Forma del tensor correcta | Bytes JPEG 120×80 | `tensor.shape == (1, 3, 224, 224)` |
| U-PRE-03 | Valores del tensor en [0, 1] | Bytes JPEG | `0.0 ≤ min`, `max ≤ 1.0` |
| U-PRE-04 | Forma del array numpy correcta | Bytes JPEG | `arr.shape == (224, 224, 3)` |
| U-PRE-05 | Valores del array en [0, 1] | Bytes JPEG | `0.0 ≤ min`, `max ≤ 1.0` |
| U-PRE-06 | Acepta PNG además de JPEG | Bytes PNG 64×64 | `tensor.shape == (1, 3, 224, 224)` |
| U-PRE-07 | Tipo de dato float32 | Bytes JPEG | `tensor.dtype == torch.float32` |

### 3.2 Pruebas unitarias — `_normalize01_robust`

| ID | Descripción | Entrada | Resultado esperado |
|---|---|---|---|
| U-NRM-01 | Salida en rango [0, 1] | Array aleatorio 50×50 | `min ≥ 0.0`, `max ≤ 1.0` |
| U-NRM-02 | Entrada uniforme devuelve ceros | Array constante 10×10 | `max < 1e-3` |
| U-NRM-03 | Preserva la forma | Array 30×40 | `shape == (30, 40)` |
| U-NRM-04 | Salida en float32 | Array float64 | `dtype == np.float32` |

### 3.3 Pruebas unitarias — `_simple_colormap`

| ID | Descripción | Entrada | Resultado esperado |
|---|---|---|---|
| U-CM-01 | Forma de salida correcta | Array 10×10 [0,1] | `shape == (10, 10, 3)` |
| U-CM-02 | Valores en [0, 1] | Array 10×10 [0,1] | `min ≥ 0.0`, `max ≤ 1.0` |
| U-CM-03 | Valor 0.0 produce azul | Array [[0.0]] | `R=0.0`, `B=1.0` |
| U-CM-04 | Valor 1.0 produce rojo | Array [[1.0]] | `R=1.0`, `B=0.0` |
| U-CM-05 | Recorta valores fuera de rango | Array [[-0.5, 1.5]] | `min ≥ 0.0`, `max ≤ 1.0` |

### 3.4 Pruebas unitarias — `_heatmap_and_overlay`

| ID | Descripción | Entrada | Resultado esperado |
|---|---|---|---|
| U-HM-01 | Devuelve dict con claves requeridas | img 224×224, mask 1×1×224×224 | Claves `heatmap_png_base64`, `overlay_png_base64` |
| U-HM-02 | Strings base64 no vacíos | ídem | `len > 0` para ambas claves |
| U-HM-03 | Base64 decodifica a PNG válido | ídem | `Image.open(...)` sin error, `format == "PNG"` |
| U-HM-04 | Máscara no cuadrada se redimensiona | img 224×224, mask 1×1×14×14 | Sin excepción, base64 no vacío |

### 3.5 Pruebas unitarias — `_norm_provenance`

| ID | Descripción | Entrada | Resultado esperado |
|---|---|---|---|
| U-PRV-01 | None → original | `None` | `"original"` |
| U-PRV-02 | Cadena vacía → original | `""` | `"original"` |
| U-PRV-03 | Variantes de original | `"bonafide"`, `"genuine"`, `"real"`, `"0"` | `"original"` |
| U-PRV-04 | Variantes de alterado | `"attack"`, `"fake"`, `"tampered"`, `"1"`, `"true"` | `"altered"` |
| U-PRV-05 | Insensible a mayúsculas | `"Original"`, `"Altered"` | `"original"`, `"altered"` |
| U-PRV-06 | Valor desconocido → None | `"unknown_xyz"` | `None` |

### 3.6 Pruebas unitarias — `_parse_region_generic`

| ID | Descripción | Entrada | Resultado esperado |
|---|---|---|---|
| U-REG-01 | Región original válida | x=10,y=20,w=50,h=30,prov=original | Dict con campos correctos |
| U-REG-02 | Región alterada válida | prov=altered | `region_provenance == "altered"` |
| U-REG-03 | Ancho cero → None | w=0 | `None` |
| U-REG-04 | Alto negativo → None | h=-5 | `None` |
| U-REG-05 | Procedencia desconocida → None | prov=unknown | `None` |
| U-REG-06 | Entrada no dict → None | `"cadena"` | `None` |
| U-REG-07 | Sin coordenadas → None | Solo region_attributes | `None` |
| U-REG-08 | Formato de ID correcto | idx=3 | `id == "r3"` |
| U-REG-09 | Coordenadas float se redondean | x=10.7, w=50.5 | `x==11`, `w==round(50.5)` (redondeo bancario) |

### 3.7 Pruebas unitarias — `_iter_regions_any_schema`

| ID | Descripción | Entrada | Resultado esperado |
|---|---|---|---|
| U-SCH-01 | Formato lista directa | `[{...}]` | Lista original |
| U-SCH-02 | Formato `{regions: [...]}` | `{"regions": [...]}` | Lista de regiones |
| U-SCH-03 | Formato VIA multi-imagen | `{"_via_img_metadata": {...}}` | Lista de regiones del doc correcto |
| U-SCH-04 | Fallback `annotations` | `{"annotations": [...]}` | Lista de anotaciones |
| U-SCH-05 | Dict vacío → lista vacía | `{}` | `[]` |
| U-SCH-06 | Tipo no soportado → lista vacía | `"cadena"` | `[]` |
| U-SCH-07 | VIA con una sola entrada | `{"_via_img_metadata": {"k": {...}}}` | Lista de regiones de esa entrada |

### 3.8 Pruebas unitarias — `_safe_doc_id`

| ID | Descripción | Entrada | Resultado esperado |
|---|---|---|---|
| U-ID-01 | IDs válidos aceptados | `"french-097_03"`, `"doc123"`, `"A.B-C_D"` | Devuelve el mismo ID |
| U-ID-02 | Path traversal rechazado | `"../etc/passwd"` | HTTP 400 |
| U-ID-03 | Barra inclinada rechazada | `"doc/id"` | HTTP 400 |
| U-ID-04 | Espacio rechazado | `"doc id"` | HTTP 400 |
| U-ID-05 | Cadena vacía rechazada | `""` | HTTP 400 |
| U-ID-06 | None rechazado | `None` | HTTP 400 |
| U-ID-07 | Punto y coma rechazado | `"doc;rm"` | HTTP 400 |

---

### 3.9 Pruebas de integración — `GET /dnis`

| ID | Descripción | Resultado esperado |
|---|---|---|
| I-DNI-01 | Respuesta exitosa | HTTP 200 |
| I-DNI-02 | Cuerpo es lista JSON | `isinstance(body, list)` |
| I-DNI-03 | Contiene documento de prueba | `"test_doc"` en lista de IDs |
| I-DNI-04 | Cada elemento tiene campos requeridos | Campos `id` y `meta_url` presentes |
| I-DNI-05 | Formato de `meta_url` correcto | `"/dnis/test_doc/meta"` |

### 3.10 Pruebas de integración — `GET /dnis/{doc_id}/meta`

| ID | Descripción | Resultado esperado |
|---|---|---|
| I-META-01 | Documento existente → 200 | HTTP 200 |
| I-META-02 | Respuesta tiene campos requeridos | Campos `id` y `regions` presentes |
| I-META-03 | Regiones con procedencia correcta | `"original"` y `"altered"` presentes |
| I-META-04 | Cada región tiene coordenadas | Campos `x`, `y`, `w`, `h` en cada región |
| I-META-05 | Documento inexistente → 404 | HTTP 404 |
| I-META-06 | Path traversal → 400 | HTTP 400 o 404 |
| I-META-07 | Barra codificada en ID → rechazado | HTTP 400 o 404 |
| I-META-08 | Espacio en ID → rechazado | HTTP 400 o 404 |

### 3.11 Pruebas de integración — `POST /analyze`

| ID | Descripción | Resultado esperado |
|---|---|---|
| I-ANA-01 | Encoder Patel + JPEG → 200 | HTTP 200 |
| I-ANA-02 | Encoder EfficientNet-B4 + JPEG → 200 | HTTP 200 |
| I-ANA-03 | Respuesta contiene campos requeridos | `probability_fake`, `heatmap_png_base64`, `overlay_png_base64` |
| I-ANA-04 | Probabilidad en rango [0, 1] | `0.0 ≤ probability_fake ≤ 1.0` |
| I-ANA-05 | Modelo mock devuelve 0.5 | `probability_fake ≈ 0.5` (sigmoid(0)) |
| I-ANA-06 | Acepta PNG además de JPEG | HTTP 200 con `image/png` |
| I-ANA-07 | Heatmap base64 decodificable | `base64.b64decode(...)` sin error, longitud > 0 |
| I-ANA-08 | Sin fichero → 400 | HTTP 400 |
| I-ANA-09 | Tipo MIME no soportado → 415 | HTTP 415 |
| I-ANA-10 | Fichero vacío → 400 | HTTP 400 |
| I-ANA-11 | Encoder desconocido → 422 | HTTP 422 |
| I-ANA-12 | Sin parámetro encoder → 422 | HTTP 422 |

---

## 4. Resultados de ejecución

Suite ejecutada el **4 de julio de 2026** con Python 3.11.0, pytest 9.1.1.

```
============================= test session starts =============================
platform win32 -- Python 3.11.0, pytest-9.1.1
testpaths: tests
collected 87 items

tests/test_api.py   ......................          25 passed
tests/test_utils.py .......................................
                    ...............................  62 passed

========================= 87 passed in 2.88s ==========================
```

| Categoría | Tests | Resultado |
|---|---|---|
| Unitarios | 62 | ✅ 62 pasados |
| Integración | 25 | ✅ 25 pasados |
| **Total** | **87** | **✅ 87 pasados** |

Tiempo de ejecución: **2,88 s** (cumple RNF-05: < 30 s).

---

## 5. Criterios de aceptación globales

| Criterio | Umbral | Estado |
|---|---|---|
| Todos los tests pasan | 100 % | ✅ 87/87 |
| Tiempo de ejecución de la suite | < 30 s | ✅ 2,88 s |
| Tests sin dependencia de pesos reales | Modelos mockeados | ✅ |
| Validación de entradas cubierta | Todos los casos de error de RF | ✅ |
| Path traversal bloqueado | HTTP 400 | ✅ |

---

## 6. Pruebas manuales pendientes

Las siguientes pruebas requieren ejecución en navegador y no están automatizadas:

| ID | Descripción | Cómo verificar |
|---|---|---|
| M-01 | Galería muestra 10 miniaturas correctamente | Abrir `http://localhost:5173`, contar miniaturas |
| M-02 | Clic en DNI lanza análisis automático | Seleccionar cualquier DNI, verificar que aparece overlay sin botón |
| M-03 | Cambio de encoder relanza análisis | Seleccionar DNI, cambiar encoder, verificar nuevo resultado |
| M-04 | Toggle de vistas funciona | Con resultado visible, pulsar cada modo y verificar imagen |
| M-05 | Probabilidad y barra se muestran correctamente | Verificar porcentaje, color de barra y veredicto |
| M-06 | Responsive en móvil | Reducir ventana a 360 px, verificar sin scroll horizontal |
| M-07 | Spinner visible durante análisis | Seleccionar DNI y observar indicador de carga |
| M-08 | Mensaje de error ante backend caído | Detener backend, seleccionar DNI, verificar mensaje de error |

---

*Fin del plan de pruebas.*
