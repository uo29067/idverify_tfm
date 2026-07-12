# Especificación de la API — Demo DocVerify

**Proyecto:** DocVerify — Detección de Documentos de Identidad Falsificados  
**TFM:** Máster Universitario en Ingeniería Informática — Universidad de Oviedo  
**Autora:** Inés Fernández Álvarez (uo29067)  
**Tutor:** Hernán Díaz Rodríguez  
**Versión:** 1.0 — Julio 2026

---

## 1. Información general

| Campo | Valor |
|---|---|
| **Base URL (desarrollo)** | `http://localhost:7860` |
| **Base URL (producción)** | `https://<usuario>-docverify.hf.space` |
| **Protocolo** | HTTP/1.1, HTTPS en producción |
| **Formato de datos** | JSON (respuestas), `multipart/form-data` (petición `/analyze`) |
| **Autenticación** | Opcional — cabecera `X-API-Key` (si `API_KEY` está configurado en el servidor) |
| **CORS** | Orígenes permitidos configurables via `CORS_ORIGINS` |

---

## 2. Endpoints

### 2.1 `POST /analyze`

Analiza una imagen de documento de identidad y devuelve la probabilidad de falsificación junto con la máscara de segmentación.

#### Petición

**Content-Type:** `multipart/form-data`

| Campo | Tipo | Obligatorio | Descripción |
|---|---|---|---|
| `encoder` | `string` | Sí | Encoder del modelo a usar. Valores: `"patel"`, `"efficientnet_b4"` |
| `image` | `file` | Sí* | Imagen del documento (JPEG o PNG, máx. 10 MB) |
| `file` | `file` | Sí* | Alternativa a `image` (mismo comportamiento) |

\* Se requiere exactamente uno de los dos campos de fichero.

**Cabeceras opcionales:**

| Cabecera | Descripción |
|---|---|
| `X-API-Key` | Clave de API (solo si el servidor tiene `API_KEY` configurado) |

#### Respuesta exitosa — `200 OK`

```json
{
  "probability_fake": 0.9974,
  "heatmap": "<base64>",
  "heatmap_png_base64": "<base64>",
  "overlay_png_base64": "<base64>"
}
```

| Campo | Tipo | Descripción |
|---|---|---|
| `probability_fake` | `float` [0, 1] | Probabilidad de que el documento sea falsificado. `sigmoid` aplicado al logit de clasificación. |
| `heatmap` | `string` | PNG codificado en base64. Mapa de calor puro (azul = auténtico, rojo = alterado). Alias de `heatmap_png_base64`. |
| `heatmap_png_base64` | `string` | Ídem. |
| `overlay_png_base64` | `string` | PNG codificado en base64. Imagen original con el mapa de calor superpuesto (α = 0,45). |

> **Nota sobre la máscara:** los valores proceden de `sigmoid(mask_logits)`, normalizados por percentiles 1–99 para maximizar el contraste visual. No representan probabilidades píxel a píxel calibradas.

#### Respuestas de error

| Código | Condición |
|---|---|
| `400 Bad Request` | No se envió fichero (`image` ni `file`), o el fichero está vacío (0 bytes) |
| `401 Unauthorized` | `X-API-Key` inválida (solo si `API_KEY` está configurado) |
| `413 Request Entity Too Large` | Fichero supera el límite de 10 MB |
| `415 Unsupported Media Type` | El `Content-Type` del fichero no es `image/jpeg` ni `image/png` |
| `422 Unprocessable Entity` | Parámetro `encoder` ausente o con valor no permitido |
| `500 Internal Server Error` | Error durante la inferencia |

#### Ejemplo de llamada (curl)

```bash
curl -X POST http://localhost:7860/analyze \
  -F "encoder=patel" \
  -F "image=@french-097_03.jpg"
```

#### Ejemplo de llamada (JavaScript / fetch)

```js
const fd = new FormData()
fd.append('encoder', 'efficientnet_b4')
fd.append('image', imageFile)

const res  = await fetch('http://localhost:7860/analyze', { method: 'POST', body: fd })
const data = await res.json()
// data.probability_fake → 0.9974
// data.overlay_png_base64 → "iVBORw0KGgo..."
```

---

### 2.2 `GET /dnis`

Devuelve la lista de documentos de demo disponibles.

#### Petición

Sin parámetros.

#### Respuesta exitosa — `200 OK`

```json
[
  {
    "id": "french-097_03",
    "image_url": "/DNIs/french-097_03.jpg",
    "meta_url": "/dnis/french-097_03/meta"
  },
  {
    "id": "usa-016_03",
    "image_url": "/DNIs/usa-016_03.jpg",
    "meta_url": "/dnis/usa-016_03/meta"
  }
]
```

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | `string` | Identificador único del documento |
| `image_url` | `string` \| `null` | Ruta relativa a la imagen (montar sobre la base URL). `null` si no existe imagen. |
| `meta_url` | `string` | Ruta relativa al endpoint de metadatos |

#### Respuestas de error

Esta operación no produce errores: si el directorio de demos no existe, devuelve lista vacía `[]`.

---

### 2.3 `GET /dnis/{doc_id}/meta`

Devuelve los metadatos de regiones anotadas de un documento de demo.

#### Parámetros de ruta

| Parámetro | Tipo | Descripción |
|---|---|---|
| `doc_id` | `string` | Identificador del documento. Solo caracteres `[A-Za-z0-9._-]`. |

#### Respuesta exitosa — `200 OK`

```json
{
  "id": "french-097_03",
  "image_url": "/DNIs/french-097_03.jpg",
  "image_width": 856,
  "image_height": 540,
  "regions": [
    {
      "id": "r0",
      "x": 42,
      "y": 108,
      "w": 312,
      "h": 48,
      "field_name": "nombre",
      "val": null,
      "language": null,
      "region_provenance": "original"
    },
    {
      "id": "r1",
      "x": 420,
      "y": 210,
      "w": 180,
      "h": 36,
      "field_name": "fecha",
      "val": null,
      "language": null,
      "region_provenance": "altered"
    }
  ]
}
```

| Campo raíz | Tipo | Descripción |
|---|---|---|
| `id` | `string` | Identificador del documento |
| `image_url` | `string` \| `null` | Ruta relativa a la imagen |
| `image_width` | `integer` \| `null` | Ancho original de la imagen en píxeles |
| `image_height` | `integer` \| `null` | Alto original de la imagen en píxeles |
| `regions` | `array` | Lista de regiones anotadas (ver tabla inferior) |

**Campos de cada región:**

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | `string` | Identificador de región (`"r0"`, `"r1"`, …) |
| `x` | `integer` | Coordenada X de la esquina superior izquierda (píxeles) |
| `y` | `integer` | Coordenada Y de la esquina superior izquierda (píxeles) |
| `w` | `integer` | Ancho del bounding box (píxeles) |
| `h` | `integer` | Alto del bounding box (píxeles) |
| `field_name` | `string` \| `null` | Nombre del campo del documento (`"nombre"`, `"fecha"`, `"face"`, …) |
| `val` | `string` \| `null` | Valor textual del campo, si está anotado |
| `language` | `string` \| `null` | Idioma del campo, si está anotado |
| `region_provenance` | `"original"` \| `"altered"` | Procedencia de la región |

> **Regla especial para caras:** si el documento tiene múltiples regiones con `field_name == "face"`, el endpoint devuelve únicamente la cara más grande por procedencia (original y/o alterada), descartando el resto.

> **Compatibilidad de esquemas JSON:** el endpoint soporta múltiples formatos de anotación: `{regions: [...]}`, formato VIA multi-imagen (`_via_img_metadata`), `{annotations: [...]}` y lista directa.

#### Respuestas de error

| Código | Condición |
|---|---|
| `400 Bad Request` | `doc_id` contiene caracteres no permitidos (path traversal, espacios, etc.) |
| `404 Not Found` | No existe el fichero JSON para ese `doc_id` |

---

### 2.4 `GET /DNIs/{filename}`

Sirve los ficheros estáticos (imágenes de documentos de demo).

#### Parámetros de ruta

| Parámetro | Tipo | Descripción |
|---|---|---|
| `filename` | `string` | Nombre del fichero con extensión (p. ej. `french-097_03.jpg`) |

#### Respuesta exitosa — `200 OK`

Binario de la imagen con `Content-Type: image/jpeg` o `image/png` según la extensión.

#### Respuestas de error

| Código | Condición |
|---|---|
| `404 Not Found` | Fichero no encontrado |

---

## 3. Modelos de datos

### 3.1 Encoders disponibles

| Valor | Arquitectura | Parámetros | Pliegue NCV | PR-AUC (test) | Dice (test) |
|---|---|---|---|---|---|
| `patel` | Patel CNN (encoder propio) | ~2M | Outer fold 9 | 0,9987 | 0,871 |
| `efficientnet_b4` | EfficientNet-B4 (ImageNet) | ~19M | Outer fold 7 | 0,9987 | 0,941 |
| `vit` *(pendiente)* | ViT-B/16 (ImageNet-21K) | ~86M | — | — | — |

### 3.2 Formato de imagen de entrada

| Atributo | Valor |
|---|---|
| Tipos aceptados | `image/jpeg`, `image/png` |
| Tamaño máximo | 10 MB |
| Resolución de entrada al modelo | 224 × 224 px (redimensionado internamente con interpolación bilineal) |
| Rango de valores | [0, 1] float32 (normalización aplicada internamente) |

---

## 4. Variables de entorno del servidor

| Variable | Por defecto | Descripción |
|---|---|---|
| `PATEL_MODEL_PATH` | `models/patel_outer9.pt` | Ruta al checkpoint del encoder Patel |
| `EFFICIENTNET_MODEL_PATH` | `models/efficientnet_outer7.pt` | Ruta al checkpoint de EfficientNet-B4 |
| `DNIS_DIR` | `DNIs/` | Directorio con las imágenes y JSON de demo |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:4173` | Orígenes CORS permitidos (separados por comas) |
| `API_KEY` | *(vacío — sin autenticación)* | Clave de API opcional |
| `MAX_IMAGE_BYTES` | `10485760` (10 MB) | Límite de tamaño de imagen en bytes |

---

*Fin de la especificación de la API.*
