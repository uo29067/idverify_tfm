# Especificación de Requisitos — Demo DocVerify

**Proyecto:** DocVerify — Detección de Documentos de Identidad Falsificados  
**TFM:** Máster Universitario en Ingeniería Informática — Universidad de Oviedo  
**Autora:** Inés Fernández Álvarez (uo29067)  
**Tutor:** Hernán Díaz Rodríguez  
**Versión:** 1.0 — Julio 2026

---

## 1. Descripción general

La demo DocVerify es una aplicación web de demostración que permite visualizar el funcionamiento del sistema de detección de documentos de identidad falsificados desarrollado en el TFM. La aplicación consta de un **backend** (API REST en FastAPI/PyTorch) y un **frontend** (interfaz web en React + Vite).

El objetivo no es producción sino **demostración**: ilustrar la capacidad del sistema DocVerify sobre documentos del conjunto de test de FantasyID, comparando los distintos encoders entrenados.

---

## 2. Requisitos funcionales

### RF-01 — Galería de documentos de demo

**Descripción:** El sistema deberá mostrar una galería de documentos de identidad precargados del conjunto de test de FantasyID.  
**Prioridad:** Alta  
**Criterio de aceptación:** La galería muestra al menos 10 documentos con miniatura de imagen identificable.  
**Endpoint:** `GET /dnis`

---

### RF-02 — Análisis automático al seleccionar documento

**Descripción:** Al hacer clic en un documento de la galería, el sistema deberá lanzar automáticamente el análisis de detección de falsificación sin intervención adicional del usuario.  
**Prioridad:** Alta  
**Criterio de aceptación:** La máscara de detección superpuesta aparece en menos de 10 segundos tras la selección (en condiciones normales de red y CPU).

---

### RF-03 — Selección de encoder

**Descripción:** El usuario podrá seleccionar el encoder del modelo DocVerify con el que se realiza el análisis. Encoders disponibles:
- **Patel CNN** (~2M parámetros, entrenado desde cero)
- **EfficientNet-B4** (~19M parámetros, preentrenado en ImageNet)
- **ViT-B/16** (~86M parámetros, preentrenado en ImageNet-21K) *(pendiente de resultados NCV)*

**Prioridad:** Alta  
**Criterio de aceptación:** Cambiar el encoder relanza el análisis automáticamente sobre el documento seleccionado.  
**Endpoint:** parámetro `encoder` en `POST /analyze`

---

### RF-04 — Visualización de la máscara de detección

**Descripción:** El sistema deberá mostrar el documento analizado con la máscara de segmentación superpuesta, resaltando las regiones identificadas como potencialmente alteradas.  
**Prioridad:** Alta  
**Criterio de aceptación:** Se muestra el documento completo (no recortado) con la máscara superpuesta en escala de color (azul = auténtico, rojo = sospechoso de alteración).

---

### RF-05 — Modos de visualización

**Descripción:** El usuario podrá alternar entre tres vistas del resultado:
- **Máscara superpuesta:** imagen original con heatmap semitransparente (vista por defecto)
- **Mapa de calor:** heatmap puro sin la imagen
- **Original:** imagen sin ninguna superposición

**Prioridad:** Media  
**Criterio de aceptación:** Los tres modos son accesibles desde la interfaz y se actualizan instantáneamente (sin nueva llamada al backend).

---

### RF-06 — Indicador de probabilidad de falsificación

**Descripción:** El sistema deberá mostrar la probabilidad de falsificación del documento como un porcentaje, con una barra de progreso y un veredicto textual.  
**Prioridad:** Alta  
**Criterio de aceptación:**
- El valor está entre 0 % y 100 %.
- Se muestra el veredicto **FALSIFICADO** (≥ 50 %) o **AUTÉNTICO** (< 50 %).
- El color de la barra varía según el nivel de riesgo: verde (bajo), amarillo (medio), rojo (alto).

---

### RF-07 — Metadatos de regiones del documento

**Descripción:** El sistema deberá exponer los metadatos de las regiones anotadas de cada documento (coordenadas, campo, procedencia original/alterado).  
**Prioridad:** Baja  
**Criterio de aceptación:** `GET /dnis/{doc_id}/meta` devuelve la lista de regiones con campos `x`, `y`, `w`, `h`, `field_name` y `region_provenance`.

---

### RF-08 — Servicio de imágenes estático

**Descripción:** Las imágenes de los documentos de demo deberán ser accesibles mediante URL directa para que el frontend pueda mostrarlas.  
**Prioridad:** Alta  
**Criterio de aceptación:** `GET /DNIs/{filename}` devuelve la imagen con el tipo MIME correcto.

---

## 3. Requisitos no funcionales

### RNF-01 — Seguridad: validación de entradas

**Descripción:** El backend deberá validar todas las entradas para prevenir vulnerabilidades comunes.  
**Criterios:**
- Los identificadores de documento se validan con expresión regular `^[A-Za-z0-9._-]+$` (RF-07), rechazando intentos de path traversal.
- Solo se aceptan imágenes JPEG y PNG (`image/jpeg`, `image/png`).
- El tamaño máximo de imagen es 10 MB.
- Las imágenes vacías (0 bytes) son rechazadas con HTTP 400.

---

### RNF-02 — Seguridad: control de acceso entre orígenes (CORS)

**Descripción:** El backend deberá restringir las peticiones a orígenes autorizados.  
**Criterios:**
- Orígenes permitidos configurables mediante variable de entorno `CORS_ORIGINS`.
- Por defecto: `http://localhost:5173` y `http://localhost:4173` (desarrollo).
- En producción: URL del despliegue de Netlify.

---

### RNF-03 — Rendimiento: tiempo de respuesta

**Descripción:** La inferencia debe completarse en un tiempo razonable para una demo interactiva.  
**Criterios:**
- Tiempo de respuesta de `POST /analyze` < 10 s en CPU (HF Spaces tier gratuito).
- Los endpoints de metadatos (`/dnis`, `/dnis/{id}/meta`) responden en < 500 ms.

---

### RNF-04 — Portabilidad: despliegue en la nube

**Descripción:** El sistema deberá poder desplegarse en servicios de nube gratuitos sin modificaciones de código.  
**Criterios:**
- Backend: imagen Docker compatible con HF Spaces (puerto 7860), configurable por variables de entorno.
- Frontend: build estático (`npm run build`) compatible con Netlify, con SPA redirect configurado en `netlify.toml`.

---

### RNF-05 — Mantenibilidad: cobertura de tests

**Descripción:** El código del backend deberá estar cubierto por pruebas automáticas.  
**Criterios:**
- Suite de 87 tests (62 unitarios + 25 de integración) ejecutable con `python -m pytest`.
- Los tests no cargan los modelos reales (modelos mock) para poder ejecutarse sin los ficheros de pesos.
- Tiempo de ejecución de la suite completa < 30 s.

---

### RNF-06 — Usabilidad

**Descripción:** La interfaz debe ser comprensible para un evaluador no técnico en el contexto de una presentación.  
**Criterios:**
- La interacción principal (seleccionar DNI → ver resultado) no requiere más de un clic.
- El resultado es visible sin necesidad de desplazamiento (*scroll*) en una pantalla de 1280×720 px o superior.
- La interfaz es responsiva y funciona en pantallas de al menos 360 px de ancho.

---

### RNF-07 — Compatibilidad de navegadores

**Descripción:** El frontend deberá funcionar en los navegadores modernos más utilizados.  
**Criterios:** Chrome ≥ 110, Firefox ≥ 110, Edge ≥ 110, Safari ≥ 16.

---

## 4. Restricciones del sistema

| Restricción | Descripción |
|---|---|
| CPU únicamente | Los modelos se ejecutan en CPU (sin GPU en el tier gratuito de HF Spaces). |
| Sin subida de imágenes | La versión demo no permite subir imágenes propias; solo documentos precargados. |
| ViT pendiente | El encoder ViT-B/16 se añadirá cuando finalice el entrenamiento NCV (previsto 5/7/2026). |
| Modelos de un pliegue | La demo usa el modelo del mejor pliegue NCV de cada encoder, no el ensemble de los 10. |

---

## 5. Casos de uso principales

### CU-01 — Analizar documento precargado

| Campo | Descripción |
|---|---|
| **Actor** | Usuario de la demo |
| **Precondición** | La aplicación está desplegada y los modelos cargados |
| **Flujo principal** | 1. El usuario abre la aplicación. 2. El sistema muestra la galería de documentos. 3. El usuario hace clic en un documento. 4. El sistema descarga la imagen, la envía al backend y muestra la máscara superpuesta con la probabilidad de falsificación. |
| **Flujo alternativo** | Si el backend no responde, se muestra un mensaje de error en la interfaz. |
| **Postcondición** | Se muestra el resultado de la inferencia sobre el documento seleccionado. |

---

### CU-02 — Comparar encoders

| Campo | Descripción |
|---|---|
| **Actor** | Usuario de la demo |
| **Precondición** | Un documento está seleccionado y analizado (CU-01) |
| **Flujo principal** | 1. El usuario selecciona un encoder diferente en el panel lateral. 2. El sistema relanza automáticamente el análisis con el nuevo encoder. 3. El sistema muestra el nuevo resultado. |
| **Postcondición** | El resultado refleja la inferencia del encoder seleccionado sobre el mismo documento. |

---

### CU-03 — Explorar vistas del resultado

| Campo | Descripción |
|---|---|
| **Actor** | Usuario de la demo |
| **Precondición** | Un resultado de análisis está disponible (CU-01) |
| **Flujo principal** | 1. El usuario pulsa uno de los botones de vista (Máscara superpuesta / Mapa de calor / Original). 2. La imagen mostrada cambia instantáneamente. |
| **Postcondición** | Se muestra la vista seleccionada sin nueva llamada al backend. |

---

*Fin del documento de requisitos.*
