# Build multi-stage: contexto de build = demo/ (esta carpeta), para poder copiar
# tanto demo_app/ (backend) como demo_frontend/ (frontend) en la misma imagen.
#   docker build -f Dockerfile -t docverify-demo .   (ejecutado desde demo/)

# ---- Etapa 1: build del frontend (React + Vite) ----
FROM node:20-slim AS frontend-build
WORKDIR /frontend
COPY demo_frontend/package.json demo_frontend/package-lock.json* ./
RUN npm ci
COPY demo_frontend/ ./
RUN npm run build

# ---- Etapa 2: backend (FastAPI) + frontend ya compilado ----
FROM python:3.11-slim
WORKDIR /app

# Dependencias del sistema para Pillow y compilación de wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python primero (capa cacheada)
COPY demo_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código y assets del backend
COPY demo_app/app.py demo_app/config.py demo_app/model.py ./
COPY demo_app/models/ ./models/
COPY demo_app/DNIs/ ./DNIs/

# Frontend ya compilado (vite build -> dist/), servido como estático por app.py
COPY --from=frontend-build /frontend/dist ./frontend/

# HF Spaces expone el puerto 7860
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
