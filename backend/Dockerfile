FROM python:3.11-slim

WORKDIR /app

# System libs needed by SimpleITK/nibabel/matplotlib/tensorflow wheels at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Model artifacts and backend source, matching ml.py's MODEL_DIR default
# (two directories above ml.py -> repo root) so no code changes are needed.
COPY clinical_model.pkl best_prognostic_model.pkl \
     unet_multiclass.h5 combined_radiomics_features.csv ./
COPY backend/app backend/app

# Render (Docker runtime) routes traffic to port 7860 by default.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}"]
