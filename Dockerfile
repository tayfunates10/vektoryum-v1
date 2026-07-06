# Hugging Face Spaces Docker entrypoint for Vektoryum.
#
# Hugging Face builds Docker Spaces from the repository root. The production
# Dockerfile used by Render lives at engine/Dockerfile, but Spaces will not use
# that nested Dockerfile automatically. Keep this root Dockerfile in sync with
# engine/Dockerfile so https://huggingface.co/spaces/ATESOGLU/Vektoryum builds
# the current app after the GitHub Action pushes this repository.

FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libcairo2 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/engine

COPY engine/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY engine/app ./app
COPY engine/models ./models

ARG FETCH_HED=0
RUN if [ "$FETCH_HED" = "1" ]; then python models/fetch_hed.py; fi

ENV PYTHONUNBUFFERED=1 \
    VEKTORYUM_WORKERS="1" \
    VEKTORYUM_TRACE_CAP="2200" \
    VEKTORYUM_MAX_INPUT_SIDE="1800" \
    VEKTORYUM_DATA_ROOT="/data/vektoryum"

EXPOSE 8000

CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
