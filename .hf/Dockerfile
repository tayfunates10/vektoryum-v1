# Hugging Face Spaces (Docker SDK) derleme imajı — derleme bağlamı repo KÖKÜdür.
# (Yerel/Render için engine/Dockerfile kullanılır; bu yalnız HF Space içindir.)
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libcairo2 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/engine

COPY engine/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY engine/app ./app
COPY engine/models ./models

# HF free kademe 16GB RAM verir: paralel aday + daha yüksek girdi sınırı rahat.
ENV PYTHONUNBUFFERED=1 \
    VEKTORYUM_WORKERS=2 \
    VEKTORYUM_MAX_INPUT_SIDE=2600

EXPOSE 8000

# HF, app_port'a (README frontmatter = 8000) yönlendirir ve PORT env'ini verir;
# exec ile uvicorn PID 1 -> temiz kapanış.
CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
