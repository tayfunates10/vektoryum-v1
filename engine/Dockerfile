# Vektoryum API — üretim imajı
#
#   docker build -t vektoryum ./engine
#   docker run -p 8000:8000 vektoryum
#
# HED derin kenar modeli (58MB, opsiyonel; yoksa motor klasik yolla çalışır):
#   docker build --build-arg FETCH_HED=1 -t vektoryum ./engine
# ya da hazır model dosyalarını çalıştırırken bağlayın:
#   docker run -p 8000:8000 -v /yol/modeller:/srv/engine/models vektoryum

FROM python:3.11-slim

# Sistem bağımlılıkları:
# * libglib2.0-0  — opencv-python-headless çalışma zamanı
# * libcairo2     — CairoSVG (PDF/EPS render; yoksa saf-Python fallback da var
#                   ama imajda bulunması çıktı kalitesi için tercih edilir)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libcairo2 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/engine

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models

# Opsiyonel: HED modelini imaja göm (varsayılan kapalı; imaj küçük kalsın)
ARG FETCH_HED=0
RUN if [ "$FETCH_HED" = "1" ]; then python models/fetch_hed.py; fi

# VEKTORYUM_WORKERS: paralel aday üretimi işçi sayısı (boş = min(çekirdek, 4))
# VEKTORYUM_TRACE_CAP: renkli modda izleme çözünürlük tavanı (px). Yüksek-bellekli
#   üretim (HF 16GB) için 3000: büyük logodaki minik öğeler (® simgesi, ince metin)
#   1600'de çokgenleşip daireyi kaçırıyordu; 3000'de temiz izlenir (LEGO ® çemberi
#   düzeldi, fidelity 97.5->98.9). Küçük girdiler zaten tavanın altında -> etkilenmez.
ENV PYTHONUNBUFFERED=1 \
    VEKTORYUM_WORKERS="" \
    VEKTORYUM_TRACE_CAP="3000"

EXPOSE 8000

# Tek uvicorn süreci: pipeline kendi süreç havuzunu yönetir (forkserver);
# ek uvicorn worker'ı çekirdek kapışması yaratır. Port, barındırıcının verdiği
# $PORT'a uyar (Render/Cloud Run vb.); tanımsızsa 8000 (yerel docker-compose).
# `exec` ile uvicorn PID 1 olur -> SIGTERM temiz kapanış.
CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
