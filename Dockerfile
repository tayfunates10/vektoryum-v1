# Vektoryum — ÜRETİM imajı (HF Space bunu derler)
#
# MOTOR BİRLEŞMESİ: üretim backend'i artık Python KALİTE MOTORUDUR
# (engine/app — analyzer -> çok-adaylı VTracer -> algısal skorlama ->
# ölçüm-kapılı refinement -> gerçek renkli SVG + 5 format export).
#
# Neden: Node/potrace hızlı tracer'ı luminance eşiklemesiyle çalışır ve
# renkli logolarda MİMARİ olarak gerçek renk üretemez (çıktı: siyah path +
# fill-opacity katmanları; kırmızı/sarı/beyaz kaybolur — LEGO vakasında
# doğrulandı). Python motoru aynı görselde gerçek #E3000B/#FFED00/#FFFFFF/
# #000000 dolgularını, opak zemini, kaynak-boyut viewBox sözleşmesini ve
# küçük-bileşen korumalarını üretir. Node sunucusu geliştirme/hızlı deneme
# için depoda kalır: `npm run dev`.
#
# Yerel Node imajı gerekirse: docker build -f Dockerfile.node . (yok; dev
# için npm yeterli). HF portu README frontmatter'daki app_port=7860'tır.

FROM python:3.11-slim

# opencv-python-headless çalışma zamanı + CairoSVG (PDF/EPS) için
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libcairo2 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/engine

COPY engine/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY engine/app ./app
COPY engine/models ./models

# HF konteyneri root olmayan kullanıcıyla çalışır; motorun yazdığı yollar
# zaten /tmp altında (JOBS_ROOT, DATA_ROOT) — ev dizini hf_hub önbelleği
# (store.py kalıcılık senkronu) için gereklidir.
RUN useradd -m -u 1000 appuser && chown -R appuser /srv/engine
USER appuser
ENV HOME=/home/appuser

# VEKTORYUM_TRACE_CAP: renkli modda izleme tavanı (px) — 2200 ölçümle seçildi
# (küçük öğeler temiz, süre 1600 ile aynı; 3000 süreyi 67s->110s yapar).
# PORT: HF README app_port=7860 ile eşleşir.
ENV PYTHONUNBUFFERED=1 \
    VEKTORYUM_WORKERS="" \
    VEKTORYUM_TRACE_CAP="2200" \
    PORT=7860

EXPOSE 7860

# `exec` ile uvicorn PID 1 olur -> SIGTERM temiz kapanış.
CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
