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
#
# HF cpu-basic GÜVENLİ profili (bellek/kararlılık > ham hız):
# - VEKTORYUM_WORKERS=1: aday üretimi SIRALI (çoklu alt-süreç bellek patlaması ve
#   OOM riski yok; motor küçük logoda bile ~826 MB tepe yapar). Güçlü makinede
#   env ile artırılabilir.
# - VEKTORYUM_JOB_TIMEOUT=300: asılı native işi kontrollü kesme.
# - Girdi sınırları (byte/kenar/piksel) sınırsız/0 bırakılmaz (decompression bomb
#   ve OOM koruması); tek kaynak app/settings.py.
ENV PYTHONUNBUFFERED=1 \
    VEKTORYUM_WORKERS="1" \
    VEKTORYUM_JOB_TIMEOUT="300" \
    VEKTORYUM_MAX_UPLOAD_BYTES="15728640" \
    VEKTORYUM_MAX_IMAGE_SIDE="12000" \
    VEKTORYUM_MAX_IMAGE_PIXELS="40000000" \
    VEKTORYUM_ALLOWED_FORMATS="PNG,JPEG,WEBP" \
    VEKTORYUM_TRACE_CAP="2200" \
    PORT=7860

EXPOSE 7860

# Konteyner sağlığı: HIZLI /livez (event loop yaşıyor mu). Ağır iş uçarken bile
# yanıt verir (iş threadpool + alt-süreç havuzunda). curl yerine stdlib python.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','7860')+'/livez',timeout=3)" || exit 1

# `exec` ile uvicorn PID 1 olur -> SIGTERM temiz kapanış.
CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
