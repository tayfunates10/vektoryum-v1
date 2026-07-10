# Proje Bağlamı — Vektoryum

Raster→vektör dönüşümü için araştırma + üretim hibrit kod tabanı.
Mimari harita ve yol haritası: `docs/VEKTORLESTIRME_SISTEMI.md`.

## Depo yerleşimi (iki motor gerçeği)

- `engine/app/` — **Python kalite motoru** (FastAPI): analyzer → preprocess →
  çok-adaylı VTracer → shape fitting → algısal skorlama (SSIM/ΔE/edge-F1) →
  ölçüm-kapılı refinement → edge_cleanup → SVG/PDF/EPS/DXF/PNG export.
  Bu depodaki asıl Ar-Ge birikimi budur; bozan değişiklik regresyonla yakalanmalı.
- `server.ts` + `vectorizer.ts` — **Node/Express hızlı tracer** (potrace/sharp).
  Kök `Dockerfile` bunu derler; HF Spaces'te şu an canlı olan motor budur.
- `services/` — v2 mikroservis iskeleti (gateway + kuyruk + worker); üretimde değil.
- `engine/app/static/index.html` — tek dosyalık vanilla HTML/CSS/JS frontend;
  Node sunucusu da aynı dosyayı servis eder.

## Zorunlu mühendislik kuralları

- Motor (engine) değişikliğinden sonra çalıştır (testler pytest değil,
  bağımsız koşuculardır; repo kökünden):
  `cd engine && .venv/bin/python test_visual_regression.py && .venv/bin/python test_artifact_quality.py`
  (tek vaka: `--case <ad>`, baseline yenileme: `--update-baseline`).
  Regresyon FAIL iken commit önerme.
- Eğri/geometri değişikliklerinde `engine/regression/fidelity_report.py` ile
  sadakat + path_count karşılaştırması raporla; skor düşüşü ölçümle gerekçelendirilmeli.
- İyileştirme adımları **ölçüm kapılıdır**: skor düşürürse geri alınır. Yeni bir
  adım eklerken aynı deseni kullan (uygula → yeniden skorla → kötüyse eski aday).
- Exporter değişikliklerinde 5 formatın tümü (`exporters.py`) üretilip
  render doğrulaması yapılmalı (resvg birincil, PyMuPDF yalnız fallback).
- Node tarafında değişiklik sonrası `npm run build` (esbuild) geçmeli.
- Frontend kuralları: arayüz ikonları **yalnızca SVG** (raster/emoji/icon-font
  yasak); header (logo + nav + Giriş/Kayıt) kullanıcı talebi olmadan değiştirilmez;
  mevcut yükleme/önizleme/vektörleştirme/indirme/auth akışları korunur.

## Deploy ve ortam

- `main`'e merge → `.github/workflows/hf-deploy.yml` → HF Space
  `ATESOGLU/Vektoryum` otomatik deploy (tüm repo rsync + temiz git init).
  Bu workflow ve kök `README.md` HF frontmatter'ı kullanıcıya aittir; geri alma.
- HF dosya sistemi **geçicidir**: kalıcı durum `engine/app/store.py` üzerinden
  HF Dataset'e senkronlanır (`VEKTORYUM_DATASET` + `HF_TOKEN` Space değişkenleri;
  ayarsızsa sessiz no-op).
- Bellek: motor küçük logoda bile ~826 MB tepe kullanır — 512 MB planlar OOM olur.
- `VEKTORYUM_TRACE_CAP=2200` ölçümle seçildi (kalite/süre dengesi); körlemesine
  yükseltme — 3000, süreyi 67 s → 110 s yapar.

## Kod stili

- Yorumlar ve kullanıcıya dönük metinler Türkçe; tanımlayıcılar İngilizce.
- Rastgelelik içeren test/benchmark'ta seed zorunlu; float karşılaştırmaları toleranslı.
- Büyük ikili test görselleri deploy'a girmez (`hf-deploy.yml` exclude listesi).
