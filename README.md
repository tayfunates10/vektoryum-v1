# Vektoryum API

FastAPI tabanlı raster-to-vector motoru. JPG, PNG ve WebP girdilerini analiz
eder, uygun trace profilini seçer, birden fazla SVG adayı üretir, geometriyi
temizler, en iyi adayı kalite skoruyla seçer ve **SVG, PDF, EPS, DXF** çıktıları
hazırlar.

> Bu proje tamamen özgün ve açık kaynak araçlarla (VTracer, OpenCV, svgpathtools,
> ezdxf, opsiyonel Potrace/AutoTrace/CairoSVG/Inkscape) çalışır. Herhangi bir
> kapalı/ticari servisin algoritması, kodu veya davranışı kopyalanmamıştır.

## Mimari

İstek akışı (`/api/vectorize`):

1. **Analyzer** (`app/analyzer.py`) — görsel tipi, renk/kenar/gradyan analizi,
   `detected_type` ve `recommended_mode`.
2. **Preprocess** (`app/preprocess.py`) — profil bazlı ön işleme + palet kontrolü.
3. **Aday üretimi** (`app/vector_engines.py`) — VTracer / OpenCV contour /
   opsiyonel Potrace / opsiyonel AutoTrace ile çoklu aday.
4. **Geometri temizleme** (`app/geometry_cleanup.py`) — düz çizgi/köşe temizliği,
   eksen yaslama, doğrusal nokta birleştirme.
5. **Skorlama** (`app/scoring.py`) — yapısal + geometrik skorlar (+ CairoSVG varsa
   raster benzerlik).
6. **Seçim** (`app/main.py`) — profil bazlı en iyi aday + `selection_reason`.
7. **Export** (`app/exporters.py`) — SVG/PDF/EPS/DXF.
8. **Kalite raporu** (`app/quality.py`) — `production_ready` / `needs_review`.

## Kurulum

```powershell
cd C:\Users\TAYFUN\Desktop\Projeler\tabela-vector-saas\engine
python -m venv .venv
.\.venv\Scripts\activate
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

API dokümanı: `http://127.0.0.1:8000/docs`

## API

- `POST /api/vectorize` — form-data: `file` (görsel), `trace_mode` (varsayılan `auto`).
  JSON döner: `analysis`, `mode_used`, `candidate_report`, `quality_report`,
  `outputs`, `output_errors`, `download_links`.
- `GET /api/download/{job_id}/{file_type}` — `file_type` ∈ `svg | pdf | eps | dxf`.
- `GET /` — sağlık kontrolü.

## Modlar

- `auto` — Analyzer sonucuna göre modu otomatik seçer.
- `geometric_logo` — Az renkli, sert kenarlı tabela/monogram/yazı logoları için
  düz çizgi, köşe temizliği ve eksen hizalama odaklı mod.
- `minimal_ai` — Sade logo ve yazı görselleri için az renkli, düzenlenebilir SVG.
- `logo_color` — Çok renkli AI logo, illüstratif logo, gölge/taş/güneş/dağ ve
  posterize görseller için detay korumaya öncelik verir.
- `lineart` — Siyah-beyaz çizim, kroki ve outline işleri için binary trace.
- `single_color` — Tek renk silüet, stencil, kesim ve folyo işleri için.
- `centerline` — AutoTrace varsa centerline dener; yoksa OpenCV skeleton fallback.
- `photo_poster` — Fotoğraf benzeri girdiler için posterize hazırlık modu.
  Tam sadakat + az path + tam editlenebilirlik aynı anda garanti edilmez.
- `flat_logo` — Eski sade logo davranışını koruyan uyumluluk modu.

## Örnekler

- Class Reklam siyah-beyaz-kırmızı logo: `trace_mode=auto` → `mode_used=geometric_logo`,
  `best_candidate` ∈ {`geo_standard`, `geo_clean`, `geo_contour`, `geo_mixed`}.
- ARCAATES tarzı taş/güneş/dağ içeren çok renkli AI logo: `trace_mode=auto` →
  `mode_used=logo_color`.

## Opsiyonel Araçlar ve Dayanıklılık

Sistem, harici araçların hiçbiri olmadan da **SVG ve DXF** üretir; diğer formatlar
mümkünse üretilir, değilse `output_errors` içinde raporlanır ve API çökmez.

- **CairoSVG** — opsiyoneldir. Windows'ta cairo DLL yoksa hem import hem render
  güvenle atlanır (API başlangıcı durmaz).
- **PDF/EPS** — sırasıyla denenir: CairoSVG → Inkscape → `svglib`+`reportlab`
  (saf Python). Üçü de yoksa hata `output_errors` içinde döner.
- **DXF** — `svgpathtools` + `ezdxf` ile saf Python; Windows'ta güvenilir çalışır.
- **Inkscape** — varsa PDF/EPS render için fallback. Yoksa SVG ve DXF çalışmaya
  devam eder.
- **Potrace** — varsa binary/monochrome adaylarda kullanılır; yoksa aday
  `success:false`, `error:"potrace not found"` ile raporlanır.
- **AutoTrace** — varsa centerline/lineart adaylarında kullanılır; yoksa aday
  raporlanır ve centerline için skeleton fallback devreye girer.

Ortam değişkenleri (opsiyonel):

```powershell
$env:INKSCAPE_PATH = "C:\Program Files\Inkscape\bin\inkscape.exe"
$env:POTRACE_PATH  = "C:\Tools\potrace\potrace.exe"
$env:AUTOTRACE_PATH = "C:\Tools\autotrace\autotrace.exe"
```

## Testler

```powershell
cd C:\Users\TAYFUN\Desktop\Projeler\tabela-vector-saas\engine
.\.venv\Scripts\activate
.\.venv\Scripts\python.exe test_vector_engine.py
```

`test_vector_engine.py` 12 kontrol yapar:

1. `app.main` import ediliyor mu
2. `ALLOWED_MODES` içinde `geometric_logo` var mı
3. Sade siyah-beyaz-kırmızı görsel `geometric_logo` seçiliyor mu
4. Çok renkli görsel `logo_color` seçiliyor mu
5. `build_vector_candidates("geometric_logo")` en az 4 aday döndürüyor mu
6. `geometry_cleanup` import ediliyor mu
7. `_path_efficiency_score(22, 4, "geometric_logo")` 100 döndürüyor mu
8. `cleanup_svg_geometry` mevcut mu
9. `vectorize_geometric_contours_to_svg` mevcut mu
10. CairoSVG eksikliği sistemi çökertmiyor mu
11. Potrace yoksa fallback çalışıyor mu
12. AutoTrace yoksa fallback/warning çalışıyor mu

## Marka Kılavuzu

Vektoryum'un marka kimliği (renk paleti, tipografi, logo kuralları, ton ve
arayüz dili) için bkz. [`docs/BRAND_GUIDE.md`](docs/BRAND_GUIDE.md).

## Bilinen Sınırlamalar

- `photo_poster` çıktısı posterize bir yaklaşımdır; tam fotoğraf sadakati hedeflenmez.
- Raster benzerlik skoru yalnızca CairoSVG render edilebildiğinde kullanılır;
  aksi halde yapısal + geometrik skorlar esas alınır.
- `centerline` modu AutoTrace yoksa basit bir skeleton fallback kullanır (placeholder kalite).
