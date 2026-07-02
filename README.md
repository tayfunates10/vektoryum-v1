# Vektoryum API

FastAPI tabanlı raster-to-vector motoru. JPG, PNG ve WebP girdilerini analiz
eder, uygun trace profilini seçer, birden fazla SVG adayı üretir, geometriyi
temizler, en iyi adayı kalite skoruyla seçer ve **SVG, PDF, EPS, DXF** çıktıları
hazırlar.

> Bu proje tamamen özgün ve açık kaynak araçlarla (VTracer, OpenCV, svgpathtools,
> ezdxf, opsiyonel Potrace/AutoTrace/CairoSVG/Inkscape) çalışır. Herhangi bir
> kapalı/ticari servisin algoritması, kodu veya davranışı kopyalanmamıştır.

## Mimari

Çekirdek akış (`app/pipeline.py` → `run_pipeline`; hem API hem ölçüm CLI'si
aynı kodu çağırır):

1. **Analyzer** (`app/analyzer.py`) — görsel tipi, renk/kenar/gradyan analizi,
   `detected_type` ve `recommended_mode`.
2. **Preprocess** (`app/preprocess.py`) — profil bazlı ön işleme + palet kontrolü.
   `logo_color`'da palet bütçesi içerik-ölçeklidir (renk/ton zenginliğine göre
   16-64; hata-güdümlü ek kümeler + hedefli gürültü birleştirme). Küçük
   girdiler (<700px) 2x LANCZOS süperörneklenir: anti-alias gradyanı bölge
   sınırlarını alt-piksel hassasiyetle konumlandırır (küçük logolarda +2..+8
   sadakat puanı ölçüldü).
3. **Aday üretimi** (`app/vector_engines.py`) — VTracer / OpenCV contour /
   gradyan-farkındalıklı motor (`app/gradient_vectorize.py`) / opsiyonel
   Potrace / opsiyonel AutoTrace ile çoklu aday.
4. **Geometri temizleme** (`app/geometry_cleanup.py`) — düz çizgi/köşe temizliği,
   eksen yaslama, doğrusal nokta birleştirme.
5. **Algısal skorlama** (`app/scoring.py` + `app/fidelity.py`) — adayı render
   edip (resvg → PyMuPDF → CairoSVG → svglib) orijinalle karşılaştırır:
   **SSIM + CIELAB ΔE + kenar-F1** birleşik **fidelity skoru**. Render hiçbir
   backend'le yapılamazsa yapısal/geometrik skorlara güvenle düşülür.
6. **Seçim** (`app/pipeline.py`) — renkli modlarda gerçek sadakate yaslanır;
   sadakat marjı içinde belirgin daha az path'li adayı tercih eder
   (`editability_preference`). `selection_reason` raporlanır.
7. **Refinement (kapalı döngü)** (`app/pipeline.py` → `refine_best`) — en iyi
   adayın komşuluğunda parametre + renk-sayısı varyantları üretip yeniden ölçer;
   yalnızca daha sadık varyantı benimser.
8. **Export** (`app/exporters.py`) — SVG/PDF/EPS/DXF.
9. **Yapı bütünlüğü denetimi** (`app/fidelity.py` → `score_structure_integrity`) —
   nihai çıktı render edilip orijinalle karşılaştırılır: kopan/eksik çizgi
   (`ink_recall`), hayalet çizik (`ink_precision`) ve şekil parçalanması
   (`component_delta`) ölçülür. Kırık yapı tespit edilirse çıktı asla
   `production_ready` işaretlenmez.
10. **Kalite raporu** (`app/quality.py`) — `production_ready` / `needs_review`.
    Ölçülen sadakat düşükse (foto/sürekli-tonlu girdi) `needs_review` + dürüst uyarı.

### Ölçüm harness'i (`regression/fidelity_report.py`)

Motordaki her değişikliğin kaliteye etkisi sayısal ölçülür:

```powershell
.\.venv\Scripts\python.exe regression\fidelity_report.py            # manifest vakaları
.\.venv\Scripts\python.exe regression\fidelity_report.py <klasör>   # bir klasördeki tüm görseller
.\.venv\Scripts\python.exe regression\fidelity_report.py <görsel> --no-refine --progress
```

Her aday için `fidelity / ssim / ΔE / edge_f1 / path / renk` ve seçilen adayın
nedeni dökülür. `regression/samples/` (gitignore'da) gerçek görsellerle toplu
ölçüm içindir.

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
  JSON döner: `analysis`, `mode_used`, `mode_warning`, `candidate_report`,
  `quality_report`, `refine_info`, `outputs`, `output_errors`, `download_links`.
- `GET /api/download/{job_id}/{file_type}` — `file_type` ∈ `svg | pdf | eps | dxf`.
- `GET /` — sağlık kontrolü.

### Yanıt alanları (frontend için)

- `candidate_report.best_candidate` / `selection_reason` — seçilen aday ve neden
  (`highest_fidelity`, `editability_preference`, `refined`, `highest_total_score`…).
- `candidate_report.candidates[].fidelity_score` — adayın algısal sadakati (0-100;
  render edilemezse `null`). `details` içinde `ssim`, `mean_delta_e`, `edge_f1`.
- `refine_info` — refinement uygulandı mı, `base_fidelity` → `refined_fidelity`,
  denenen varyantlar.
- `quality_report.status` — `production_ready` | `needs_review` | `failed`.
  `quality_report.warnings` — kullanıcıya gösterilecek uyarılar (ör. düşük sadakat:
  "görsel fotografik görünüyor, çıktı yaklaşık").
  `quality_report.metrics.fidelity_score` — nihai çıktının ölçülen sadakati.

> **Sadakat skorunu UI'da göster:** brand guide'daki "Genel Kalite %92" göstergesi
> doğrudan `quality_report.metrics.fidelity_score` ile beslenebilir.

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

## Render ve Algısal Sadakat

Aday SVG'leri raster'a çevirip orijinalle karşılaştırmak (skorlamanın çekirdeği)
için bir render backend'i gerekir. Sırayla denenir:

- **resvg** (`resvg-py`) — **birincil**. Referans-kalite; gradyan/pattern/clip
  dahil tam destek. Gradyan-farkındalıklı adayların doğru puanlanması için şart.
- **PyMuPDF** — fallback; DLL'siz çalışır ama SVG gradyanlarını render etmez.
- **CairoSVG / svglib+reportlab** — ek fallback'ler (cairo DLL gerektirebilir).

Hiçbiri yoksa skorlama yapısal/geometrik metriklere güvenle düşer; sistem çökmez.

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
- **HED derin kenar modeli** (`app/dl_segmentation.py`) — opsiyonel derin
  öğrenme segmentasyon katmanı (Holistically-Nested Edge Detection, açık
  kaynak caffemodel, `cv2.dnn` ile CPU'da; ek pip bağımlılığı yok).
  `python models/fetch_hed.py` ile indirilir. Varsa kuantizasyonun düz-bölge /
  yapılı-bölge kararları anlamsal kenar haritasıyla harmanlanır (Sobel +
  derin kenar): gürültülü zeminlerde leke temizliği belirginleşir, gerçek
  detaylar korunur. Yoksa tüm kararlar salt Sobel'le, önceki davranışla
  birebir aynı alınır.

Ortam değişkenleri (opsiyonel):

```powershell
$env:INKSCAPE_PATH = "C:\Program Files\Inkscape\bin\inkscape.exe"
$env:POTRACE_PATH  = "C:\Tools\potrace\potrace.exe"
$env:AUTOTRACE_PATH = "C:\Tools\autotrace\autotrace.exe"
$env:HED_PROTO_PATH = "C:\Models\deploy.prototxt"            # opsiyonel
$env:HED_MODEL_PATH = "C:\Models\hed_pretrained_bsds.caffemodel"
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
10. Render backend yoksa sadakat skoru güvenle `None` dönüyor mu (çökme yok)
11. Potrace yoksa fallback çalışıyor mu
12. AutoTrace yoksa fallback/warning çalışıyor mu

Ek olarak `test_real_fixtures.py` (gerçek fixture regresyonu, 12 kabul kriteri),
`test_synthetic_vector_quality.py` (6 sentetik uçtan-uca vaka: geometrik /
tek-renk kesim / lineart / çok renkli / gradyan / foto),
`test_visual_regression.py` (manifest + baseline PNG karşılaştırmalı görsel
regresyon; `--update-baseline` ile baseline yenilenir) ve
`regression/fidelity_report.py` (algısal sadakat ölçümü) bulunur.

### Artefakt regresyonu (`test_artifact_quality.py`)

Kırık çizgi / hairline çizik / kenarlık kopması / renk hatası artefaktlarını
hedefleyen sentetik stres vakalarını (`regression/artifact_probe.py`) uçtan uca
çalıştırır ve şu kabul kriterlerini kilitler: `ink_recall >= 0.995` (hiçbir
çizgi kopmaz), `ink_precision >= 0.975` (hayalet çizik yok),
`component_delta == 0` (şekil parçalanmaz), `seam_ratio <= 0.002` (bitişik
renkler arasında zemin sızmaz), `halo_ratio <= 0.02` (palet dışı renk bandı
yok) + vaka bazlı sadakat tabanları. Kalite kapısının kırık çıktıyı asla
`production_ready` işaretlemediği de doğrulanır.

```powershell
.\.venv\Scripts\python.exe test_artifact_quality.py
.\.venv\Scripts\python.exe regression\artifact_probe.py   # yalnız ölçüm/teşhis
```

## Marka Kılavuzu

Vektoryum'un marka kimliği (renk paleti, tipografi, logo kuralları, ton ve
arayüz dili) için bkz. [`docs/BRAND_GUIDE.md`](docs/BRAND_GUIDE.md).

## Bilinen Sınırlamalar

- Fotoğraf / sürekli-tonlu görseller vektörleştirmenin doğal tavanındadır
  (gerçek görsel survey'inde ~57-77 sadakat). Bunlar `needs_review` + uyarı ile
  işaretlenir; in-domain logolar tipik olarak 85-98 alır.
- Algısal sadakat skoru bir render backend'i (resvg/PyMuPDF/CairoSVG/svglib)
  gerektirir; hiçbiri yoksa yapısal + geometrik skorlara düşülür.
- Gradyan-farkındalıklı aday yalnızca gradyan-baskın renkli logolarda kazanır;
  çok-renkli karmaşık logolarda VTracer adayları seçilir.
- `centerline` modu AutoTrace yoksa basit bir skeleton fallback kullanır (placeholder kalite).
