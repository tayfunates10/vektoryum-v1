# Core Vector Engine Closure Roadmap

Bu roadmap yalnız çekirdek raster-to-vector motorunun kapanışını kapsar. Görsel
sınıflandırma, güven skoru ve otomatik mod/motor seçimi ayrı AI analyzer
roadmap'inde ele alınır. İlerleme, yalnız `main` dalına merge edilmiş kabul
fazları üzerinden hesaplanır.

Makinece doğrulanan kaynak: `engine/core_vector_engine_roadmap.json`.

## CVE-1 — Finite capability and closure contract

Durum: **complete**

- Üretim modları ve bilinen motorlar tek manifestte sabitlenir.
- Her mod en az bir zorunlu adaya sahip olmalıdır.
- Placeholder veya ürün limiti olan her konu tek bir kapanış fazına bağlanır.
- Roadmap şeması, kanıt dosyaları ve aday planları CI'da fail-closed doğrulanır.
- Production davranışı değişmez.

## CVE-2 — Deterministic centerline fallback closure

Durum: **complete in this PR**

- `opencv_skeleton` public engine adı korunur, ancak fallback artık skeleton
  konturunu değil skeleton grafını izler.
- Komşu junction pikselleri tek düğümde birleştirilir; her graph edge tam bir kez
  açık stroke yollarına serileştirilir.
- Yalnız kısa endpoint→junction spur'ları `min_branch` sözleşmesiyle budanır;
  bağımsız çizgiler, endpoint'ler, çok-yollu junction'lar ve loop'lar korunur.
- SVG içine backend, topoloji, edge coverage ve confidence raporu deterministik
  metadata olarak gömülür.
- Ölçülemeyen, izole düğümlü veya edge coverage'ı eksik graph hiçbir SVG
  yayımlamaz; defense-in-depth kalite kapısı da `production_ready` hükmünü
  engeller.
- Line, polyline, T, cross, loop, spur-pruning ve repeat-digest fixture'ları CI'da
  zorunludur.

## CVE-3 — Curve-preserving cutout and topology closure

Durum: **pending**

Mevcut `shape_stacking=cutouts` yolu pyclipper için eğrileri poligona
örnekleyebiliyor. Production API'de Bézier/yay geometri artık
polygon-flattening yoluna sokulmayacak. Promotion-ready canonical yüz belgesi
veya eğri-koruyan counter modeli kullanılamıyorsa stacked çıktı aynen korunacak.
Kısmi mutation yayımlanmayacak; seam/halo ve komut büyümesi sözleşmeleri CI ile
kilitlenecektir.

## CVE-4 — All-mode artifact and corpus release closure

Durum: **pending**

Tüm açık üretim modları için üç tekrarlı deterministik corpus sonucu veya açık
bir `needs_review/unavailable` hükmü zorunlu olacak. Final artifact digest,
bitmap, sonlu koordinat, cycle kapanışı ve stale metric kontrolleri tek release
kapısında birleşecektir. In-domain corpus için mevcut artefakt eşikleri korunur:

- `ink_recall >= 0.995`
- `ink_precision >= 0.975`
- `component_delta == 0`
- `seam_ratio <= 0.002`
- `halo_ratio <= 0.02`

Fotoğraf/sürekli-ton girdilerde düşük sadakat doğal ürün sınırı olarak açıkça
`needs_review` kalır; yanlış `production_ready` hükmü kapanış kriterini bozar.

## Yüzde hesabı

Toplam dört merge-kapılı faz vardır. Bir faz ancak kendi PR'ı tüm zorunlu CI
işleri yeşilken sabit head SHA ile `main` dalına merge edildiğinde tamamlanmış
sayılır. Dolayısıyla her faz toplam çekirdek motor roadmap'inin `%25`'idir.
