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

Durum: **complete**

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

Durum: **complete**

- Public `shape_stacking=cutouts` girişi strict polygonal source contract ile
  korunur.
- Bézier/yay komutları, desteklenmeyen transform, stroke, açık path, unsupported
  primitive veya doğrulanamayan fill modeli görülürse boolean motor çalışmaz ve
  exact stacked baytları korunur.
- Yalnız kapalı `M/L/H/V/Z` fill path'leri private pyclipper motoruna geçebilir.
- Converter ikinci candidate dosyada çalışır; XML, sonlu koordinat, path coverage,
  command-growth ve digest kontrolleri geçmeden atomik publish yoktur.
- Dependency yokluğu, converter exception ve kısmi yazma stacked çıktıyı
  değiştiremez.
- Adjacent-color fixture için `seam_ratio <= 0.002`, `halo_ratio <= 0.02` ve
  bounded command growth CI'da zorunludur.

## CVE-4 — All-mode artifact and corpus release closure

Durum: **complete in this PR**

- Sekiz explicit üretim modu küçük, deterministik fixture'larla fresh process
  içinde üçer kez çalıştırılır.
- Bir modun üç artifact digest'i aynı değilse, repeat durumları karışıksa veya
  gerekli metriklerden biri ölçülemiyorsa release kapısı fail-closed kapanır.
- Final SVG için bitmap, non-finite geometri, açık olması yasak fill cycle,
  evaluator/output digest sapması ve score snapshot eskimesi doğrulanır.
- In-domain corpus eşikleri değişmeden korunur:

  - `ink_recall >= 0.995`
  - `ink_precision >= 0.975`
  - `component_delta == 0`
  - `seam_ratio <= 0.002`
  - `halo_ratio <= 0.02`

- `photo_poster`, skorları yüksek olsa bile açık ürün limiti olarak daima
  `needs_review` kalır; yanlış `production_ready` hükmü üretilemez.
- `Exact final SVG contract`, `Core all-mode release contract` ve
  `Benchmark v1 seed corpus` kapanışın zorunlu CI kanıtlarıdır.

## Yüzde hesabı

Toplam dört merge-kapılı faz vardır. Bir faz ancak kendi PR'ı tüm zorunlu CI
işleri yeşilken sabit head SHA ile `main` dalına merge edildiğinde tamamlanmış
sayılır. CVE-4 merge edildiğinde çekirdek motor roadmap'i `%100` olur ve bu
roadmap kapsamında yeni değişiklik açılmaz.
