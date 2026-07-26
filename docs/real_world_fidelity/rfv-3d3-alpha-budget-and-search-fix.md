# RFV-3D3 — alfa bütçesi ve aday arama süresi düzeltmesi

## Kanıt bağlaması

Bu düzeltme RFV-3B canlı production ölçümünün başarısız shard loglarına dayanır.

- base main SHA: `c76dfaa2c55233783e9547d6176dbbea1ec8ef0a`
- ölçüm sözleşmesi: immutable 24 vaka × 3 tekrar, 6 shard
- tamamlanan shard: 0, 2, 3
- başarısız shard: 1, 4, 5 (aggregate çalıştırılamadı)

Başarısızlıklar tek bir sınıfta toplanmadı; iki farklı kök neden ölçüldü.

### 1. Byte-budget reddi — `qualification-public-14`

```
source_alpha_candidate_knockout_byte_budget_rejected:
1800596 > 1273782
```

Bütçe `_journal_limits` ile üretilir (`max(3 × parent_bytes, parent_bytes + 250_000)`);
parent ≈ 424.594 bayt olduğundan sınır 1.273.782 bayttır. Reddedilen aday alfa doğruluğunu
sağlıyordu, sınırı aşan şey **markup maliyetiydi**.

### 2. Süre aşımı — `qualification-public-11`, `qualification-public-12`

```
TimeoutError: isolated benchmark repeat timed out
```

`qualification-public-12` loglarında kabul edilebilir bir polygon adayı zaten vardı
(`alpha_iou ≈ 0,9991`, `alpha_mae ≈ 0,00089`, `path_count = 36`,
`serialized_bytes = 188.115`, journal geçti). Buna rağmen turnuva, sonucu değiştiremeyecek
adayları da tam alfa değerlendirmesi + journal render'ından geçirmeye devam etti.

## Kök production yolu

### Knockout clip geometrisi

`alpha_candidate_knockout._build_reconstruction_tree` her alfa seviyesi için bir `clipPath`
üretir ve birleşmiş satır-koşusu dikdörtgenlerinin **her birini ayrı `<rect>` elementi**
olarak, 12 anlamlı basamaklı user-space koordinatlarla yazar. Bir dikdörtgen ≈ 87 bayt
tutar; on binlerce koşu içeren yumuşak alfa alanlarında toplam maliyet bütçeyi aşar.

`alpha_svg_mask` tarafında kompakt path kodlaması yıllardır mevcut (`alpha_mask_budget`
preflight'i rect/path seçer), ancak knockout adayı bu seçeneğe hiç sahip değildi.

### Painter turnuvası

`alpha_candidate_painter._evaluate_phase` her (encoding, stroke) kombinasyonu için
tam `_assess_painter_candidate` + `_run_painter_geometry_journal` zinciri çalıştırır.
Kazanan anahtar sözlükseldir: `(byte, path, node, sıra)`. Bir aday kabul edildikten sonra
bile, baytı kesinlikle daha büyük adaylar aynı pahalı zinciri kat ediyordu. Ayrıca her
journal çağrısı yeni bir `TransformJournal` yarattığı için **ebeveyn SVG her denemede
yeniden render ediliyordu**.

## Dar düzeltme

Hiçbir eşik yükseltilmedi, hiçbir tolerans genişletilmedi, timeout artırılmadı.

1. **Knockout clip kodlaması fallback'i** (`engine/app/alpha_candidate_knockout.py`)
   - `_CLIP_GEOMETRY_ENCODINGS = ("rect", "path-transform")`.
   - `rect` bugünkü kanıtlanmış davranıştır ve bütçeye sığdığı sürece **bayt-aynıdır**;
     hâlihazırda geçen vakalar etkilenmez.
   - `path-transform` YALNIZ rect bütçeyi aştığında denenir: aynı birleşmiş dikdörtgenler
     tam sayı raster koordinatlarında tek bir `<path>` olarak yazılır
     (`M{x} {y}h{w}v{h}h-{w}z`), raster→user dönüşümü clipPath'in kendi `transform`
     niteliğine taşınır. clipPath içine transform'lu `<g>` **yuvalanmaz**; resvg'nin
     daraltılmış alt kümesi korunur.
   - Alt yollar ayrıktır, bu yüzden nonzero/even-odd aynı bölgeyi verir; koordinatlar tam
     sayı olduğundan yuvarlama kaybı yoktur.
   - Renderer bu alt kümeyi desteklemezse değişmemiş alpha IoU/MAE ve journal kapıları
     adayı fail-closed reddeder; sessiz kabul yoktur.

2. **Turnuva elemesi** (`engine/app/alpha_candidate_painter.py`)
   - Bütçe içi bir aday, mevcut kazananın baytından **kesin olarak** büyükse sözlüksel
     anahtar gereği kazanması imkânsızdır; pahalı alfa değerlendirmesi ve journal render'ı
     atlanır, ledger'a `status="byte_dominated"`, `validation_stage="tournament_prune"`
     olarak yazılır.
   - Eşitlikte eleme yapılmaz (`>` kullanılır), böylece path/node/sıra kırıcıları aynen
     çalışır. Seçilen aday ölçüm-eşdeğerdir.

3. **Paylaşılan ölçüm önbelleği** (`engine/app/transform_journal.py`)
   - `TransformJournal(..., measurement_cache=...)` isteğe bağlıdır; verilmezse davranış
     birebir eskisidir (özel önbellek).
   - Anahtar; `source_rgb` digest'i, `max_side` ve `required_metrics` ile ad-alanlanır,
     böylece farklı konfigürasyonlar birbirine karışamaz.
   - Önbellek isabetinde **ilk ölçümün süresi yeniden işlenir**: değerlendirme bütçesi
     muhasebesi ve dolayısıyla `budget_exhausted` davranışı korunur; kazanç yalnız duvar
     saatindedir.
   - Painter, bir çalıştırma boyunca tek bir önbellek paylaşır; ebeveyn baytı her denemede
     aynı olduğu için render'ı bir kez ölçülür.

## Ölçüm

Sentetik yumuşak alfa alanı (400×400 viewBox, 200×200 raster, 31 seviye, 4.609 birleşmiş
dikdörtgen), `engine/test_rfv3d3_knockout_clip_encoding.py`:

| clip kodlaması | serileşmiş bayt | oran |
| --- | --- | --- |
| `rect` | 238.470 | 1,00× |
| `path-transform` | 84.524 | 0,35× |

- Render karşılaştırması: alpha IoU `1,0`, alpha MAE `0,0` (geometri birebir).
- `qualification-public-14` izdüşümü: 1.800.596 × 0,354 ≈ 638.000 bayt < 1.273.782 sınır.
  Bu vakadaki ölçek kesirli olduğu için gerçek kazancın daha yüksek olması beklenir; bu
  yalnız alt sınırdır ve nihai karar CI ölçümüne aittir.

Painter turnuvası, `engine/test_rfv3d3_search_pruning.py` ve mevcut painter ledger'ı:
kabul edilen polygon adayından (2.702 bayt) sonra dört rect denemesi
`byte_dominated` olarak elendi; hiçbiri alfa değerlendirmesine veya journal render'ına
girmedi. Çıktı digest'i tekrarlar arasında aynıdır.

## Zorunlu yerel regresyon

- `engine/test_artifact_quality.py`: tüm kontroller PASS.
- `engine/test_visual_regression.py`: `class_reklam` PASS, `gradient_logo` PASS,
  **`arcaates` FAIL**.

`arcaates` başarısızlığı bu düzeltmeden bağımsızdır ve `main`'in kendisinde mevcuttur.
Aynı vaka üç ayrı ağaçta koşuldu ve **birebir aynı imzayı** verdi:

| ağaç | sonuç | imza |
| --- | --- | --- |
| bu düzeltme (`ee39501`) | FAIL, exit 1 | `source_alpha_mask_rectangle_budget_exceeded:50488>8251` |
| PR #121 head (`fdfbe63`) | FAIL, exit 1 | `source_alpha_mask_rectangle_budget_exceeded:50488>8251` |
| `main` (`c76dfaa`) | FAIL, exit 1 | `source_alpha_mask_rectangle_budget_exceeded:50488>8251` |

Kök neden `alpha_mask_budget._preflight`'tedir: ayrıntılı rect geometrisi bütçeyi aşar
(50.488 dikdörtgen > 8.251 sınır) ve `_build_contour_plan` alternatifi path/node/byte
limitlerini geçemediği için fail-closed hata atılır. Bu, bu PR'de knockout için çözülen
sorunun **aynı sınıfı**dır ancak farklı bir üretim yolundadır; kapsam dışıdır ve ayrı,
ölçüm-kapılı bir düzeltme gerektirir.

## Değişmeyenler

- evaluator ve kalite eşikleri
- immutable corpus ve ölçüm sözleşmesi
- benchmark retry/timeout politikası
- `_journal_limits` bütçe formülü
- seçilen aday politikası ve turnuva sıralama anahtarı
- serializer ve export yolu

## Release durumu

Bu düzeltme kök nedenleri kapatır ancak tek başına karar üretmez. RFV-3 kararı yalnız
24 vaka × 3 tekrar = 72 başarılı örnek ve aggregate artefaktı üretildikten sonra
güncellenir.

- `RFV-3: pending`
- `release_decision: NO-GO`
- `production_rollout_allowed: false`
- `rfv4_allowed: false`
