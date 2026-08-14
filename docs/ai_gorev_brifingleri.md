# Vektoryum — 4 AI için ayrı görev brifingleri

Bu belgedeki her bölüm **bağımsız bir sohbete** kopyalanmak üzere yazıldı.
Hedef asistanlar sohbet modundadır: kod çalıştıramaz, dosya okuyamaz, CI göremez.
Bu yüzden her brifing kendi içinde eksiksizdir — ölçülmüş sayılar, ilgili kod
parçaları, elenmiş hipotezler ve kısıtlar içeridedir.

**Hepsi için ortak kurallar:**

1. Kalite kapısı eşiklerini gevşetme önerisi getirme (`alpha_iou_min`, `alpha_mae_max`,
   `seam_ratio`, `node_complexity_explosion` sınırı vb.). Kabul yetkisi değişmemiş
   evaluator + TransformJournal'da kalmalı.
2. Çıktı formatı: **(a)** kök neden analizi, **(b)** somut yama önerisi (unified diff
   veya tam fonksiyon gövdesi), **(c)** öneriyi çürütecek deney/ölçüm, **(d)** riskler.
3. Emin olmadığın yerde tahmin etme; "şu dosyanın şu fonksiyonunu görmem gerek" diye
   açıkça iste. Uydurma kod yolu/satır numarası verme.
4. Değişiklik kesin **eklemeli** olmalı: hâlihazırda kabul edilen adayların yolu
   değişmemeli.

**Ortak bağlam:** Motor, raster→vektör dönüşümünde kaynak alfayı bir SVG `<mask>`
üzerinden uyguluyor. Bir aday üretildikten sonra `TransformJournal` onu ebeveyn
artifact'la karşılaştırıp geometri kapılarından geçiriyor. Kapı düşerse "painter"
yeniden inşa turnuvası devreye giriyor; o da başarısız olursa fail-closed kalınıyor
(SVG değişmeden bırakılıyor).

RFV-3B ölçümünde 24 vakalık nitelenmiş korpustan **3 vaka** bu şekilde düşüyor.
Ledger sayaçları üç vakanın **üç ayrı kusur** olduğunu gösterdi:

| | `public-05` | `public-04` | `public-15` |
|---|---|---|---|
| deficit piksel | 3 243 | 189 246 | 1 099 083 |
| bileşen (toplam/çapalı/kopuk) | 1 / 1 / 0 | 12 / 12 / 0 | 5 / 5 / 0 |
| native alfa IoU | **0,4274** | 0,9989 | kapıya ulaşmıyor |
| evaluator alfa IoU | — | **0,9914** | — |
| öldüğü aşama | native alfa kapısı | evaluator | bayt bütçesi |

---

# GÖREV A — `public-05`: native alfa IoU 0,4274, kafesten bağımsız sabit

## Ölçülmüş kanıt

Painter, paint-deficit adayını 5 farklı alfa kafesinde üretti. Bayt bütçesi 327 120.

| kodlama | seviye | bayt | native alfa IoU | sonuç |
|---|---|---|---|---|
| `paint-deficit-cumulative` | 23 | 367 399 | — | bayt reddi |
| `paint-deficit-cumulative-q20` | 19 | 325 208 | 0,42740240 | native alfa reddi |
| `paint-deficit-cumulative-q16` | 15 | 287 060 | 0,42739385 | native alfa reddi |
| `paint-deficit-cumulative-q12` | 11 | 249 181 | 0,42738059 | native alfa reddi |
| `paint-deficit-cumulative-q8` | 7 | 211 350 | 0,42734977 | native alfa reddi |

Kapı: `native_iou_gate_failed: 0.4274 < 0.995`.

Ek sayaçlar: `source_component_count=1`, `anchored=1`, `detached=0`,
`paint_deficit_pixel_count=3243`.

## Elediğim hipotezler (tekrar önerme)

1. **Nicemleme/kafes çözünürlüğü değil.** Seviye sayısı 19'dan 7'ye düşerken IoU
   yalnızca 5×10⁻⁵ oynuyor. Kapsama kaybı olsaydı kafesle birlikte bozulurdu.
2. **Kopuk bileşen elemesi değil.** Tek kaynak bileşen var, o da çapalı; hiçbir şey
   elenmiyor (`detached=0`).
3. **Eksik boya değil.** Deficit dedektörü yalnızca 3 243 piksel "boya eksik" buluyor.
   Kaynağın %57'si üretilemiyor olsaydı bu sayı çok büyük olurdu.

## Ana hipotezim (senden bunu sınamanı istiyorum)

**Maske geometrik olarak hizasız.** Sabit örtüşme oranı, uzamsal kayma/ölçek hatasının
imzasıdır: maske doğru üretilip yanlış yere düşerse örtüşme kafes çözünürlüğünden
bağımsız sabit kalır — tam olarak gözlenen davranış.

Maske şu dönüşümü taşıyor (`engine/app/alpha_candidate_painter.py`,
`build_painter_reconstruction_tree`):

```python
content = ET.SubElement(mask, qname("g"))
content.set(
    "transform",
    f"translate({view_x:.12g} {view_y:.12g}) "
    f"scale({view_width / float(raster_width):.12g} "
    f"{view_height / float(raster_height):.12g})",
)
# Opak siyah taban:
ET.SubElement(content, qname("rect"), {
    "x": "0", "y": "0",
    "width": str(raster_width), "height": str(raster_height),
    "fill": "rgb(0,0,0)",
})
```

Maske elemanının kendisi:

```python
mask = ET.SubElement(defs, qname("mask"), {
    "id": mask_id,
    "maskUnits": "userSpaceOnUse",
    "x": f"{view_x:g}", "y": f"{view_y:g}",
    "width": f"{view_width:g}", "height": f"{view_height:g}",
})
```

Uygulama:

```python
layer = ET.SubElement(root, qname("g"), {...})
ET.SubElement(layer, qname("use"), {"href": f"#{paint_id}", "mask": f"url(#{mask_id})"})
```

`view_x, view_y, view_width, view_height` `_viewbox(root)`'tan; `raster_height,
raster_width = quantized.shape`.

## Dikkatini çekmek istediğim noktalar

- `mask` elemanında `x/y/width/height` **`:g`** formatıyla yazılıyor, içerideki
  `transform` ise **`:.12g`** ile. `:g` varsayılan olarak 6 anlamlı basamağa yuvarlar.
  ViewBox değerleri büyük veya kesirliyse maske kutusu ile içerik dönüşümü arasında
  tutarsızlık doğar mı?
- `maskUnits="userSpaceOnUse"` ile `x/y/width/height` kullanıcı uzayındadır. Ancak
  `<use>` elemanı köke eklenmiş bir `<g>` içindedir. Ata zincirinde bir `transform`
  varsa maske kutusu hangi uzayda değerlendirilir?
- Kaynak SVG'nin viewBox orijini sıfır değilse (`view_x`/`view_y` ≠ 0) veya viewBox
  en-boy oranı raster ızgarasınınkinden farklıysa ne olur?
- `_viewbox(root)` viewBox yoksa ne döndürüyor? (Bu depoda "viewBox yok" durumu için
  onarım geçmişi var — RFV-3E.)

## İstediğim çıktı

1. Yukarıdaki koddan hareketle hizasızlığın **tam mekanizması**: hangi girdi
   koşulunda (viewBox orijini/ölçeği/oranı) maske kayar veya yanlış ölçeklenir?
2. IoU ≈ 0,427 sayısıyla tutarlı bir açıklama. 0,427 kabaca hangi kayma/ölçek
   hatasına karşılık gelir? (Örn. tek eksende ~%57 kayma mı, iki eksende ölçek hatası mı?)
3. Somut yama önerisi.
4. **Ayırt edici deney:** hizasızlığı kapsamadan ayıracak, kod çalıştırmadan
   tarif edilebilir bir test. (İpucu: render edilen alfanın bbox/centroid'i ile kaynak
   alfanınkini karşılaştırmak; hizasızlıkta sabit kayma, kapsamada oran kaybı görülür.)
5. Hizasızlık **değilse** ne olabilir? En az bir alternatif hipotez ve onu ayıracak ölçüm.

---

# GÖREV B — `public-04`: native 0,9989 ama evaluator 0,9914 (sabit ~0,0075 boşluk)

## Ölçülmüş kanıt

| kodlama | seviye | bayt | native IoU | bounded IoU | **evaluator IoU** | **evaluator MAE** |
|---|---|---|---|---|---|---|
| `cumulative` | 23 | 316 289 | 0,99889 | 0,99886 | 0,99142 | 0,006740 |
| `…-q20` | 19 | 277 391 | 0,99888 | 0,99886 | 0,99149 | 0,006680 |
| `…-q16` | 15 | 239 498 | 0,99887 | 0,99886 | 0,99158 | 0,006609 |
| `…-q12` | 11 | 201 619 | 0,99885 | 0,99886 | 0,99172 | 0,006505 |
| `…-q8` | 7 | 163 793 | 0,99880 | 0,99886 | 0,99190 | 0,006361 |

Red kodu: `evaluator_rejected: alpha_iou_below_min, alpha_mae_above_max`

Sayaçlar: `source_component_count=12`, `anchored=12`, `detached=0`,
`paint_deficit_pixel_count=189246`.

## Anlamlı olan üç örüntü

1. **Native ve bounded IoU mükemmele yakın (0,9989) ama evaluator 0,9914.**
   Sabit ~0,0075'lik bir boşluk var. Aday, native ızgarada kaynak alfayı neredeyse
   birebir üretiyor; evaluator'ın ölçtüğü yerde bu sadakat kayboluyor.
2. **Kafes kabalaştıkça evaluator metriği İYİLEŞİYOR** (IoU 0,99142→0,99190,
   MAE 0,006740→0,006361) — sezgiye aykırı ama monoton ve tutarlı. Daha az alfa
   seviyesi evaluator'a göre daha iyi. Bu, boşluğun kaynağına dair güçlü bir ipucu.
3. **Bounded IoU kafesten bağımsız neredeyse sabit** (0,99886), native hafif düşüyor.

## Bilinen ölçüm bağlamı

Motorda üç ayrı alfa ölçümü var: `native_alpha_iou/mae` (kaynak-native ızgara),
`bounded_alpha_iou/mae` (sınırlı değerlendirme ızgarası) ve `evaluator_alpha_iou/mae`
(`FinalArtifactEvaluator`, tam çözünürlüklü kaynağa karşı).

`engine/app/alpha_candidate_painter.py` modül docstring'inden:

> Validation is fail-closed with unchanged thresholds: the native-grid render must
> reproduce the staged alpha plane, the INTER_AREA-downscaled native render must match
> the identically downscaled source on the bounded evaluation grid, FinalArtifactEvaluator's
> alpha-plane codes are enforced against the full-resolution source...

İlgili sabitler: `_PAINTER_EVAL_SIDE = 512.0`, `_PAINTER_GRID_MAX_SIDE = 1600`.

Eşikler `engine/app/final_artifact_evaluator.py` içinde `_thresholds(image_class, None)`
ile sınıf bazlı geliyor (`clean_logo`, `geometric`, `lineart`, `illustration`, `photo`).
Paylaşılan alfa eşikleri `shared_alpha` altında.

## İstediğim çıktı

1. **Neden kaba kafes evaluator'a göre daha iyi?** Bu ters ilişkinin mekanizması ne?
   (Düşün: çok sayıda ince alfa seviyesi → çok sayıda komşu bölge sınırı → tam
   çözünürlükte AA karışımı; az seviye → daha az sınır. Ama native ölçümde bu neden
   görünmüyor?)
2. **~0,0075'lik sabit boşluğun kaynağı.** Native ile evaluator arasında ne farklı:
   çözünürlük, yeniden örnekleme (INTER_AREA vs başka), kompozisyon zemini (beyaz
   üzerine mi, düz alfa mı), yoksa karşılaştırılan görüntünün kendisi mi?
3. Boşluğu kapatacak somut öneri. Dikkat: **eşik gevşetmek yasak.** Aday üretimi
   veya ölçüm hizalaması tarafında çözüm ara.
4. Ekstrapolasyon: mevcut eğilim basamak başına ~0,0002 IoU kazandırıyor, kapatılması
   gereken fark ~0,0035. Kafesi daha da kabalaştırmak (q4, q2) matematiksel olarak
   yeterli mi? Değilse neden?
5. Bu boşluğun **diğer vakalarda da** var olup olmadığını anlamak için ledger'dan
   hangi alanları karşılaştırmalıyım?

---

# GÖREV C — `public-15`: deficit örtüsü bayt bütçesini domine ediyor

## Ölçülmüş kanıt

Bayt bütçesi **304 990**. Tüm kümülatif basamaklar reddedildi:

| kodlama | seviye | bayt | bütçeye oran |
|---|---|---|---|
| `paint-deficit-q24` | 23 | 1 194 110 | 3,92× |
| `paint-deficit-cumulative` | 23 | 870 542 | 2,85× |
| `paint-deficit-cumulative-q20` | 19 | 831 463 | 2,73× |
| `paint-deficit-cumulative-q16` | 15 | 793 597 | 2,60× |
| `paint-deficit-cumulative-q12` | 11 | 755 710 | 2,48× |
| `paint-deficit-cumulative-q8` | 7 | 717 866 | **2,35×** |

Sayaçlar: `paint_deficit_pixel_count=1099083` (!), `source_component_count=5`,
`anchored=5`, `detached=0`.

## Kilit gözlem

Alfa kafesini 23 seviyeden 7 seviyeye indirmek baytı yalnızca **%17,5** düşürüyor
(870 542 → 717 866). Karşılaştırma için `public-05`'te aynı seyreltme baytı %42
düşürmüştü. Demek ki bu vakada baytın büyük kısmı **maskeden değil**, 1,1 milyon
piksellik deficit'ten üretilen **örtü geometrisinden** geliyor. Maskeyi küçülten
her yaklaşım burada yapısal olarak yetersiz.

## İlgili kod

`engine/app/alpha_candidate_paint_deficit.py` — deficit tespiti:

```python
def _paint_deficit_labels(source_rgba, artwork_rgba):
    source_white = _composite_on_white(source)
    artwork_white = _composite_on_white(artwork)
    source_foreground = np.any(np.abs(source_white.astype(np.int16) - 255) > 12, axis=2)
    artwork_missing = np.all(artwork_white > 244, axis=2)
    anchored_components, component_stats = _anchored_source_component_mask(
        source[:, :, 3], artwork[:, :, 3]
    )
    deficit = source_foreground & artwork_missing & anchored_components
    ...
    palette = _dominant_opaque_palette(source)          # en fazla 8 renk
    source_rgb = source[:, :, :3].astype(np.int32)
    distances = (source_rgb[:, :, None, :] - palette.astype(np.int32)[None, None, :, :]) ** 2
    nearest = np.argmin(distances.sum(axis=3), axis=2).astype(np.int32)
    labels = np.zeros(deficit.shape, dtype=np.int32)
    labels[deficit] = nearest[deficit] + 1
```

`_PALETTE_LIMIT = 8`, `_ALPHA_LEVELS = 24`.

Örtü, `labels` üzerinden palet rengi başına kontur/poligon geometrisi olarak yazılıyor
(`_painter_loops` / `_simplify_rectilinear_loop` kullanılıyor).

## İstediğim çıktı

1. **1,1 milyon deficit pikseli normal mi?** Bu, "sanat eseri kaynağın büyük kısmını
   kaçırıyor" anlamına mı geliyor, yoksa `artwork_missing` eşiği (`> 244`) bu vakada
   yanlış mı pozitif veriyor? Açık gri/pastel bir kaynakta ne olur?
2. **Örtü baytını düşürmenin yolları.** Alfa kafesinden bağımsız olarak örtü
   geometrisini küçültecek seçenekleri değerlendir ve sırala:
   - palet limitini vakaya göre daraltmak (8 → daha az renk = daha az bölge)
   - deficit maskesini morfolojik olarak sadeleştirmek (küçük adacıkları elemek)
   - kontur sadeleştirme toleransını artırmak
   - örtüyü `<path>` yerine düğüm saymayan bir kodlamayla yazmak
   Her biri için: bayt kazancı tahmini, kalite riski, hangi kapıyı tehlikeye atar.
3. **Bütçenin kendisi doğru mu?** Bayt bütçesi ebeveyn boyutundan türetiliyor. 1,1M
   deficit pikselli bir vakada 304 990 bayt gerçekçi bir hedef mi, yoksa bu vaka
   paint-deficit adayı için yapısal olarak uygun değil mi? Uygun değilse hangi aday
   türü uygundur?
4. Önerinin **hâlihazırda geçen vakaları bozmayacağını** nasıl garanti ederiz?

---

# GÖREV D — Düğüm eklemeyen `polygon` kodlamasına nicemleme taraması

## Bağlam

Painter turnuvasında dört maske kodlaması var. İkisi düğüm bütçesine katkı vermiyor
(`<polygon>` ve `<rect>` SVG'de `path_count`/`node_count` olarak sayılmıyor), ikisi
kompakt ama `<path>` yazdığı için sayılıyor:

| kodlama | sayılan düğüm | bayt (public-05) | dikiş |
|---|---|---|---|
| `polygon` | **0** | 1 201 085 | yok |
| `rect` | **0** | 725 597 | iç kenar dikişi var |
| `contour` | 15 862 – 17 323 | 71 926 – 81 820 | yok |
| `cumulative` | 14 305 – 45 817 | 99 365 – 180 601 | kapatır |

Düğüm eşiği: `parent_nodes + 2500` (public-05'te 3 018). Bayt bütçesi 327 120.

**Kusur:** `contour` ailesine q128/q64/q32 nicemleme taraması verilmiş, ama düğüm
eklemeyen `polygon`/`rect` ailelerine **hiç tarama yok** — yalnızca tam 127 seviyelik
kafeste deneniyorlar, o yüzden bayt bütçesini aşıyorlar. Sonuç: `<path>` tabanlılar
düğümden, `<polygon>`/`<rect>` tabanlılar bayttan eleniyor; hiçbiri iki kısıtı birden
sağlayamıyor.

Sentetik ölçümüm (yoğun alfa rampası, maske baytı):

```
polygon exact (127 sv)   976 077  (100,0%)
polygon-q64    (63 sv)   484 009  ( 49,6%)
polygon-q32    (31 sv)   239 622  ( 24,5%)
polygon-q16    (15 sv)   118 035  ( 12,1%)
```

public-05 için q32 oranı 1 201 085 → ~294 000 demek: **bütçeye sığar, sıfır düğüm
ekler, iç kenarı olmadığı için dikiş üretmez.** İki kısıtı birden sağlayabilecek
tek aday bu.

`rect` kasıtlı olarak kapsam dışı: iç kenar dikişi, onarılmaya çalışılan
`seam_regression`'ı besler.

## ⚠️ Bunu ben denedim ve GERİ ALDIM

Naif "faz sonuna ekle" yaklaşımı üç mevcut test sözleşmesini kırıyor:

**1.** `engine/test_alpha_painter_ledger.py:495`

```python
self.assertEqual(quantized_labels, {"contour-q128", "contour-q64", "contour-q32"})
# Hata: 'polygon-q64', 'polygon-q32', 'polygon-q16' fazladan geldi
```

**2.** `engine/test_alpha_painter_stroke_continuation.py:84`
— `test_noneligible_journal_rejection_stops_and_rolls_back`

**3.** `engine/test_alpha_painter_stroke_continuation.py:110`
— `test_mixed_reason_set_is_not_retry_eligible`

İkisi de şunu doğruluyor:

```python
self.assertTrue(all(entry["stroke_width"] == 1.5 for entry in validated))
```

**En kritiği (2):** "uygun olmayan journal reddinde painter **durmalı ve geri
almalı**". Yeni bir fazı koşulsuz olarak en sona eklemek bu "dur" semantiğini ihlal
ediyor. Bu testleri değiştirip geçmek **yanlış olur** — kasıtlı davranış sözleşmeleri.

## Mevcut faz düzeni

```python
count_preserving_specs = [
    ("polygon", "polygon", "exact", quantized, opacity_by_level),
    ("rect",    "rect",    "exact", quantized, opacity_by_level),
]
compact_specs = [("contour", "contour", "exact", quantized, opacity_by_level)]
quantized_specs = []
for target_levels in (128, 64, 32):
    requant, requant_opacity = _requantize_alpha(grid_alpha, target_levels)
    quantized_specs.append(
        (f"contour-q{target_levels}", "contour", "quantized", requant, requant_opacity)
    )

# ...
winner = _evaluate_phase(count_preserving_specs)
if winner is None:
    winner = _evaluate_phase(compact_specs)
if winner is None:
    winner = _evaluate_phase(quantized_specs)
if winner is None:
    winner = _evaluate_paint_deficit()
```

Ledger girdisindeki ilgili alanlar: `encoding_label`, `encoding_family`,
`exact_or_quantized` (`"exact"` | `"quantized"` | `"paint_deficit"`), `stroke_width`,
`validation_stage`, `status`.

## İstediğim çıktı

1. **Üç testi de değiştirmeden** polygon nicemleme taramasını ekleyen bir tasarım.
   İpuçları: yeni fazı `exact_or_quantized` alanında ayrı bir değerle işaretlemek
   (`"quantized"` etiketini kirletmemek); fazı yalnızca **retry-eligible** red
   yolunda çalıştırmak, uygun olmayan redde çalıştırmamak.
2. Somut yama (tam fonksiyon gövdesi veya diff).
3. Test (1)'in `quantized_labels` kümesini nasıl topladığını bilmiyorsan **sor** —
   tahmin etme; tasarım buna bağlı.
4. Yeni fazın hangi koşulda tetikleneceğinin net kuralı ve bunun (2)/(3) numaralı
   testlerin doğruladığı "dur ve geri al" semantiğiyle nasıl uyumlu olduğu.
5. Bu değişikliğin `public-15` gibi örtü-baytı domine eden vakalarda **işe
   yaramayacağını** teyit et (Görev C ile çakışmasın).
