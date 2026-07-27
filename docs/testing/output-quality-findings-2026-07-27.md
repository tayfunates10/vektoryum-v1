# Vektoryum çıktı kalite bulguları — 2026-07-27

## Ölçüm kapsamı

- Branch: `agent/output-quality-diagnostic-v1`
- Ölçülen head: `0c71c7bcb126e55f8f4a03f3ff1718ccb54cbdd0`
- Sentetik hedef vaka: 7
- Tekrar: vaka başına 2
- Tam production pipeline çalıştırması: 14
- Final-artifact değerlendirmesi: 14
- Byte-deterministic artifact: 7 / 7
- Structural failure: 0
- Severity: 3 high, 2 medium, 2 pass

Bu paket gerçek RFV corpusunun yerine geçmez. Amaç, belirli kusurları hızlı ve tekrarlanabilir biçimde yerelleştirmektir.

## Yönetici özeti

Test hattı yapısal olarak sağlam çalıştı: tüm vakalar SVG üretti, yeniden render edildi ve iki tekrar aynı seçilmiş SVG SHA-256 değerini verdi. Bununla birlikte çıktı kalitesinde üç farklı sorun sınıfı ayrıştırıldı:

1. **Gerçek production kaybı:** gri dış bant içeren üç tonlu logo `single_color` moda düşüyor ve gri bant tamamen kayboluyor.
2. **Gerçek fakat orta düzey kalite sapması:** küçük bileşenler ve alpha içeren renk geçişlerinde şekil/renk sınırı sapıyor.
3. **Evaluator hassasiyeti / olası yanlış pozitif:** geometri neredeyse birebir olduğu halde ton kanonikleştirmesi ve açık-gri bölgeler SSIM/seam kapısını gereğinden sert tetikliyor.

## Bulgular

### OQ-01 — Gri bantlı logo tek renge çöküyor

**Durum:** gerçek yüksek öncelikli production hatası.

- Vaka: `qa-gray-border-counter`
- Auto mode: `single_color`
- Analizör renk tahmini: 3
- Kazanan: `single_contour` / `opencv_contour`
- Seçim: `highest_total_score`
- Kazanan candidate fidelity: 21.88
- Final SSIM: 0.5639
- Edge F1 (1 px): 0.2864
- Delta E 2000 p95: 26.06
- Topology component delta: 21
- Seam ratio: 0.36455
- Hard kodlar: `seam_gap`, `ssim_below_min`, `topology_component_delta`

**Gözlenen sonuç:** gri dış bant tamamen kayboluyor; siyah bölge kaba çokgen forma dönüşüyor. İç sayaç/delik biçimi de belirgin bozuluyor.

**Kök neden:** analizör, üç ayrı nötr ton bulunmasına rağmen `single_color` öneriyor. Bu modun palet üst sınırı ve aday havuzu gri + siyah + beyaz yapıyı temsil edemiyor. Üstelik `single_contour`, daha sadık `single_clean` adayına göre daha düşük fidelity üretmesine rağmen structural `total_score` ile kazanıyor.

**Önerilen düzeltme:**

- `estimated_color_count >= 3` ve ayrık nötr bantlar varsa `single_color` önerisini engelle.
- Bu sınıfı `geometric_logo` veya renk koruyan bir logo moduna yönlendir.
- `single_color` seçiminde minimum fidelity tabanı koy; en yüksek `total_score`, açıkça daha düşük fidelity adayı seçemesin.
- Bu vaka için kalıcı regresyon testi ekle: gri bandın alanı ve palet rengi korunmalı.

### OQ-02 — Düşük çözünürlüklü rozette evaluator seam/SSIM hassasiyeti

**Durum:** evaluator yanlış-pozitif adayı + hafif küçük bileşen sapması.

- Vaka: `qa-lowres-badge`
- Auto mode: `geometric_logo`
- Kazanan: `geo_clean` / `vtracer`
- Candidate fidelity: 62.59
- Edge F1: 1.0000
- Component delta: 0
- Hole delta: 0
- Delta E p95: 5.37
- Seam ratio: 0.20779
- SSIM: 0.7449
- Hard kodlar: `seam_gap`, `ssim_below_min`

**Gözlenen sonuç:** dış halka ve kırmızı artı geometrik olarak korunuyor. Kaynaktaki açık gri iç alan beyaza kanonikleşiyor; evaluator bunu geniş seam alanı gibi sayıyor. Minimum component IoU 0.0 ayrıca en küçük sınıflandırılmış bileşenin renk/alan eşleşmesinde sorun olduğunu gösteriyor.

**Kök neden adayı:** seam maskesi, beyaza yakın fakat kaynakta foreground kabul edilen bölgeleri çıktıdaki beyaz zeminle karşılaştırırken eksik bölge olarak sayıyor. SSIM de piksel-art ton farkına geometri metriklerinden çok daha sert tepki veriyor.

**Önerilen düzeltme:**

- Seam hesabında beyaza yakın bölgeler için algısal Delta E veya adaptif foreground eşiği kullan.
- SSIM hard fail kararını edge F1, topology ve Delta E ile birlikte değerlendir.
- Low-resolution sınıfında min-component eşleştirmesinin renk kimliği değişimine karşı davranışını ayrıca test et.

### OQ-03 — Delikli şekiller geometriyi koruyor fakat SSIM hard-fail veriyor

**Durum:** yüksek güvenli evaluator yanlış-pozitif adayı.

- Vaka: `qa-ring-holes`
- Auto mode: `single_color`
- Kazanan: `single_clean` / `vtracer`
- Edge F1: 1.0000
- Chamfer p95: 0.0
- Hausdorff max: 0.0
- Component delta: 0
- Hole delta: 0
- Min / mean component IoU: 1.0 / 1.0
- Palette agreement: 1.0
- Delta E p95: 3.22
- SSIM: 0.7623
- Tek hard kod: `ssim_below_min`

**Gözlenen sonuç:** üç dış şekil ve üç iç delik doğru korunuyor. Kaynak koyu gri, çıktı kanonik siyah olduğu için tüm dolgu alanı ton farkı taşıyor.

**Kök neden:** geometri/topoloji kusursuz olmasına rağmen `clean_logo` SSIM eşiği ton kanonikleştirmesini ağır yapısal hata gibi değerlendiriyor.

**Önerilen düzeltme:**

- Tek renkli kanonik modlarda SSIM'i renk-normalize veya maske-temelli yardımcı metrikle destekle.
- Edge F1 = 1, hole/component delta = 0 ve component IoU = 1 olduğunda yalnız SSIM nedeniyle hard fail üretme; renk sapmasını ayrı soft/hard renk kapısına bırak.

### OQ-04 — Küçük bileşenlerde şekil sapması

**Durum:** gerçek orta öncelikli detay koruma sorunu.

- Vaka: `qa-small-details`
- Auto mode: `geometric_logo`
- Kazanan: `geo_standard` / `vtracer`
- Seçim: `near_score_geometric_preference`
- Winner fidelity: 95.99
- En yüksek candidate: `geo_clean`, fidelity 96.49
- SSIM: 0.9883
- Edge F1: 1.0000
- Min component IoU: 0.8571
- Soft kodlar: `component_iou_below_min`, `worst_face_de00`

**Gözlenen sonuç:** ana çerçeve korunuyor; küçük noktalar ve ince kırmızı çizgilerde yarıçap/konum farkları oluşuyor.

**Kök neden:** yakın-skor geometrik tercih, en yüksek fidelity aday yerine daha düzenli `geo_standard` adayını seçiyor. Genel skor yüksek olsa da en küçük bileşen hedefi aşamıyor.

**Önerilen düzeltme:**

- Küçük bileşen içeren görüntülerde seçim tie-breaker'ına `min_component_iou` veya küçük-parça recall ekle.
- `near_score_geometric_preference` sadece küçük detay kaybı belirli toleransın altındaysa çalışsın.

### OQ-05 — Alpha doğru, görünür renk sınırı yanlış

**Durum:** gerçek orta öncelikli alpha/gradient kalite sorunu.

- Vaka: `qa-transparent-overlap`
- Auto mode: `logo_color`
- Kazanan: `logo_gradient` / `gradient`
- Seçim: `highest_fidelity+source_alpha_vector_mask`
- Alpha IoU: 0.99767
- Alpha MAE: 0.00078
- Edge F1: 0.9169
- Delta E p95: 10.14
- SSIM: 0.9875
- Soft kodlar: `color_de00_p95`, `edge_f1_below_min`

**Gözlenen sonuç:** alpha maskesi ve dış silüet güçlü biçimde korunuyor; ancak mavi-kırmızı birleşiminde gri/cyan geçiş bandı oluşuyor ve görünür sınır kaynakla uyuşmuyor.

**Kök neden:** gradient adayının renk geçişi, sert overlap sınırını sürekli gradient olarak modelliyor. Alpha düzlemi iyi olduğu için sorun maskede değil, renk/compositing modelinde.

**Önerilen düzeltme:**

- Alpha-overlap bölgelerinde gradient ve katmanlı düz-dolgu adaylarını ayrı karşılaştır.
- Sert renk sınırı sinyali varsa gradient geçişini sınırlandır veya piecewise gradient üret.
- Candidate seçiminde alpha IoU yanında visible-composite edge F1 ve Delta E p95 tabanı uygula.

## Başarılı vakalar

### `qa-monoline`

- Mode: `lineart`
- Winner: `lineart_detail`
- SSIM: 0.99977
- Edge F1: 1.0000
- Min component IoU: 0.99954
- Hard/soft bulgu: yok

### `qa-shared-boundary`

- Mode: `logo_color`
- Winner: `logo_standard`
- SSIM: 0.99856
- Edge F1: 1.0000
- Min component IoU: 0.99406
- Seam ratio: 0.0
- Hard/soft bulgu: yok

Bu sonuç shared-boundary testinin mevcut düz renkli örnekte crack/gap üretmediğini doğrular.

## Düzeltme önceliği

1. **P0 — OQ-01:** auto mode üç tonlu nötr logoyu `single_color` moda düşürmemeli.
2. **P1 — OQ-05:** alpha-overlap gradient seçimi görünür renk sınırını korumalı.
3. **P1 — OQ-02/OQ-03:** evaluator seam ve SSIM kapıları geometriyi doğru çıktı için yanlış hard-fail üretmemeli.
4. **P2 — OQ-04:** küçük bileşen koruması candidate seçim tie-breaker'ına bağlanmalı.

## Release etkisi

Bu test çalışması production kodunu veya release kararını değiştirmez. Mevcut `RFV-3: pending` ve `release_decision: NO-GO` durumu korunur. Bulgular düzeltilip aynı vakalar iki tekrar `hard` kapıda geçtiğinde bu sentetik teşhis paketinin kalite borcu kapanmış sayılabilir; gerçek RFV kabulü ayrıca tamamlanmalıdır.
