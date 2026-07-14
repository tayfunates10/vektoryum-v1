# AI Analyzer Closure Roadmap

Bu roadmap görsel analizi, otomatik sınıflandırma, güven metadata'sı ve `auto` mod kararının kapanışını kapsar. Vektör üretimi ve final artifact kalitesi Core Vector Engine roadmap'inde; SaaS yönetim işleri ayrı ürün roadmap'lerinde izlenir.

Makinece doğrulanan kaynak: `engine/ai_analyzer_roadmap.json`.

## Mod kapsamı

Analyzer şu altı modu otomatik önerebilir: `geometric_logo`, `minimal_ai`, `logo_color`, `single_color`, `lineart`, `photo_poster`.

`flat_logo` ve `centerline` explicit-only kalır.

## AA-1 — Finite analyzer capability and roadmap contract

Durum: **complete**

- Dört fazlı ve makinece doğrulanan roadmap oluşturuldu.
- Public modlar otomatik ve explicit-only kümelerine ayrıldı.
- Analyzer public rapor alanları ve mevcut seed kararları testlerle sabitlendi.
- Production analyzer davranışı değiştirilmedi.

## AA-2 — Versioned deterministic features and calibrated confidence

Durum: **complete**

- Feature adları, türleri, birimleri, aralıkları ve `analyzer-features-v1` sürümü tanımlandı.
- Decoded RGBA pikseli, feature snapshot'ı ve recommendation raporu ayrı SHA-256 digestleriyle bağlandı.
- Altı otomatik mod için 0..1 aralığında destek skorları ve ikinci seçenek marjı üretildi.
- Confidence, repoya eklenen etiketli sentetik feature örneklerinden margin-bin yöntemiyle hesaplanıyor.
- HED yoksa durum `unavailable` olarak raporlanıyor ve confidence artırılmıyor.
- Eksik, sonlu olmayan, aralık dışı veya yarım opsiyonel feature verisi geçersiz metadata raporu üretiyor.
- Mevcut `recommended_mode` ve sınıflandırma eşikleri değiştirilmedi.

## AA-3 — Review-aware auto-mode decision gate

Durum: **complete**

- Düşük güven veya çelişkili sinyal açık `needs_review` sonucu üretiyor.
- `auto` yalnız kaynak, feature ve recommendation digestleri eşleşen analyzer raporunu kullanıyor.
- Geçerli fakat belirsiz raporda analyzer önerisi korunuyor ve sonuç review durumunda kalıyor.
- Geçersiz veya eski rapor renk-koruyan fallback moduna geçiyor.
- Manual explicit modlar confidence mantığı tarafından değiştirilmiyor.
- Final artifact, review gereken bir auto kararından sonra production-ready olarak raporlanamıyor.

## AA-4 — Labeled analyzer corpus and release closure

Durum: **complete**

- Altı otomatik modun her biri için `in_domain` ve `boundary` image-level corpus vakası oluşturuluyor.
- Her vaka ayrı süreçte tam üç kez çalıştırılıyor.
- HED açıkça kapalı `no_hed` ortamı release sözleşmesinin parçasıdır.
- Kaynak pikseli, feature, recommendation, mod kararı, confidence ve review sonucu üç tekrarda birebir aynı olmalıdır.
- Confusion, mod bazlı accepted precision, doğru kabul kapsamı, Brier skoru ve expected calibration error raporlanır.
- Kabul edilmiş yanlış mod, geçersiz analyzer contract veya determinism sapması için tolerans sıfırdır.
- `AI analyzer release contract` workflow'u unit sözleşmelerini ve gerçek corpus runner'ını çalıştırır; JSON rapor ve fixture'ları artifact olarak saklar.
- Production sınıflandırma eşikleri veya manuel mod davranışı AA-4 kapsamında değiştirilmez.

## Zorunlu kapanış kontrolleri

AA-4 PR'ı yalnız aşağıdaki kontroller tamamen yeşilken sabit head SHA ile merge edilir:

- `AI analyzer release contract`
- `Exact final SVG contract`
- `Benchmark v1 seed corpus`
- `Core all-mode release contract`
- `Core centerline graph contract`

## Yüzde hesabı

Toplam dört merge-kapılı faz vardır. AA-4 yeşil CI ile `main` dalına merge edildiğinde AI Analyzer roadmap ilerlemesi `4/4`, yani `%100` olur. Bu noktadan sonra bu roadmap için yeni faz veya davranış değişikliği açılmaz.
