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

Durum: **complete in this PR**

- Feature adları, türleri, birimleri, aralıkları ve `analyzer-features-v1` sürümü tanımlandı.
- Decoded RGBA pikseli, feature snapshot'ı ve recommendation raporu ayrı SHA-256 digestleriyle bağlandı.
- Altı otomatik mod için 0..1 aralığında destek skorları ve ikinci seçenek marjı üretildi.
- Confidence, repoya eklenen 18 etiketli sentetik feature örneğinden margin-bin yöntemiyle hesaplanıyor.
- HED yoksa durum `unavailable` olarak raporlanıyor ve confidence en fazla `0.85` olabiliyor.
- Eksik, sonlu olmayan, aralık dışı veya yarım opsiyonel feature verisi geçersiz metadata raporu üretiyor.
- Mevcut `recommended_mode` ve sınıflandırma eşikleri değiştirilmedi.

## AA-3 — Review-aware auto-mode decision gate

Durum: **pending**

- Düşük güven veya çelişkili sinyal açık `needs_review` sonucu üretecek.
- `auto` yalnız kaynak ve feature digestleri eşleşen analyzer raporunu kullanacak.
- Manual explicit modlar confidence mantığı tarafından değiştirilmeyecek.
- Sınır vakaları için deterministik review testleri eklenecek.

## AA-4 — Labeled analyzer corpus and release closure

Durum: **pending**

- Her otomatik mod için etiketli in-domain ve sınır corpus'u oluşturulacak.
- Üç tekrarlı sonuçlar deterministik olacak.
- Confusion, calibration ve review metrikleri raporlanacak.
- Mod bazlı precision ve classification-error eşikleri uygulanacak.
- Analyzer contract, corpus ve benchmark workflow'ları zorunlu olacak.

## Yüzde hesabı

Toplam dört merge-kapılı faz vardır. Her faz yalnız kendi PR'ı tüm zorunlu CI kontrolleri yeşilken sabit head SHA ile `main` dalına merge edildiğinde tamamlanmış sayılır. AA-2 merge edildiğinde analyzer roadmap ilerlemesi `2/4`, yani `%50` olacaktır.
