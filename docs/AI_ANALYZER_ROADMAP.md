# AI Analyzer Closure Roadmap

Bu roadmap yalnız görsel analizi, otomatik sınıflandırma, güven skoru ve `auto` mod kararının kapanışını kapsar. Vektör aday üretimi, topoloji ve final artifact kalitesi tamamlanan Core Vector Engine roadmap'inde; SaaS yönetim işleri ayrı ürün roadmap'lerinde izlenir.

Makinece doğrulanan kaynak: `engine/ai_analyzer_roadmap.json`.

## Mod kapsamı

Public trace modları dokuzdur. `auto`, analyzer kararını kullanır. Analyzer şu altı modu otomatik önerebilir: `geometric_logo`, `minimal_ai`, `logo_color`, `single_color`, `lineart`, `photo_poster`.

`flat_logo` ve `centerline` explicit-only kalır ve otomatik öneri kümesine sessizce eklenemez.

## AA-1 — Finite analyzer capability and roadmap contract

Durum: **complete in this PR**

- Dört fazlı, sonlu ve makinece doğrulanan roadmap oluşturulur.
- Public modlar otomatik önerilen ve explicit-only kümelerine ayrılır.
- Analyzer public rapor alanları sabitlenir.
- Mevcut geometrik ve çok-renkli seed kararları deterministik regresyon olarak kilitlenir.
- Bilinen her açık tam bir kapanış fazına bağlanır.
- Production analyzer davranışı değişmez.

## AA-2 — Versioned deterministic features and calibrated confidence

Durum: **pending**

- Özellik adları, birimleri, aralıkları ve extractor sürümü raporlanır.
- Aynı decoded pixel girdisi aynı feature ve recommendation digest'ini üretir.
- Öneriyle birlikte etiketli kanıta dayalı confidence ve runner-up margin döner.
- Opsiyonel HED sinyalinin yokluğu açıkça raporlanır ve confidence yükseltemez.
- Eksik, non-finite veya drift etmiş feature raporu fail-closed reddedilir.

## AA-3 — Fail-closed abstention and auto-mode decision gate

Durum: **pending**

- Düşük güven veya çelişkili sinyal yanlış kesin öneri yerine açık `needs_review` abstention üretir.
- `auto` yalnız kaynak ve feature digest'i eşleşen doğrulanmış analyzer kararını tüketir.
- Manual explicit modlar analyzer tarafından yeniden yazılmaz.
- Sınır vakaları ve stale veya missing report durumları regresyonlarla kilitlenir.

## AA-4 — Labeled analyzer corpus and release closure

Durum: **pending**

- Her otomatik öneri modu için in-domain ve sınır vakalı etiketli corpus bulunur.
- Üç tekrarlı sonuçlar deterministik olur.
- Confusion, calibration ve abstention metrikleri raporlanır.
- Mod-bazlı precision ve tehlikeli yanlış-pozitif eşikleri fail-closed uygulanır.
- Analyzer exact-contract, corpus ve benchmark workflow'ları zorunlu ve yeşil olur.

## Yüzde hesabı

Toplam dört merge-kapılı faz vardır. Her faz yalnız kendi PR'ı tüm zorunlu CI kontrolleri yeşilken sabit head SHA ile `main` dalına merge edildiğinde tamamlanmış sayılır. Her faz toplam analyzer roadmap'inin `%25`'idir.
