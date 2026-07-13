# Vektoryum Benchmark v1

Bu klasör ölçülebilir, tekrar üretilebilir ve lisans kaynağı izlenebilir benchmark altyapısını barındırır.

## Kategoriler

`logos`, `seals`, `technical`, `signatures`, `gradients`, `low_resolution`, `transparent`, `multilingual`.

## Zorunlu manifest alanları

Her örnek benzersiz `case_id`, kategori, repository içindeki göreli kaynak yolu, lisans kimliği, kaynak SHA-256 değeri ve isteğe bağlı etiketler içerir. Lisansı veya kaynağı doğrulanamayan görsel benchmark korpusuna alınmaz.

## Zorunlu sonuç metrikleri

- fidelity
- SSIM
- edge F1
- alpha IoU
- Delta E00
- SVG path sayısı
- SVG dosya boyutu
- render süresi
- tepe RSS bellek

Ölçülemeyen metrik `null` olabilir ancak anahtar raporda mutlaka bulunur. Böylece eksik ölçüm sessizce başarı olarak kabul edilmez.

## CI katmanları

1. **PR contract:** Manifest ve rapor şemasını hızlı şekilde doğrular; binary veri setini çalıştırmaz.
2. **Scheduled benchmark:** Geniş korpusu ayrı workflow ile çalıştırır, JSON/HTML artifact üretir ve baseline karşılaştırması yapar.
3. **Release gate:** Yeterli veri toplandıktan sonra tanımlanacak kalite ve performans bütçelerini zorunlu hale getirir.

## Veri politikası

- Yalnız açık lisanslı, izinli veya proje tarafından üretilmiş sentetik girdiler kullanılmalıdır.
- Her dosyanın lisansı ve SHA-256 değeri manifestte yer almalıdır.
- Müşteri dosyaları açık izin olmadan benchmark korpusuna eklenmez.
- Büyük binary korpus Git LFS/Hugging Face Dataset veya ayrı object storage üzerinde tutulmalı; ana Git geçmişine doğrudan eklenmemelidir.
