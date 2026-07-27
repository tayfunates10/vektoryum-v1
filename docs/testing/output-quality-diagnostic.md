# Vektoryum Output Quality Diagnostic v1

## Amaç

Bu test hattı, Vektoryum'un değişmemiş production pipeline çıktısını hedefli sentetik vakalarla ölçer. Production winner seçimini, evaluator eşiklerini, serializer'ı, RFV corpusunu veya release kararını değiştirmez.

Yeni hattın görevi hata tespitini hızlandırmaktır:

- gri kenarlık içi kalıntı
- ortak sınır seam/gap
- sayaç ve delik kaybı
- monoline kopması
- küçük detay kaybı
- düşük çözünürlüklü rozet bozulması
- transparan/yarı transparan çakışma
- eksik, geçersiz, render edilemeyen veya nondeterministic SVG

Sentetik rasterlar çalışma anında üretilir. Repository'ye binary fixture eklenmez.

## Üretilen kanıt

Her vaka dizini mümkün olduğunda şunları içerir:

- `source.png`: deterministik kaynak
- `selected.svg`: production pipeline tarafından seçilen gerçek SVG
- `render.png`: seçilen SVG'nin yeniden rasterize edilmiş hali
- `difference.png`: kaynak ile render arasındaki piksel farkı ısı haritası

Kök dizin ayrıca şunları içerir:

- `output-quality-report.json`
- `output-quality-summary.md`

JSON raporu SHA-256, pipeline durumu, artifact determinism, evaluator verdict, hata kodları ve aşağıdaki metrikleri taşır:

- SSIM ve MS-SSIM
- edge F1 1 px / 2 px
- Chamfer p95 ve Hausdorff max
- Delta E 2000 ortalama / p95
- palette agreement
- component ve hole delta
- minimum / ortalama component IoU
- seam ratio
- alpha IoU / MAE
- path, node ve SVG byte sayısı
- median render süresi

## Severity

- `critical`: çıktı yok, parse/render güvenliği bozuk veya artifact nondeterministic
- `high`: `FinalArtifactEvaluator` hard kalite hatası
- `medium`: soft kalite uyarısı
- `low`: gerekli bir ölçüm tamamlanamadı
- `pass`: hard/soft/unmeasured bulgu yok

## Gate modları

- `none`: yalnız rapor üretir
- `structural`: eksik, tehlikeli, render edilemeyen veya nondeterministic çıktı varsa başarısız olur
- `hard`: structural hatalara ek olarak evaluator hard kalite hatalarında da başarısız olur

Pull request koşumu bir tekrar ve `structural` kapı kullanır. Manuel ve haftalık koşum iki tekrar ve `hard` kapı kullanır. Böylece PR hattı gerçek production kırılmalarını engellerken haftalık hat kalite eşiklerini de fail-closed uygular.

## Yerel kullanım

```bash
PYTHONPATH=.:engine python -m engine.regression.output_quality_suite \
  --output /tmp/vektoryum-output-quality \
  --engine-version "$(git rev-parse HEAD)" \
  --repeat-count 2 \
  --fail-on hard
```

Sözleşme testleri:

```bash
PYTHONPATH=.:engine python -m unittest \
  engine.regression.test_output_quality_suite
```

## RFV ile ilişki

Bu hat RFV-3B'nin 24 gerçek vaka × 3 tekrar kabul ölçümünün yerine geçmez. Hızlı ve görsel hata-yerelleştirmeli bir ön teşhis katmanıdır. Gerçek corpus, release kararı ve kalite eşikleri mevcut RFV sözleşmelerinde otorite olmaya devam eder.
