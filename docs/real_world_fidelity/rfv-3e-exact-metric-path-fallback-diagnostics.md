# RFV-3E — Exact metric path fallback diagnostics

## Amaç

Bu faz yalnız PR #100'ün başarılı RFV-3B canlı ölçüm artifact'ındaki RFV-3D2 provenance kayıtlarını inceler. Production pipeline, winner selection, serializer, final-artifact evaluator, kalite eşikleri, corpus, repeat politikası ve release kararı değiştirilmez.

İncelenen kapsam:

- `qualification-public-10`
- `qualification-public-14`
- `qualification-public-18`

Kaynak kimliği:

- source PR: `#100`
- source head SHA: `0c09bcbad152c2661673e214dd159183e50a6525`
- workflow: `Real-world fidelity RFV-3B live production measurement`
- workflow run ID: `29555984755`
- run attempt: `1`
- artifact ID: `8399777964`
- artifact name: `rfv3-live-measurement-0c09bcbad152c2661673e214dd159183e50a6525`
- artifact digest: `sha256:d747ce3d8b1eb5e403bea39ba2607a7b75d3cb8cff2f77f26a1b528d7a7dd037`
- pipeline results SHA-256: `6e6c8f1458a1725153f7187fb3eefeef645b9565c56df4aed9d4c6fc35b7631a`
- retry audit SHA-256: `c7eec85ced0f5504ec0bf52aa29961c943df436bd922814ba3f9f328cc6c347f`
- measurement envelope SHA-256: `abaa39ec421a980b711c8b54b49b4768a28a55a966e4ea05d71480b6a051a988`

## Kanıtlanan gözlem

PR #100 remediation planındaki “winner SVG exact evaluator'a ulaşmıyor” varsayımı canlı provenance tarafından doğrulanmamıştır. Üç vakanın tamamında:

- selected SVG path mevcuttur;
- selected SVG dosyası mevcuttur;
- selected SVG SHA-256, result artifact SHA-256 ile aynıdır;
- exact evaluator denenmiştir;
- exact evaluator tamamlanmamıştır;
- kaydedilen failure class `exact_metrics_incomplete` değeridir;
- SSIM, edge F1 ve delta-E00 exact component değerleri non-finite/null kalmıştır;
- `partial_quality_report` fallback'i açıkça kaydedilmiştir.

| Case | Repeat audit | Path | File | Exact attempted | Exact completed | Failure class | Fallback | Winner SHA |
|---|---:|---:|---:|---:|---:|---|---|---|
| qualification-public-10 | 3/3 success, retry yok | var | var | evet | hayır | `exact_metrics_incomplete` | `partial_quality_report` | `c617bbf0…e1e74` |
| qualification-public-14 | 3/3 success, retry yok | var | var | evet | hayır | `exact_metrics_incomplete` | `partial_quality_report` | `f2ace52e…620a` |
| qualification-public-18 | 3/3 success, retry yok | var | var | evet | hayır | `exact_metrics_incomplete` | `partial_quality_report` | `621759af…e43a` |

## Neden kök neden hâlâ unresolved

Yayınlanan aggregate artifact, vaka başına provenance ve repeat başarı audit'ini içerir; ancak aşağıdakileri içermez:

- repeat başına evaluator provenance;
- evaluator report status;
- `hard_fail_codes`;
- hangi evaluator metric grubunun oluşmadığı;
- yapı kontrolü, render veya metric extraction aşamasını ayıran reason code.

Bu nedenle `exact_metrics_incomplete` kanıtlanmış bir **sonuç sınıfıdır**, fakat bu sonucun production sebebi değildir. Routing değişikliği yapmak mevcut kanıta aykırıdır. Final-artifact evaluator davranışını değiştirmek de hangi evaluator aşamasının eksik olduğu kanıtlanmadan yetkilendirilemez.

Kanonik diagnostic sonucu:

- root cause status: `unresolved`
- original routing hypothesis: `disproven`
- production fix allowed: `false`
- next branch: `agent/rfv-3e-exact-metric-path-provenance-completion`

Bu faz production düzeltmesini yetkilendirmez.

## Sonraki dar kapsam

Bir sonraki faz yalnız provenance tamamlamalıdır. Her repeat için aşağıdakiler sanitize edilerek publish edilmelidir:

- evaluator report status;
- hard/soft failure code listesi;
- B_visual, C_color ve D_edge_geometry grup varlığı;
- render sonucu;
- exact metric extraction reason code;
- artifact SHA bağları.

Bu veriler olmadan routing, evaluator, winner selection veya serializer değişikliği yapılmamalıdır.

## Değişmeyen kararlar

- RFV-3: `pending / remediation`
- release decision: `NO-GO`
- `rfv4_allowed`: `false`
- RFV-4 roadmap: `pending`
- thresholds: fidelity `99.0`, SSIM `0.98`, edge F1 `0.98`, alpha IoU `0.98`
- corpus identity: `5f151a6cb1a433b0cb0989a67bd7cc7940162f4b36d67903d6ccdd173f9e7d89`

Bu kanıt yalnız belirtilen source generation ve üç vaka için geçerlidir. Evrensel `%99 fidelity` iddiası yapılmamıştır.
