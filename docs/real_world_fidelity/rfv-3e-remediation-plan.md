# RFV-3E — Kalan gerçek-dünya fidelity hatalarının sınıflandırılması

RFV-3 canlı ölçümü (24 vaka × 3 tekrar, head `4abb684`) kanonik kararı
`release_decision: no_go`, `rfv4_allowed: false` olarak üretti. Bu faz üretim
kodunu değiştirmez: yalnız kalan kalite hatalarını gerçek ölçümlere göre
kök-neden kümelerine ayırır ve her küme için ölçüm-kapılı küçük remediation
fazlarını tanımlar. Karar, eşikler ve corpus kimliği değişmeden korunur.

Makine-okunur plan: `evidence/rfv3e_remediation_plan.json`
(`vektoryum-rfv3e-remediation-plan-v1`). Plan, kaynak generation'a SHA ile
bağlıdır (`source_results_sha256`, `source_decision_sha256`,
`source_cases_sha256`, `source_measurement_head_sha`) ve
`engine/regression/rfv3e_remediation_plan.py` doğrulayıcısı ile fail-closed
doğrulanır.

## Ölçüm özeti (24-case qualified corpus, ölçülen generation için)

- fidelity: 24/24 ihlal (min 21.34, medyan 81.0, maks 92.47; eşik 99.0)
- alpha_iou: 21/24 ihlal (eşik 0.98; yalnız `-01`, `-10`, `-13` geçiyor)
- edge_f1: 13 ihlal + 3 eksik (eşik 0.98)
- ssim: 13 ihlal + 3 eksik (eşik 0.98)
- `-10`, `-14`, `-18`: exact metrik yolu çalışmadı (RFV-3D1 kanıtlı
  `partial_quality_report_fallback`); eksik metrikler null, fail-closed.

## Kümeler (öncelik sırasıyla)

Her failing vaka **tam bir** kümeye atanmıştır; atama yalnız ölçülen baskın
sinyale göredir. `failed_metrics` alanları elle yazılmaz — kümedeki vakaların
gerçek eşik ihlallerinin birleşiminden hesaplanır ve CI'da yeniden doğrulanır.

1. **exact-metric-path-fallback** (proven, 3 vaka: `-10`, `-14`, `-18`) —
   ölçüm/metrik tamlığı. Winner SVG exact evaluator'a ulaşmıyor; RFV-3D2
   provenance enstrümantasyonu canlıda sınıfı kaydediyor. Kapsam: yalnız
   ölçüm-yolu routing düzeltmesi, ardından tam 24-vaka canlı rerun.
2. **transparent-logo-alpha-boundary** (strong, 6 vaka: `-02`, `-04`, `-05`,
   `-08`, `-15`, `-16`) — alpha IoU 0.78–0.96'da sıkışık kümelenme; SSIM ≥
   0.979 ve edge F1 ≥ 0.974 iken. Sistematik şeffaf-sınır işleme sapması.
3. **line-art-edge-structure** (strong, 6 vaka: `-03`, `-06`, `-07`, `-09`,
   `-13`, `-19`) — edge F1 0.687–0.944 ve/veya SSIM 0.903–0.974; monoline en
   kötü (0.687), small-text `-07` ayrıca delta_e00 11.56.
4. **photographic-content-representation** (strong, 5 vaka: `-20`…`-24`) —
   tüm JPEG fotoğraf vakalarında SSIM 0.333–0.496, delta_e00 13.6–15.8;
   fotoğrafik içerik sınıf olarak yetersiz temsil ediliyor.
5. **multicolor-illustration-alpha-collapse** (strong, 3 vaka: `-11`, `-12`,
   `-17`) — alpha IoU 0.24–0.42 çöküşü + path patlaması (858–142058 path,
   11 MB SVG).
6. **fidelity-composite-only** (tentative, 1 vaka: `-01`) — tüm bileşen
   eşikleri geçiyor, kompozit fidelity 85.06. **Yalnız tanı**: kanıt üretmeden
   üretim değişikliği yasak.

## Faz kuralları

- Her küme için ayrı fazlar: `agent/rfv-3e-<cluster-id>-diagnostics` →
  `-fix` → `-rerun`; her faz ayrı PR ve yeşil CI ile squash merge.
- "Tentative" kök neden için üretim düzeltme PR'ı açılmaz; önce
  enstrümantasyon/minimal reproduction.
- Düzeltmeler yalnız kanıtlanan modüllere dokunur; önce etkilenen vakalar,
  sonra regression sentinel'leri, sonra tam 24-vaka canlı rerun ölçülür.
- Tam 24-vaka rerun olmadan release kararı değiştirilmez.

## Yasaklar (A4)

Eşik düşürme, tolerans artırma, source resize ile kolaylaştırma, alpha'yı
beyaza flatten edip karşılaştırma, vaka atlama/çıkarma, metriği optional
yapma, path-count gözlemlenebilirliğini kaldırma, SVG içine raster/base64
gömme, fidelity'yi diğer metriklerden yapay üretme, winner seçimini yalnız
benchmark'a özel değiştirme — hiçbiri yapılmaz.

## Durum

- RFV-3: remediation sürüyor (incomplete); RFV-4: pending.
- Empirik release kararı: **NO-GO** (ölçülen generation için).
- Evrensel %99 fidelity iddiası yapılmadı; tüm sayılar 24-case qualified
  corpus üzerindeki ölçülen generation içindir.
