# Oturum devir brifingi — Vektoryum RFV-3B / motor kalite hattı

Bu belge, tükenmiş bir oturumdan devralacak yeni oturum için yazıldı. Amacı,
aynı yolları tekrar yürümeni önlemek: nelerin **ölçüldüğünü**, nelerin
**çürütüldüğünü** ve sıradaki iki somut deneyi içerir.

---

## 1. Depo durumu (bu belge yazıldığında)

```
main                                          8ef163c   #168 merge edildi (korpus geri dönüşü)
 └── #158  claude/yoneticilik-sorunlari-...   7dfab89   main'e rebase edildi, main atası
      ├── #164  codex/task-c-paint-deficit-byte-fallback        public-15  (Görev C)
      ├── #166  codex/public05-alpha-support-fallback           public-05  (Görev A)
      └── #170  codex/task-b-public04-evaluator-aligned-rfv3b   public-04  (Görev B)
```

Üçü de yeni tabana taşındı ve sözleşme paketini geçiyor. **Hiçbiri canlı
korpusa karşı ölçülmedi** — asıl eksik bu.

Açık issue'lar: **#159** korpus determinizmi (zaman baskılı — sabitlenmiş
artefaktın saklama süresi dolarsa RFV-3B tamamen ölür), **#160** CI kapı
hijyeni, **#161** PR yığını tasfiyesi, **#162** polygon nicemleme (uygulandı,
`f5cc32c`), **#97/#152/#153** motor kalite sahipliği.

---

## 2. Ölçülmüş gerçek: 5 vaka, 5 AYRI mekanizma

RFV-3B 24 vakalık korpusta 5 vaka düşüyor. Baştaki "hepsinde `seam_regression`
var, demek ki tek kök neden" okuması **yanlıştı**; ortak olan yalnızca en
dıştaki fail-closed mesajı.

| vaka | mekanizma | ölçülmüş sayı |
|---|---|---|
| `public-04` | native/evaluator arası sabit boşluk | native 0,99888 · evaluator 0,9914 · fark ~0,0075 |
| `public-05` | hizasızlık şüphesi | native alfa IoU **0,4274**, kafesten bağımsız sabit |
| `public-12` | **TimeoutError** (kalite kapısı değil) | iki koşuda da tekrarladı, ~135 dk |
| `public-14` | knockout bayt bütçesi | 1 800 596 > 1 273 782 (%41 aşım) |
| `public-15` | deficit örtüsü baytı | 1 099 083 deficit px; kafes seyreltmesi baytı yalnız %17,5 düşürüyor |

`measure (0)` geçiyor — kusur evrensel değil.

---

## 3. ÇÜRÜTÜLEN hipotezler (tekrar önerme)

1. **"Stroke taraması etkisiz, baytlar aynı."** Yanlış: `stroke-width="0"` ile
   `"1"` aynı bayt uzunluğunda. Bayt eşitliği beklenen davranış.
2. **"Basamaklar arası in-place mutasyon."** Çürütüldü: `ladder_grid_alpha_sha256_16`
   üç basamakta da aynı (`0c15b833090dc67f`). Girdi değişmiyor.
3. **"Kopuk bileşen elemesi kapsamayı düşürüyor."** Çürütüldü: `public-05`'te
   1/1/0, `public-04`'te 12/12/0 — hiçbir bileşen elenmiyor.
4. **"`nodes=0` polygon düğüm eklemediği için."** Yanlış: o satırlar bayt reddi
   yiyip hiç ölçülmemiş olanlar; `nodes=0` "ölçülmedi" demek.
5. **⚠️ EN ÖNEMLİSİ — nicel öncül hatası.** #162 brifingimde sentetik rampa
   ölçüp "polygon q32 → baytın %24,5'i, bütçeye sığar" demiştim. **Gerçek veri
   %89.** Polygon baytını seviye sayısı değil **kontur döngü geometrisi**
   (köşe koordinatları) domine ediyor. Sentetik düz rampa gerçek logo alfasını
   temsil etmiyor. Bayt tahmini yapacaksan gerçek korpus verisine dayan.

---

## 4. Çözülmüş kusur: siluet sihirli sayısı (`7dfab89`)

`alpha_candidate_identity` üretimde `painter._requantize_alpha`'yı sarmalıyordu
ve `max_levels == 32`'yi sabit kodla yakalayıp **ikili siluet** döndürüyordu.
Kısayol contour merdiveni için kasıtlı, ama seçim **değere** bağlı olduğundan
aynı değeri kullanan polygon merdiveni de sessizce yakalanıyordu.

Belirti: contour (128,64,32) → 127/63/**1**, polygon (64,32,16) → 63/**1**/15.
Konumdan değil değerden kaynaklanıyordu.

Çözüm: `allow_silhouette_shortcut` bayrağı — seçim değerden **niyete** taşındı.
Doğrulandı: q32 artık 31 seviye, baytlar monoton (970 K → 866 K → 811 K),
`ladder_monotonicity_violated` üçünde de `False`.

---

## 5. ⚠️ Bu depoda dikkat edilecekler

**Monkey-patching.** `alpha_candidate_identity`, üretimde şunları değiştirir:
`_requantize_alpha`, `_PAINTER_STROKE_PIXELS` → `(0,1,2,3,4,6,8)`,
`_expand_candidate_paint`, `build_painter_reconstruction_tree`,
`apply_candidate_painter_reconstruction`. **Bir fonksiyonu izole test edip
"sağlam" demek yanıltıcıdır** — üretimde koşan onun yamalı hâli olabilir. Ben
bu tuzağa düştüm: 60 sentetik dağılımda gerçek `_requantize_alpha`'yı
doğruladım, oysa üretimde başka fonksiyon koşuyordu.

**Değiştirilmemesi gereken sözleşmeler** (kasıtlı davranış, testi değiştirip
geçmek yanlış):
- `engine/test_alpha_painter_ledger.py:495` — quantized etiket kümesi
- `engine/test_alpha_painter_stroke_continuation.py:84` — "uygun olmayan redde dur ve geri al"
- `engine/test_alpha_painter_stroke_continuation.py:110`

**Kalite kapıları dokunulmaz.** `alpha_iou_min`, `alpha_mae_max`, `seam_ratio`,
`node_complexity_explosion` (= `parent_nodes + 2500`). Eşik gevşetmek "çözüm"
değildir. Değişiklikler kesin **eklemeli** olmalı.

**Yerel kurulum:**
```
pip install numpy opencv-python-headless pillow scipy defusedxml resvg-py \
            vtracer ezdxf svgpathtools pyclipper pymupdf
cd /home/user/vektoryum-v1
PYTHONPATH=engine:. python3 -m unittest engine.test_alpha_painter_ledger \
  engine.test_alpha_painter_stroke_continuation engine.test_alpha_painter_paint_deficit \
  engine.test_alpha_painter_cumulative_levels engine.test_alpha_painter_polygon_quantized_retry \
  engine.test_alpha_painter_silhouette_shortcut
```

**Regresyon koşucusu:** `cd engine && python3 test_visual_regression.py`
(tek vaka: `--case <ad>`). **`arcaates` main'de de düşüyor** —
`source_alpha_mask_rectangle_budget_exceeded:50488>8251`, worktree ile
doğrulandı, önceden var, senin değişikliğin değil. Uzun koşuları
`run_in_background: true` ile başlat; `nohup ... &` tool çağrısı bitince ölüyor.

**CI günlüğü çıkarma.** Günlükler 250 KB+; `get_job_logs` bağlamı patlatır.
`tail_lines=300-400` ver → dosyaya kaydedilir → python ile ayrıştır:
```python
t=open(F).read().replace('\\"','"').replace('\\u003e','>')
i=t.find('source_alpha_candidate_painter_attempts=')
att,_=json.JSONDecoder().raw_decode(t[t.find('[',i):])
```
Sebep satırı: `re.search(r'"reason":\s*"([^"]{0,240})"', t)`

**Ledger tanılama alanları** (bu oturumda eklendi): `ladder_target_levels`,
`ladder_encoded_levels`, `ladder_grid_alpha_sha256_16`, `ladder_requant_distinct`,
`ladder_monotonicity_violated`, `paint_deficit_pixel_count`,
`source_component_count`, `anchored_source_component_count`,
`detached_source_component_count`.

---

## 6. SIRADAKİ İKİ DENEY

### A) #164'ü canlı korpusa karşı ölç — en yüksek değerli
Bu koşu, doğru yönün **kontur geometrisi** olduğunu doğruladı (kafes
çözünürlüğü değil). #164 zaten `public-15` için o yolu izliyor: döngü
sadeleştirme + adacık eleme. Sözleşme paketini geçiyor ama canlı kanıtı yok.
Ölç, `public-15`'in 1,1 M deficit pikselinden doğan örtü baytı düştü mü bak.

### B) `public-12` timeout sorumluluğunu kapat
Timeout iki koşuda da tekrarladı ama **hiçbir zaman merdivensiz ölçülmedi**
(baseline koşusunda shard 5 iptal olmuştu). Bizim eklediğimiz adayların
(kümülatif +4, polygon-q +3 kafes) yükü sorumlu mu, bilinmiyor.
Kapatmanın yolu: `public-12`'yi **merdivensiz bir head'e** karşı tek vaka
koşturmak. Yerelde yapılamaz — korpus depoda değil (`raw_assets_in_repository:
false`), CI gerekiyor.

Zayıf ama işaret: bu koşuda q32 gerçek nicemleme yapıyor (sahte siluetten
*daha fazla* iş) ve süre benzer kaldı; ayrıca uzun koşanlar hep shard 1 ve 5,
yani süre vakaya özgü olabilir. Kanıt değil.

---

## 7. Kanıtın yeri

- **#158 PR yorumları** — tüm ölçümler, tablolar, kök neden analizleri
- `docs/ai_gorev_brifingleri.md` — 4 görevin ayrıntılı brifingi (⚠️ Görev D'deki
  bayt tahmini yanlış çıktı, madde 3.5'e bak)
- `docs/ai_baslatma_promptu.md` — sohbet modundaki asistanlar için başlatma promptu
- `scratchpad/rfv3b_measure_failures.md` — ham ölçüm notları

## 8. Yayın durumu

RFV-3 `pending`, RFV-4 bloke, yayın kararı `no_go`. Bu koşuda hiçbiri
değişmedi ve **canlı korpus kanıtı gelmeden değişmemeli**.
