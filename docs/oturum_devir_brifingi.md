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

---

## 9. ÖLÇÜLDÜ — Deney A kapandı: #164 canlı korpusa karşı ölçüldü

Brifing yazıldıktan **sonra** üç görev dalı da canlı korpusa karşı koştu.
Deney A'nın "canlı kanıtı yok" durumu artık geçerli değil.

| koşu | head | taban | sonuç |
|---|---|---|---|
| `31872218993` | `a513c52` (#164) | `5c4f790` | measure 0 ✅, 1-5 ❌ |
| `31875365505` | `7dfab89` (#158, siluet düzeltmeli) | — | measure 0 ✅, 1-5 ❌ |

Kafes eşlemesi (`ordered[shard::6]`, case id'leri sıralı): shard 1→public-14,
2→**public-15**, 3→public-04, 4→public-05, 5→public-12. Yani düşen 5 shard,
brifingin 5 vakasıyla birebir örtüşüyor; mekanizmalar da değişmedi.

### 9.1 #164 çalışıyor ama **2,29× yetersiz**

`public-15`'te #164'ün yeni ailesi gerçekten devreye giriyor — tabanda
bulunmayan bir aday üretiyor:

```
encoding_label  = paint-deficit-cumulative-q8-base-delta
encoding_family = paint_deficit_support_compact
actual_serialized_bytes = 769804   byte_limit = 335503   -> byte_rejected
paint_deficit_support_serialized_bytes        = 617139
paint_deficit_support_legacy_serialized_bytes = 701866
paint_deficit_support_saved_bytes             =  84727
paint_deficit_support_rect_count              =  13348
paint_deficit_palette_count = 8    paint_deficit_pixel_count = 943372
```

Yani: en iyi eski paint-deficit adayı 854 531 B iken yeni aday 769 804 B —
**%9,9 kazanç, ölçülmüş**. Ama bütçe 335 503 B; kalan açık **434 301 B** ve
sığmak için bir **%56,4 daha** düşüş gerekiyor. Yön doğru, ölçek yanlış.

Baytı süren şey seviye sayısı değil **destek katmanının dikdörtgen sayısı**:
13 348 rect / 617 139 B ≈ **46 B/rect**. Bütçeye girmek için rect sayısının
kabaca yarıya inmesi gerek. (Madde 3.5'teki nicel öncül hatasının aynısı:
kontur/döngü geometrisi domine ediyor.)

### 9.2 Eski merdiven baytları iki koşuda **bayt bayt aynı**

`public-15` için taban ile #164 head'i arasında eski adayların hiçbiri
oynamadı — #164'ün "mevcut yolları yerinden etme" sözleşmesi tutuyor:

| ölçüm | taban (`7dfab89`) | #164 (`a513c52`) |
|---|---|---|
| `paint_deficit_pixel_count` | 943372 / 943310 | 943372 / 943310 |
| `ladder_grid_alpha_sha256_16` | `c3e91c292a493bb1` | `c3e91c292a493bb1` |
| `paint-deficit-cumulative` | 1007207>335503 | 1007207>335503 |
| `-q8` / `-q12` / `-q16` / `-q20` | 854531 / 892375 / 930262 / 968128 | aynı |

⚠️ Brifingdeki 1 099 083 deficit pikseli **artık 943 372**. Bu düşüş #164'ten
gelmiyor (iki koşuda da aynı); 5c4f790 öncesindeki polygon-q/#162 çalışmasından
geliyor. Yeni bayt tahminlerini 943 372 üzerinden kur.

### 9.3 Siluet düzeltmesi korpus sonucunu değiştirmedi

`7dfab89` merdiveni gerçekten onarıyor — `ladder_encoded_levels` 63/**31**/15,
`requant_distinct` 64/**32**/16, `ladder_monotonicity_violated` **hep false**
(#164 tabanında hâlâ 63/**1**/15, distinct 2, violated **true**). Ama
`public-15`'in bayt redleri iki koşuda da birebir aynı → siluet kusuru bu
vakanın bayt tabanını hiç etkilemiyormuş. Düzeltme doğru, etkisi başka yerde.

### 9.4 Üç görev dalı da siluet düzeltmesinden ÖNCEKİ tabanda

`#164`, `#166`, `#170` üçü de `5c4f790` üzerinde; `7dfab89` hiçbirinde yok
(`git merge-base --is-ancestor 7dfab89 <dal>` → NO). Ölçümleri bu yüzden
merdiven kusuru **açıkken** alındı. `public-15` için bunun sonucu değiştirmediği
yukarıda ölçüldü, ama `public-04`/`public-05` için aynı şey **gösterilmedi** —
o iki dalı yeniden ölçmeden önce `65bc297` üzerine taşımak gerekir.

## 10. Deney B kuruldu — `public-12` merdiven sorumluluğu

`.github/workflows/public12-ladder-timeout-check.yml` (tek seferlik):
aynı runner sınıfında iki kol koşar — `ladder-on` (üretimdeki hâl) ve
`ladder-free` (`_PAINT_DEFICIT_CUMULATIVE_LEVELS` ve
`_NODE_FREE_QUANTIZED_LEVELS` boşaltılmış). Korpus, kimliği zaten doğrulanmış
`31875365505` koşusunun artefaktından iner; yeniden edinim yok.

Merdiven boşaltma **yalnız ölçüm betiğinin kendi sürecinde** yapılır —
`engine/app` altında tek satır değişmez. Bu bilinçli: `engine/app/**`'e
dokunmak RFV-3B canlı iş akışını (6 shard × saatler) tetiklerdi ve varsayılan
davranış değişmediği için o koşu `7dfab89` ile aynı sonucu verirdi.

### SONUÇ (gözlemlenmiş, artefakt beklemeden)

Koşu `31884729592`, iki kol da **12:30:15'te** ölçüme girdi. 13:48 itibarıyla
**ikisi de hâlâ koşuyordu: ~78 dk = ~4 680 s**, yani üretimdeki
`repeat_timeout_seconds = 3600` bütçesini **ikisi de aştı**.

Karar tablosuna göre okuma: **timeout sorumluluğu merdiven adaylarında DEĞİL.**
`public-12` merdivenler tamamen boşaltılmışken bile 3600 s'yi aşıyor. Yani bu
bir kalite işi değil, **vakaya özgü bir performans işi** — brifingdeki "zayıf
işaret" (süre vakaya özgü olabilir) ölçümle desteklendi.

⚠️ Bu, duvar saatinden **doğrudan gözlem**; kesin `elapsed_seconds` değil.

### Harness kusuru: `signal.alarm` yerli kodu kesemiyor

Bütçeyi `signal.alarm(3600)` ile zorlamıştım. Fire etmedi. Nedeni: Python
sinyal işleyicileri yalnız bytecode'lar arasında çalışır; boru hattı zamanının
çoğunu C uzantılarında (vtracer/cv2/resvg) bloke geçirdiğinden SIGALRM
**ertelenir**. Bu yüzden kol 3600 s'de temiz "budget_exceeded" verip
`public12-timing-<kol>.json` yazamadı; iş 150 dk'lık job timeout'una kadar
sürecek ve zamanlama artefaktı üretmeyecek.

Üretim ölçüm koşucusunun bunu doğru yapmasının sebebi de bu: o, tekrarları
**izole işçi süreçlerinde** koşturup süreci dışarıdan öldürüyor. Aynı deney
tekrar kurulacaksa alarm değil, `subprocess` + hard kill kullanılmalı.

Bu kusur sonucu **değiştirmiyor** (gözlem zaten 3600 s'yi aşmış durumda), yalnız
kesin sayıyı vermiyor. Koşu, iptal edip 2 saat daha CI yakmamak için kendi
haline bırakıldı; `if: always()` yüklemesi kısmi kanıtı taşıyabilir.

## 11. Geliştirme: destek katmanı geometri kodlaması (`<rect>` → `<path>`)

Madde 9.1'in işaret ettiği yere yapılan ilk somut müdahale. Ölçüm şunu
söylüyordu: baytı **seviye sayısı değil, destek katmanının rect sayısı**
sürüyor (public-15: 13 348 rect / 617 139 B ≈ **46 B/rect**), ve palet zaten
8'e inmiş durumda — nicemlemeden sıkacak bir şey kalmamış.

### Önce ölçüldü, sonra yazıldı

Sentetik rampa yerine deponun **kendi** `_merged_rectangles_by_level`
birleştiricisiyle logo benzeri bir alfadan gerçek dikdörtgen dağılımı üretildi
(7 seviye, 6 824 rect). Bu profil `<rect>` biçiminde **49,1 B/rect** veriyor —
public-15'te ölçülen 46,2 B/rect'e çok yakın, yani profil temsil edici:

| kodlama | bayt | B/rect | kazanç |
|---|---|---|---|
| `<rect>` kümesi | 335 291 | 49,1 | — |
| `<path>` mutlak | 113 081 | 16,6 | %66,3 |
| `<path>` bağıl (`m dx dy h w v h h-w z`) | **96 240** | **14,1** | **%71,3** |

### Piksel denkliği resvg ile doğrulandı

Endişe, bitişik dikdörtgenlerin tek path altında birleşince kenar
yumuşatmasının değişmesiydi. Ölçüldü, **çürütüldü**: iki bağlamda da
262 144 pikselin **0'ı** farklı, maksimum kanal farkı **0**:

- doğrudan boyama (destek katmanı deseni): AYNI
- `clipPath` içinde (knockout deseni): AYNI

Birleştirici ayrık dikdörtgen üretiyor ve alt-yollar aynı yönde olduğundan
nonzero doldurma kuralı birleşimi veriyor — `<rect>` kümesiyle birebir aynı.

### Uygulama kesinlikle eklemeli

`_emit_paint_deficit_support_geometry` iki biçimi de kurar, **serileştirilmiş
baytı ölçer** ve küçük olanı yazar. Kazanç ölçülemezse `<rect>` biçimi aynen
korunur; bugün bütçeye sığan hiçbir aday büyüyemez. Yeni ledger alanları:
`paint_deficit_support_geometry_encoding`,
`paint_deficit_support_rect_form_bytes`,
`paint_deficit_support_path_form_bytes`,
`paint_deficit_support_geometry_saved_bytes`.

`paint_deficit_support_rect_count` anlamını korudu (geometrideki dikdörtgen
sayısı), kodlamadan bağımsız.

### SONUÇ: ÇÜRÜTÜLDÜ ve GERİ ALINDI

Tahmin şuydu: %71 kazanç public-15'in 617 139 B destek katmanını ≈177 000 B'ye
indirir, #164'ün adayı 769 804 → ≈330 000 B olur ve 335 503 B bütçesine sınırda
sığar.

**Bu hiç sınanamadı, çünkü değişiklik zorunlu regresyon kapısını kırdı:**

```
class_reklam  taban (65bc297)          : PASS  (mode=geometric_logo, best=geo_standard)
class_reklam  <path> kodlamasıyla      : FAIL
  source_alpha_mask_transform_gate_rejected:
      topology_component_regression,topology_hole_regression
```

Aynı vaka, aynı koşucu, tek değişken kodlama. CI'da da `hard-svg-regressions`
aynı head'de düştü. `arcaates` FAIL'i ilgisiz (madde 5: main'de de düşüyor).

### Neden yanıldım — izole doğrulama tuzağı (madde 5'in tekrarı)

resvg piksel denkliğini **destek katmanını tek başına** render ederek ölçtüm ve
0 fark buldum. Bu ölçüm doğruydu ama **yanlış soruyu** yanıtlıyordu: üretimde bu
geometri bir `<mask>` bağlamında, başka dönüşümlerin altında kullanılıyor ve
topoloji kapısı bileşen/delik sayıyor. Ayrı `<rect>`'ler kenar paylaşsalar bile
ayrı bileşen olarak ölçülürken, tek `<path>` altındaki alt-yollar **tek bileşene
kaynıyor** — bileşen ve delik sayısı bu yüzden değişiyor. Yani piksel aynı
kalırken topoloji değişebiliyor; kapının ölçtüğü şey piksel değil.

Brifingin 5. maddesi tam olarak bunu söylüyordu: "Bir fonksiyonu izole test edip
'sağlam' demek yanıltıcıdır." Ben de aynı tuzağa düştüm.

### Geriye kalan sağlam bilgi

- Bayt kazancı **gerçek ve ölçülmüş**: bağıl `<path>` kodlaması 49,1 → 14,1
  B/rect (%71,3). Bu sayı hâlâ geçerli.
- Ama kazanç **bu biçimde** alınamaz: topoloji kapısı kodlamaya duyarlı.
- Doğru yön, baytı düşürürken **bileşen kimliğini korumak**: ya her dikdörtgen
  ayrı `<path>` olarak kalmalı (kazanç çok daha az: sarmalayıcı `<g>` düşer,
  `<rect>`→`<path>` başına ~10 B), ya da topoloji kapısının kodlamadan bağımsız
  ölçmesi gerekir — ikincisi kapı sahibiyle konuşulacak bir sözleşme değişikliği,
  tek taraflı yapılmamalı.
- Knockout'a dokunulmadı; iyi ki dokunulmamış — aynı kaynaşma orada da olurdu.

⚠️ Sıradaki oturuma: bu yolu "bayt kazancı yok" diye kapatma. Kazanç var, engel
topoloji kapısının kodlama duyarlılığı. Ölçülmemiş tek şey, dikdörtgen başına
ayrı `<path>` biçiminin ne kadar kazandırdığı.
