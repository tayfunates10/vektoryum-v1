# Oturum devir brifingi — Vektoryum RFV-3B / motor kalite hattı

Bu belge, tükenmiş bir oturumdan devralacak yeni oturum için yazıldı. Amacı,
aynı yolları tekrar yürümeni önlemek: nelerin **ölçüldüğünü**, nelerin
**çürütüldüğünü** ve sıradaki iki somut deneyi içerir.


---

## 0. ÖNCE BUNU OKU — bu oturumda neler kapandı

⚠️ Aşağıdaki 1-8. maddeler **ilk** oturumun brifingidir ve bazı yerleri
sonradan **çürütüldü**. Çakışma görürsen 9+ maddeler geçerlidir.

**Kapanan sorular**

| soru | cevap | madde |
|---|---|---|
| #164 canlı korpusta işe yarıyor mu? | Evet ama 2,29× yetersiz; sözleşmesi temiz | 9 |
| `public-12` timeout'undan merdivenler mi sorumlu? | **Hayır** — merdivensiz de 4 197 s | 10 |
| `public-12`'nin kök nedeni ne? | Kontur fallback düğüm patlaması, 56× | 10.1 |
| `public-15`'in destek katmanı kaç bayt? | 701 625 B / 15 283 rect (canlı ölçüm) | 11.1 |

**Tekrar denenmemesi gereken kapalı yollar** (hepsi ölçüldü)

1. Destek katmanını tek birleşik `<path>` yapmak → render bozuluyor (206 660 px) — 11
2. Rect başına ayrı `<path>` → topoloji kapısı düşüyor, eleman tipi downstream'e bağlı — 11
3. `_rect_path`'i yeniden icat etmek → depoda **zaten var**, yalnız `<mask>`'te güvenli — 11.2
4. Pahalı dalda erken durdurma → kazanç **1,0×**, dizge kurma maliyetin %2'si — 10.2
5. `4*C` alt sınır kapısı → sözleşme testini kırdı, `_encode_contours` köprü kullanıyor — 10.4
6. Toplam kontur sayısıyla kapı → **karşı örnek** ölçüldü (16 860 toplam / 3 budanmış) — 10.7

**Doğrulanmış araçlar** (kullan)

- `kontur = 2*C8 - E8` kimliği — 10 ms'de tam, `findContours` 103 s. Tanılama için değerli (10.6)
- Sentetik profil kurarken deponun **kendi** `_merged_rectangles_by_level`'ını kullan:
  tahmin %68,1 / gerçek %67,9 tuttu (11.1)
- Bu depoda izole doğrulama yanıltır. Tek güvenilir kontrol:
  `cd engine && python3 test_visual_regression.py --case class_reklam`

**Sıradaki en umutlu iş:** `public-14` — bütçeye **en yakın** vaka (1,41×),
bugüne dek hiç çalışılmadı. Madde 12.

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
| `public-12` | ~~TimeoutError~~ → **kontur fallback düğüm patlaması** (madde 10) | plan 4 034 685 düğüm üretiyor, paya düşen 71 838 → **56×** |
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

### KESİN SAYILAR (artefaktlar geldi — önceki tahminimi düzeltiyorum)

⚠️ "Zamanlama artefaktı üretilemeyecek" demiştim. **Yanlıştı** — iki artefakt da
indi. Alarm gerçekten hiç ateşlenmedi, ama koşular kendiliğinden bitti:

| kol | `elapsed_seconds` | bütçe içi | nasıl bitti |
|---|---|---|---|
| `ladder-free` | **4 197,0** | **hayır** | boru hattının kendi hatası |
| `ladder-on` | **≈5 482** (zaman damgasından) | hayır | — |

`ladder-free` kolunda merdivenlerin gerçekten boş olduğu ledger'da doğrulandı:
`effective_paint_deficit_cumulative_levels: []`,
`effective_node_free_quantized_levels: []`.

**Sonuç kesinleşti:** merdivenler süreye ~1 285 s (~%23) ekliyor, ama tamamen
kaldırıldığında bile 4 197 s > 3 600 s. Yani merdivenler **katkıda bulunuyor,
sorumlu değil**. `public-12` merdivensiz de bütçeyi aşar.

### KÖK NEDEN ÖLÇÜLDÜ: düğüm patlaması (timeout değil)

`ladder-free` kolu zaman aşımıyla değil, boru hattının kendi bütçe reddiyle
bitti:

```
source_alpha_mask_contour_fallback_budget_rejected:
  path_bytes = 14 277 024 / 3 138 171   (4,5x)
  path_count =      1 157 /     4 120   (uygun)
  path_nodes =  4 058 631 /    95 784   (42x)
```

`alpha_mask_adaptive.py:631` `_contour_fallback_plan`. Yani `public-12`'nin
sorunu "yavaş çalışması" değil: kontur fallback'i **4,06 milyon düğüm**
üretiyor, bütçenin **42 katı**. Süre bunun sonucu, sebebi değil.

Bu, `public-12`'yi tamamen yeniden sınıflandırıyor: bir zaman aşımı vakası
değil, bir **geometri karmaşıklığı** vakası. Doğru iş, süreyi hızlandırmak
değil, kontur fallback'inin düğüm sayısını düşürmek (ya da bu vakada o
fallback'e hiç düşmemek).

### Harness kusuru: `signal.alarm` yerli kodu kesemiyor

Bütçeyi `signal.alarm(3600)` ile zorlamıştım; 4 197 s'de bile ateşlenmedi
(status `failed`, `budget_exceeded` değil). Python sinyal işleyicileri yalnız
bytecode'lar arasında koşar, zaman C uzantılarında bloke geçer. Üretim ölçüm
koşucusunun tekrarları izole **süreçlerde** koşturup dışarıdan öldürmesinin
sebebi bu. Aynı deney tekrar kurulursa `subprocess` + hard kill gerekir.

Bu kez sonucu etkilemedi çünkü boru hattı kendi hatasıyla zaten sonlandı.

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

### SONUÇ: İKİ VARYANT DA ÇÜRÜTÜLDÜ, İKİSİ DE GERİ ALINDI

Bayt kazancı gerçek, ama bu sitede **kodlama biçimi serbest değil**. İki varyant
denendi, ikisi de aynı kapıdan düştü:

```
class_reklam  taban (65bc297)              : PASS
class_reklam  tek birleşik <path>          : FAIL  topology_component_regression,
class_reklam  dikdörtgen başına <path>     : FAIL  topology_hole_regression
```

Aynı vaka, aynı koşucu, tek değişken kodlama.

### ⚠️ ÖNCEKİ AÇIKLAMAM YANLIŞTI — düzeltme

İlk yazdığım "alt-yollar tek bileşene kaynıyor, o yüzden topoloji sayacı farklı
okuyor" açıklaması **yanlıştı**. Topoloji SVG elemanlarından sayılmıyor:
`transform_journal.py:196-210` render edilmiş **rasteri** `cv2` ile bağlı
bileşenlere ayırıp kaynakla karşılaştırıyor (`component_delta`, `hole_delta`).

Doğru ölçümler:

| karşılaştırma (üretimdeki gibi kesirli ölçek altında) | farklı piksel | bayt |
|---|---|---|
| ayrı `<rect>` vs tek birleşik `<path>` | **206 660** (max kanal 80) | %68,1 kazanç |
| ayrı `<rect>` vs rect başına `<path>` | **0 — piksel özdeş** | %35,3 kazanç |

Yani *birleşik* varyantın düşmesinin sebebi gerçekten render farkıydı: birleşik
path'te paylaşılan kenarlar iç kenar olup yumuşatılmıyor, ayrı dikdörtgenlerde
ayrı ayrı harmanlanıyor. İlk testimi **ölçeksiz** yaptığım için bunu kaçırmıştım.

### Asıl ders: rect başına path piksel-özdeş AMA yine de düştü

Ve kritik nokta bu. `rect başına <path>` kesirli ölçekte bile 0 piksel farkı
veriyor, buna rağmen üretimde aynı kapıdan düşüyor. Demek ki bu sitede sorun
**render değil**: destek katmanının eleman tipi downstream aşamalar için
yük taşıyor.

Somut ipucu: `app/alpha_candidate_support_compact.py:753`
`_install_runtime_compactors()` **import anında** koşuyor ve
`apply_direct_element_alpha` ile adaptive mask fabrikasını sarmalıyor
(`_compact_direct_artifact`, `_compact_complex_clip`, `make_compact_primitive_alpha_first`).
Bu katman `<rect>` bekliyor olabilir; `<path>` verince devreye girmiyor ve
nihai artefakt değişiyor.

### Sıradaki oturuma kural

Bu siteyi **izole doğrulamayla onaylama**. Üç kez yanıldım: ölçeksiz render,
ölçekli render, ikisi de "güvenli" dedi, üretim ikisini de reddetti. Bu depoda
tek güvenilir kontrol:

```
cd engine && python3 test_visual_regression.py --case class_reklam
```

Bayt kazancını almak isteyen önce şunu yanıtlamalı: **hangi downstream aşama
`<rect>` eleman tipine bağlı?** Cevap bulunmadan kodlama değiştirilmemeli.

### Saklanacak sayılar

- `<rect>` biçimi: 49,1 B/rect (public-15'te ölçülen 46,2'ye yakın)
- rect başına `<path>`: %35,3 kazanç, piksel-özdeş
- tek birleşik `<path>`: %68-71 kazanç ama render'ı bozuyor — bu yol kapalı
- public-15 için %35,3 bile yetmez: 617 139 → ≈399 000, aday 769 804 → ≈552 000,
  bütçe 335 503 (hâlâ 1,65×). Yani bu site tek başına public-15'i kurtarmıyor.
- Knockout'ta aynı oran public-14'ü **sığdırabilirdi** (1 800 596 × 0,70 ≈
  1 260 000 < 1 273 782) — ama oraya hiç dokunulmadı ve `test_alpha_clip_encoding.py:142`
  sözleşmesi orada da `<rect>` bekliyor. Aynı downstream sorusu orada da geçerli.

### 11.1 CANLI KORPUS SAYILARI (bedava geldi — sentetik tahmini değiştirin)

Geri alınan ilk varyant (birleşik `<path>`) `a3025963` head'inde canlı korpusa
karşı koştu (run `31887421488`, shard 2). Ledger alanlarını oraya bağlamış
olmam sayesinde **gerçek** sayılar elimizde:

| ölçüm | `public-15` gerçek değeri |
|---|---|
| destek katmanı rect sayısı | **15 283** |
| `<rect>` biçimi | **701 625 B** |
| birleşik `<path>` biçimi | **225 166 B** |
| kazanç | **%67,9** |

Sentetik profilim %68,1 demişti; gerçek %67,9. **Ölçüm yöntemi sağlam** —
`_merged_rectangles_by_level` ile üretilen profil gerçek korpusu temsil ediyor.
(Madde 3.5'teki hatanın tekrarı değil: bu kez öncül doğrulandı.)

Adayların toplam baytı (birleşik `<path>` desteğiyle):

| aday | toplam bayt | bütçe 335 503'e göre |
|---|---|---|
| `paint-deficit-q24` | 854 316 | 2,55× |
| `paint-deficit-cumulative` | 530 748 | 1,58× |
| `-q20` | 491 669 | 1,47× |
| `-q16` | 453 803 | 1,35× |
| `-q12` | 415 916 | 1,24× |
| **`-q8`** | **378 072** | **1,13×** |

Yani `-q8` bütçeye **yalnız 42 569 B** uzakta kalmış (önceki 854 531'de 2,55×
idi). Bu, public-15 için şimdiye kadarki en yakın nokta.

### ⚠️ Ama bu sayı GÜVENLİ BİÇİMDE ALINAMAZ

Yukarıdaki 378 072, **birleşik** `<path>` biçiminin sayısıdır ve o biçim
render'ı bozuyor (206 660 piksel farkı, madde 11). Piksel-özdeş olan biçim
rect başına ayrı `<path>` ve onun kazancı yalnız ~%35:

```
destek:  701 625 -> ~454 000   (rect basina path, ~%35)
-q8 toplam: 378 072 + (454 000 - 225 166) ~= 606 900   -> butce 335 503'un 1,81 kati
```

Yani **public-15'i bu site tek başına kurtarmıyor**: güvenli biçim yetmiyor,
yeten biçim güvenli değil. Üstelik rect başına path varyantı da topoloji
kapısından düştü (eleman tipi downstream'e bağlı, madde 11).

**Sonuç:** buradaki asıl kilit bayt değil, `<rect>` eleman tipine bağlı
downstream aşamanın kim olduğu. O çözülürse 42 569 B'lik açık (#164'ün
base_delta kazancı ~84 727 B ile birlikte) kapanabilir görünüyor — ama bu da
ölçülmeden iddia edilmemeli.

### 11.2 CEVAP BULUNDU: bu optimizasyon depoda ZATEN var — ama başka yerde

"Hangi downstream aşama `<rect>`'e bağlı?" sorusunun cevabı:

**`engine/app/alpha_mask_adaptive.py:36` `_rect_path()` + `:68`
`_compact_mask_rectangles()`** — depo, dikdörtgenleri tek bir path'e birleştiren
optimizasyonu **zaten yapıyor**:

```python
commands.append(f"M{x} {y}h{width}v{height}h-{width}Z")   # _rect_path
```

Bu, benim "icat ettiğim" birleşik biçimin **birebir aynısı**. Ama yalnız
`<mask>` içindeki `data-vektoryum-alpha-level` gruplarına uygulanıyor:

```python
if group.get("data-vektoryum-alpha-level") is None: continue
rectangles = [c for c in group if _local_name(c.tag) == "rect"]
```

Yani seçici olarak **`rect` çocuk bekliyor**; `<path>` verilirse liste boşalır ve
adım sessizce atlanır. `alpha_candidate_support_compact.py:632`
(`_compact_complex_clip`) de aynı deseni izliyor: çocukların **hepsi** `rect`
değilse `return 0`.

Her ikisi de `_install_runtime_compactors()` ile **import anında** üretim
fonksiyonlarına takılıyor (brifing madde 5'teki monkey-patch uyarısı).

### Üç başarısızlığımın tek açıklaması

| katman | birleştirme | neden |
|---|---|---|
| `<mask>` alfa-seviye grupları | **GÜVENLİ, zaten yapılıyor** | ikili maske; paylaşılan kenarda yumuşatma sorunu yok |
| paint-deficit **destek** katmanı (görünür boya) | **GÜVENLİ DEĞİL** | kesirli ölçekte 206 660 piksel farkı |

Ben `_rect_path`'i yeniden icat edip **görünür boya** katmanına uyguladım.
Depo bunu maske geometrisinde yapıyor çünkü orada güvenli; destek katmanında
yapmıyor çünkü orada değil. Sınır bilinçliymiş, ben fark etmemişim.

Rect başına `<path>` varyantının düşmesi de aynı kökten: eleman tipini
değiştirince yukarıdaki `rect` filtreleri devre dışı kalıyor ve daha önce
koşan compaction adımı artık koşmuyor — nihai artefakt bu yüzden değişiyor.

### Bunun public-15 için anlamı

Maske tarafı **zaten sıkıştırılmış**. `public-15`'in 701 625 B'lik destek
katmanı, geriye kalan sıkıştırılmamış kütle — ve orada birleştirme yasak.
Dolayısıyla 42 569 B'lik açık **eleman tipi değiştirerek kapatılamaz**.

Kalan gerçek kaldıraçlar:
1. **Dikdörtgen sayısını düşürmek** (15 283 rect). Bayt/rect değil, rect sayısı.
   #164'ün adacık eleme + döngü sadeleştirme yönü doğru olan buydu.
2. **#164'ün `base_delta`'sı** (~84 727 B ölçülmüş kazanç).
3. Destek katmanının hiç üretilmemesi (deficit'i baştan azaltmak).

⚠️ Sonraki oturuma: `<rect>` → `<path>` yolunu **kapalı** say. Denendi (iki
varyant), ölçüldü, ikisi de düştü ve nedeni artık biliniyor. Bunun yerine rect
SAYISINI düşüren işlere bak.


### 10.1 Düğüm patlamasının ayrıştırması (aritmetik)

`alpha_mask_budget.py:123` → `node_limit = max(parent*4, parent+2500)`.
Gözlenen limit **95 784** olduğuna göre parent×4 baskın ve
**parent_node_count = 23 946** (23 946 × 4 = 95 784).

| bileşen | düğüm | not |
|---|---|---|
| ana çizim (parent artwork) | **23 946** | normal, sorun değil |
| kontur fallback planı | **4 034 685** | 4 058 631 − 23 946 |
| plana kalan pay | 71 838 | 95 784 − 23 946 |
| aşım | **56×** | 4 034 685 / 71 838 |

Yani patlama **ana çizimden gelmiyor**; tamamen
`alpha_mask_adaptive._build_contour_plan()` içinde üretiliyor. Bu fonksiyon
`_quantize_alpha` çıktısındaki **her alfa seviyesi için kontur** çıkarıyor;
`public-12` gibi gürültülü/fotoğrafımsı alfada seviye başına kontur sayısı
patlıyor. Süre de bunun sonucu: 4 milyon komut üretmek 4 197 s sürüyor.

### 10.2 Erken durdurma DENENDİ ve ÖLÇÜMLE ÇÜRÜTÜLDÜ

Önerdiğim adım şuydu: pahalı dal reddedilecekse dev dizgeleri kurmayalım,
sayıları aritmetikle çıkaralım. Uygulandı ve aritmetik **birebir** doğrulandı
(`command_count`, `path_markup_bytes`, `path_count`, `contour_count` tümü aynı;
`d` yalnız sayaç yolunda `None`).

Ama kazanç ölçülünce: **1,0×**. 1200×1200 dama tahtası, 720 000 hücre,
3 600 000 komut: dizge kuran yol 202,26 s, sayaç yolu 197,85 s. Yani dizge
kurma toplam maliyetin **~%2'si**. Değişiklik geri alındı — karmaşıklık ekleyip
hiçbir şey kazandırmıyor.

### Profil: zaman NEREDE geçiyor (ölçüldü)

Tek seviye, 1200×1200, 717 603 kontur:

| adım | süre |
|---|---|
| maske kurma | 0,00 s |
| **`cv2.findContours`** | **204,52 s** |
| `_canonical_contour` Python döngüsü | 5,14 s |
| `np.nonzero` | 0,01 s |
| dizge kurma (yukarıdan) | ~4 s |

**Darboğaz `cv2.findContours`** — `RETR_CCOMP` ile parçalı alanda 700 binden
fazla kontur çıkarıyor. `public-12`'nin 4 197 s'si buradan geliyor, pahalı
yeniden ifadeden değil.

Kritik ayrıntı: bu maliyet `_build_contour_plan`'ın **ilk** döngüsünde, yani
pahalı dala düşmeden ÖNCE ödeniyor. Dolayısıyla pahalı dalı kısaltan hiçbir
optimizasyon `public-12`'yi hızlandıramaz.

### 10.3 ÖN-KONTROL TASARIMI — ölçüldü, kanıtlanabilir, uygulanmadı

Ön-kontrol adaylarının maliyeti ve tahmin gücü ölçüldü (1200x1200):

| ölçüt | dama tahtası (patolojik) | logo benzeri (normal) |
|---|---|---|
| `count_nonzero` | 0,14 ms | 0,15 ms |
| satır geçişi (`np.diff`) | 24,76 ms | 6,20 ms |
| `connectedComponents` 8-komşuluk | 4,95 ms → **1 bileşen** | 1,11 ms → 2 |
| **`connectedComponents` 4-komşuluk** | **3,01 ms → 720 000 bileşen** | **1,32 ms → 2** |
| `cv2.findContours` | **98 647 ms** | 1,24 ms |

İki sonuç:
1. `findContours` normal girdide zaten hızlı (1,24 ms); ön-kontrolün yalnız
   **patolojik** durumu yakalaması yeterli.
2. **8-komşuluk kullanılamaz** — dama tahtasına "1 bileşen" diyor. Doğru olan
   **4-komşuluk**: köşeden değen dolgular ayrı kapalı alt-yollardır.

### Kanıt (bu sefer ampirik benzerlik değil, alt sınır)

Her SVG dolgu gösterimi, 4-komşu bileşen başına **en az bir kapalı alt-yol**,
alt-yol başına **en az 4 komut** (M + en az iki çizgi + Z) gerektirir.
Dolayısıyla `komut_sayısı >= 4 * C`. Eğer `4 * C > node_allowance` ise kabul
**matematiksel olarak imkânsızdır** ve `findContours`'a hiç girilmemelidir.

- `public-12` benzeri alan: 4 × 720 000 = 2 880 000 > 71 838 → kanıtlı ret,
  ~98 s (çok seviyede ~4 000 s) hesap atlanır, maliyet 3 ms.
- Normal logo: 4 × 2 = 8, bütçenin çok altında → hiçbir şey değişmez.

Bu, bu oturumdaki diğer denemelerden farklı: gerekçe "izole testte aynı
görünüyor" değil, **matematiksel alt sınır**. Yine de regresyonla doğrulanmalı.

### ⚠️ 10.4 ÖN-KONTROL UYGULANDI ve ÇÜRÜTÜLDÜ — "kanıt" yanlıştı

Yukarıdaki alt sınır tasarımı uygulandı ve deponun **kendi sözleşme testi**
tarafından anında reddedildi:

```
engine/test_alpha_mask_adaptive.py
  test_rect_alpha_seams_retry_as_budgeted_contours -> ERROR
  source_alpha_mask_contour_fallback_budget_rejected:
      path_nodes=>=2533/2505
```

Yani şu anda **başarıyla kabul edilen** bir vaka, benim kapımla kıl payı
(2533 > 2505) reddedildi. Kapı yanlış.

**Neden yanıldım — kanıtı yanlış gösterime kurdum.** "Her bileşen en az bir
kapalı alt-yol, alt-yol başına en az 4 komut" önermesi genel SVG semantiği
için doğru, ama bu depo o gösterimi kullanmıyor. `alpha_mask_budget.py:220`
`_encode_contours` docstring'i açıkça söylüyor:

> "Encode disconnected contours as one even-odd walk with doubled bridges."

Konturlar **köprülerle tek bir even-odd yürüyüşe** bağlanıyor; bileşen başına
ayrı `M...Z` yok. Ek kontur başına maliyet ~4 değil, köprü gidiş-dönüşü kadar
(~2). Ben `_encode_contours`'a bakmadan genel semantikten akıl yürüttüm —
madde 11.2'deki hatanın (deponun mevcut mekanizmasına bakmamak) aynısı.

**Gözlenen veri noktası:** başarısız vakada 4×C = 2 533 iken gerçek plan
2 505 limitine sığıyor → C ≈ 633, gerçek maliyet bileşen başına **4'ten
küçük**. Dama tahtasında 2×C = 1 440 000 hâlâ 71 838'i aşıyor, yani daha
küçük bir katsayı patolojik vakayı yine yakalar.

⚠️ Ama katsayıyı **tahmin etmeyin**. Doğrusu `_encode_contours`'un ürettiği
komut sayısını kontur sayısına bağlayan gerçek formülü o fonksiyondan
türetmek. Ölçmeden konulan her katsayı bu tuzağın tekrarıdır.

Darboğaz teşhisi (madde 10.2: `cv2.findContours` 98 647 ms) **geçerliliğini
koruyor**; çürüyen yalnız kapının eşiği. Ucuz ön-kontrol fikri de geçerli:
4-komşuluk `connectedComponents` 3,01 ms ve `public-12` benzeri alanı
ayırt ediyor. Eksik olan tek şey doğru katsayı.

### ⚠️ Uygulamadan önce çözülecek tek soru

Ön-kontrol hangi hatayı fırlatmalı? Şu an bu yol
`..._budget_rejected:path_bytes=...,path_count=...,path_nodes=...` veriyor;
erken çıkışta gerçek sayılar hesaplanmamış olacak. `..._unavailable`
fırlatmak **farklı bir dal** olduğundan çağıranın davranışını değiştirebilir.
Güvenli seçenek: aynı `budget_rejected` kodunu, `path_nodes` yerine kanıtlı
alt sınırla (`>=4C`) bildirmek. Karar verilmeden uygulanmamalı.

Doğrulama planı: sözleşme testleri + `test_visual_regression.py` (kabul edilen
vaka etkilenmemeli) + canlı korpusta `public-12` shard süresi (asıl kanıt).

### Sıradaki gerçek kaldıraç (ölçülmedi, öneri)

`cv2.findContours` çağrılmadan ÖNCE ucuz bir parçalanma ön-kontrolü: alan
kabul edilebilir hiçbir bütçeye sığmayacak kadar parçalıysa kontur çıkarmaya
hiç girmemek. Aday ucuz ölçüt: satır-içi geçiş sayısı (`np.diff` ile
vektörel, `np.nonzero` gibi ~0,01 s) ya da `cv2.connectedComponents`
(findContours'tan ucuz olması beklenir ama **ölçülmeli**).

⚠️ Sonuç korunmalı: ön-kontrol yalnız zaten reddedilecek alanları elemeli.
Eşik, kabul edilen hiçbir vakayı etkilemeyecek şekilde bütçeden türetilmeli.


### 10.5 BELİRLEYİCİ: `_encode_contours` seviye başına **2 komut** üretiyor

`alpha_mask_budget.py:255` son satır:

```python
return f"M{x0} {y0}l" + ",".join(deltas)
```

`_PATH_COMMAND` yalnız harfleri sayar → bu dizge **her zaman 2 komut** (`M` +
`l`), kontur sayısından ve nokta sayısından **bağımsız**. Docstring zaten
söylüyor: *"One M and one l command encode an entire alpha level."*

Bu, madde 10.3/10.4'teki tüm akıl yürütmeyi geçersiz kılar ve resmi düzeltir:

- **Kompakt dalın düğüm maliyeti önemsizdir** (2 × seviye sayısı). Bu dal
  düğüm bütçesini pratikte hiç zorlamaz.
- `public-12`'nin **4 058 631 düğümünün tamamı pahalı piksel-başı daldan**
  geliyor (hücre başına 5 komut).
- Dolayısıyla asıl soru "düğüm sayısı ne olacak" değil, **"pahalı dala
  düşecek miyiz"**. O karar tek yerde veriliyor:
  `contour_count > _MAX_COMPACT_CONTOURS` (**4096**) ya da tüm konturların
  degenerate olması.

### Doğru kapının koşulu (kısmen ölçüldü, tamamlanmadı)

Pahalı dala düşüleceği ÖNCEDEN bilinirse, o dalın düğüm maliyeti **tam olarak
5 × hücre_sayısı**'dır ve `np.count_nonzero` ile **0,14 ms**'de hesaplanır.
`5 × hücre > node_allowance` ise ret kanıtlıdır.

Eksik olan tek halka: `contour_count > 4096` olacağını `findContours`
çağırmadan ucuza bilmek. Ölçülenler:

| girdi | 4-komşu bileşen | 8-komşu bileşen | gerçek kontur |
|---|---|---|---|
| dama tahtası | 720 000 | **1** | 717 603 |
| logo benzeri | 2 | 2 | 3 |

⚠️ Dikkat: **8-komşuluk kontur sayısının alt sınırıdır ama işe yaramaz**
(dama tahtasında 1). **4-komşuluk ise üst sınır tarafındadır**, yani
`C4 > 4096` tek başına `kontur > 4096` demek DEĞİLDİR — sağlam bir kapı için
yeterli değil.

Dama tahtasındaki 717 603 konturun kaynağı `RETR_CCOMP`'un **delikleri** de
sayması. Dolayısıyla doğru ucuz ölçüt muhtemelen **delik sayısı**, o da Euler
karakteristiğinden gelir (2×2 bit-quad sayımı, numpy ile vektörel ve ucuz
olması beklenir). **Ölçülmedi.** Sonraki oturum buradan devam etmeli:
önce bit-quad Euler maliyetini `findContours`'a karşı ölç, sonra kapıyı kur.

Yani zincir: `delik/kontur tahmini > 4096` → pahalı dal kesin →
`5 × hücre > pay` ise kanıtlı ret → `findContours` hiç çağrılmaz.


### 10.6 EKSİK HALKA BULUNDU: kontur sayısı `findContours` çağrılmadan TAM hesaplanıyor

2×2 bit-quad Euler sayımı ölçüldü ve `RETR_CCOMP` kontur sayısını **birebir**
veriyor:

```
kontur_sayisi = C8 + H8   ve   H8 = C8 - E8   =>   kontur = 2*C8 - E8
```

| girdi | öngörü `2*C8 - E8` | GERÇEK kontur | Euler+connComp | `findContours` |
|---|---|---|---|---|
| dama tahtası | **717 603** | **717 603** | 9,6 ms | **103 350 ms** |
| logo benzeri | **3** | **3** | 8,5 ms | 0,93 ms |

Bu bir sezgisel tahmin değil **kimlik**: `RETR_CCOMP` dış konturları + delikleri
döndürür, Euler karakteristiği deliği tam verir. Maliyet ~**10 000×** daha az.

Bit-quad (numpy, vektörel):

```python
b = np.pad(mask.astype(np.uint8), 1)
idx = (b[:-1,:-1] + 2*b[:-1,1:] + 4*b[1:,:-1] + 8*b[1:,1:]).ravel()
q = np.bincount(idx, minlength=16)
E8 = (Q1 - Q3 - 2*QD) / 4.0     # Q1=q1+q2+q4+q8, Q3=q7+q11+q13+q14, QD=q6+q9
```

### ⚠️ Kalan tek sağlamlık boşluğu (uygulamadan önce kapatılmalı)

Koddaki `contour_count`, `_canonical_contour` ile **budanmış** (3 noktadan az
konturlar atılmış) sayıdır. `2*C8 - E8` ise **budanmamış toplamdır**, yani
koddaki sayının ÜST sınırıdır. Üst sınırın 4096'yı aşması, budanmış sayının
aştığını **kanıtlamaz**.

Kritik orta durum: toplam kontur çok yüksek ama neredeyse hepsi degenerate ve
birkaç gerçek kontur var → kod kompakt dalı kullanır ve rahatça sığar; kapı
ise yanlışlıkla reddeder. (Not: hepsi degenerate ise
`not layers and pruned_contour_count` koşulu zaten pahalı dala düşürür, o
durum sorun değil.)

Kapatma yolu: degenerate bileşenleri düşmek. `cv2.connectedComponentsWithStats`
alanları veriyor; alanı 1-2 olan bileşenler degenerate dış kontur üretir.
Delik tarafı için de aynısı arka plan üzerinde yapılabilir. **Ölçülmedi.**

### Tamamlanmış zincir (son halka hariç)

1. `2*C8 - E8` ile kontur sayısı — ~10 ms, tam. *(degenerate düzeltmesi eksik)*
2. Budanmış kontur > 4096 → kompakt dal kullanılamaz → pahalı dal kesin.
3. Pahalı dal düğüm maliyeti **tam olarak** `5 * hücre` — `count_nonzero`, 0,14 ms.
4. `5 * hücre > pay` → ret kanıtlı → `findContours` hiç çağrılmaz.

`public-12` için: ~10 ms maliyetle ~4 000 s hesap atlanır.


### 10.7 KARŞI ÖRNEK: toplam kontur sayısı SAĞLAM BİR KAPI DEĞİL (kapandı)

Madde 10.6'daki sağlamlık boşluğu teorik değilmiş. Ölçüldü:

Girdi: iki gerçek şekil (dikdörtgen + halka) + 800x800 alanda izole tek piksel
gürültü (taranmış/gürültülü kaynağı temsil eder).

| ölçüm | değer |
|---|---|
| öngörü `2*C8 - E8` | 16 860 |
| GERÇEK toplam kontur | **16 860** (kimlik yine tam) |
| **budanmış = kodun `contour_count`'u** | **3** |
| `_MAX_COMPACT_CONTOURS` | 4 096 |
| üst sınır 4096'yı aşıyor mu | **evet** |
| **kod pahalı dala düşer mi** | **HAYIR** — kompakt dal, başarılı |

`2*C8 - E8 > 4096` kapısı bu girdiyi **yanlışlıkla reddederdi**. Kapı sağlam
değil; bu bir boşluk değil **kesin karşı örnek**.

Sebep: izole tek pikseller findContours'ta kontur üretir ama
`_canonical_contour` onları budar (3 noktadan az). Kod bunu bilinçli yapıyor —
`_build_contour_plan` docstring'i: *"Degenerate one/two point islands are
omitted when measurable contours exist."* Toplam sayı bu budamayı göremez ve
alan tabanlı düzeltme de kesin değildir (1x3 düz çizgi 3 piksel ama
CHAIN_APPROX_SIMPLE ile 2 nokta üretip budanır).

### Bu hattın durumu: KAPALI

Toplam kontur sayısından türetilen hiçbir ucuz kapı sağlam olamaz, çünkü
karar **budanmış** sayıya bağlı ve budama ancak konturlar çıkarıldıktan sonra
bilinir. Sonraki oturum bu yolu tekrar denemesin.

**Saklanacak sağlam sonuçlar:**
- `kontur = 2*C8 - E8` **kimliği tamdır** (üç ayrı girdide birebir doğrulandı:
  717 603 / 3 / 16 860). Kapı olarak değil ama **tanılama** olarak değerli:
  bir vakanın neden yavaş olduğunu `findContours` çağırmadan 10 ms'de söyler.
- `findContours` maliyeti kontur sayısıyla **süperdoğrusal** artıyor:
  16 860 kontur → 16 ms, 717 603 kontur → 103 350 ms.
- Pahalı dalın düğüm maliyeti tam olarak `5 * hücre` (0,14 ms'de hesaplanır).
- Kompakt dal seviye başına yalnız **2 komut** üretir (madde 10.5).

**Geriye kalan gerçek seçenek:** `public-12`'nin maliyeti `findContours`'un
süperdoğrusal davranışından geliyor ve bunu sonucu değiştirmeden ucuzlatmanın
yolu bulunamadı. Seçenekler (hiçbiri ölçülmedi): konturları seviye seviye
tembel çıkarıp bütçe aşılınca durmak (sonuç aynı kalır mı? ölçülmeli), ya da
`public-12`'yi kalite değil **performans** sınırı olarak kabul edip korpus
zaman aşımını vakaya özel yönetmek.


## 12. `public-14` ilk teşhis (bugüne kadar hiç çalışılmamıştı)

Shard 1 günlüğü, tek değil **iki** knockout bütçe reddi gösteriyor — biri
legacy denemesinde, biri retry'da (`alpha_pipeline_retry.legacy_first`
`raise retry_error from first_error`):

| aşama | bayt | bütçe | oran |
|---|---|---|---|
| legacy (first_error) | 1 737 177 | 1 083 525 | **1,60×** |
| retry (bildirilen) | 1 800 596 | 1 273 782 | **1,41×** |

`apply_candidate_geometry_knockout` (`alpha_candidate_knockout.py:507`) birden
çok kodlamayı deneyip `last_budget_error`'ı saklıyor; yani bildirilen sayı en
son denemenin sayısı.

**Neden önemli:** 1,41× oranı, beş vakanın **bütçeye en yakını**
(`public-15` 2,29× / 2,55×). Yani kapanmaya en yakın aday bu olabilir.

### Ölçülmemiş ama somut ipucu: `%.12g` koordinatlar

`alpha_candidate_knockout.py:254` clipPath dikdörtgenlerini şöyle yazıyor:

```python
"x": f"{view_x + x * sx:.12g}",   "y": f"{view_y + y * sy:.12g}",
"width": f"{width * sx:.12g}",    "height": f"{height * sy:.12g}",
```

`%.12g` koordinat başına 13 karaktere kadar çıkabilir (`123.456789012`).
Bu, `public-15`'te ölçülen tamsayı rect'lerin (~46 B/rect) çok üstünde bir
maliyet demek.

⚠️ İki uyarı, ikisi de bugünkü hatalardan öğrenildi:
1. **Rect'lerin 1,8 MB'ı domine ettiği ÖLÇÜLMEDİ.** Önce bayt ayrıştırması
   yapılmalı (`public-15`'te olduğu gibi ledger alanı ekleyip canlı koşudan
   okumak işe yaradı — madde 11.1).
2. Hassasiyet düşürmek **serileştirme değil kalite** değişikliğidir; geometri
   alt-piksel kayar. Kalite kapılarıyla ölçülmeli, "bayt düştü" yeterli değil.
   Ayrıca `test_alpha_clip_encoding.py:142` clipPath'ten `<rect .../>` regex'le
   çekiyor; kodlama değil ama **biçim** değişikliği o testi etkileyebilir.

### Sıradaki adım (öneri)

`public-15`'te işe yarayan deseni tekrarla: knockout ledger'ına bayt
ayrıştırması ekle (rect baytı / diğer marküp / koordinat karakter sayısı),
canlı korpustan oku, sonra hassasiyet seçeneklerini **ölçülmüş** taban
üzerinde değerlendir. Tahminle başlama.


### 12.1 ÖLÇÜLDÜ — `public-14` bayt ayrıştırması (canlı korpus)

Tanılama satırı `4f6360b` head'inde koştu (run `31911453085`, shard 1):

```
encoding=quantized_128  parent_bytes=424594  added_bytes=1376002
rects=16712  clips=127  uses=127  added_bytes_per_rect=82.3
```

İki aşamada da `added_bytes` ve `rects` **aynı** (1 376 002 / 16 712); yalnız
`parent_bytes` değişiyor (361 175 → 424 594), bütçe farkı oradan geliyor.

**Hipotez doğrulandı:** 82,3 B/rect, `public-15`'te ölçülen tamsayı rect
maliyetinin (~46 B/rect) **1,8 katı**. `%.12g` koordinat biçimi baytı
gerçekten domine ediyor.

### Hedef sayısal olarak net

| ölçüm | değer |
|---|---|
| eklenebilir pay (`limit - parent`) | 849 188 B |
| mevcut eklenen | 1 376 002 B (**1,62×**) |
| gerekli düşüş | **%38,3** |
| mevcut B/rect | **82,3** |
| **gerekli B/rect** | **50,8** |
| `public-15` tamsayı referansı | ~46,0 → **hedefin ALTINDA** |

Yani koordinat biçimini tamsayı mertebesine indirmek `public-14`'ü bütçeye
**sokabilir**. Beş vaka içinde kapanmaya en yakın olan bu.

### ⚠️ Uygulamadan önce

1. **Hassasiyet düşürmek serileştirme değil KALİTE değişikliğidir.** Geometri
   alt-piksel kayar; alpha IoU/MAE kapılarıyla ölçülmeli. "Bayt düştü"
   yeterli gerekçe değil.
2. `alpha_candidate_knockout.py:254`'teki yorum `%.12g`'yi **bilinçli**
   seçtiğini söylüyor: *"Resvg supports a stricter clipPath subset than Cairo.
   Emit exact user-space rectangles directly instead of nesting a transformed
   group inside clipPath so both evaluator renderers agree."* Yani kısa tamsayı
   koordinatları **dönüşümlü grup** ile almak bu gerekçeyle reddedilmiş.
   Kalan yol, dönüşüm eklemeden **basamak sayısını** kısaltmak (ör. `%.4g`
   yerine sabit ondalık ya da 1/64 px yuvarlama).
3. `test_alpha_clip_encoding.py:142` clipPath'ten `<rect .../>` regex'le
   çekiyor; biçim değişikliği o sözleşmeyi etkileyebilir, önce bakılmalı.
4. 46,0 referansı **başka bağlamdan** (paint-deficit destek katmanı, tamsayı
   ızgara). Garanti değil, yalnız hedefin ulaşılabilir olduğunun göstergesi.

## 13. Uçtan uca durum ölçümü (yerel, `class_reklam` fixture)

`engine/regression/fixtures/class_reklam.png` (1001×538, 36 657 B), `mode=auto`.

| katman | sonuç |
|---|---|
| **1 — çekirdek** | `refine` **applied=False**; `edge_cleanup` **applied=False, journal=rolled_back** |
| **2 — alfa sonlandırma** | ✅ başarılı, `source_alpha_vector_mask` ile kapanıyor |
| **3 — export** | ✅ **5/5 format**, 1,9 s |

**Toplam süre: 388 s** (tek küçük logo için).

### Bulgu 1 — `edge_cleanup` kalite yüzünden değil, ÖLÇÜLEMEDİĞİ için geri alınıyor

```
applied=False  journal=rolled_back  fidelity=91.69  absorbed=0
reasons=['required_metric_unmeasured', 'alpha_stage_metrics_incomplete']
```

Aşama koşuyor, işini yapıyor, sonra **gerekli metrik ölçülemediği için** geri
alınıyor. Bu bir kalite reddi değil, ölçüm zincirinde boşluk. Kapı hiçbir zaman
geçemeyeceği için aşama pratikte ölü ağırlık. Deponun "ölçüm-kapılı iyileştirme"
felsefesiyle doğrudan çelişen tek yer burası: kapı ölçemiyorsa iyileştirme
hiç değerlendirilemiyor.

### Bulgu 2 — `refine` de uygulanmıyor

`refine_info.applied=False`. Reddin ölçülmüş mü yoksa aday hiç üretilmemiş mi
olduğu ayrımı **yapılmadı**; `refine_best` (`pipeline.py:672`) incelenmeli.

### Bulgu 3 — `refine_cache` %2,8 isabet

`render_calls=36, hits=1, misses=35, evictions=33` (cls: 42/5/37/31).
Önbellek sürekli tahliye edip yeniden hesaplıyor. Kapasite iş kümesinin
altında görünüyor. 388 s'nin ölçülebilir bir parçası olabilir ve düzeltmesi
kalite kapılarına dokunmaz.

### Katman 3 sağlıklı — yanlış alarma dikkat

pdf/eps ilk koşuda "render edilemedi" verdi; sebep **yerel ortamda
`cairosvg`/`svglib` kurulu olmamasıydı** (ikisi de `engine/requirements.txt`'te
var, brifingdeki kısa kurulum listesi onları atlıyor). Kurulunca:

```
svg 267 523 B | pdf 35 657 B | eps 209 377 B | dxf 768 766 B | png 26 453 B
```

⚠️ Brifing madde 5'teki yerel kurulum komutu eksik; `cairosvg svglib reportlab`
eklenmeli yoksa export sahte hata verir.

### Sıradaki en iyi hedef

**Bulgu 1.** Ölçüm boşluğu, kalite kapısına dokunmadan kapatılabilir ve
kapanınca `edge_cleanup` ilk kez gerçekten değerlendirilebilir hâle gelir.
`required_metric_unmeasured` / `alpha_stage_metrics_incomplete` kodlarının
kaynağı izlenmeli.

### 13.1 KÖK NEDEN BULUNDU — aşama/gereksinim uyuşmazlığı (yerelde, CI'sız)

`transform_journal.py:312-313`:

```python
capture_render = self._measurement_stage_id == "restore_source_dimensions"
measure_alpha  = capture_render
```

Alfa **yalnızca** `restore_source_dimensions` aşamasında ölçülüyor; diğer tüm
aşamalarda (`edge_cleanup` dahil) `measure_alpha=False`.

Ama `alpha_fidelity`'yi `required_unmeasured` listesinden çıkaran satır
(`:191-194`) **`if alpha_required:` bloğunun içinde**. Blok atlanınca:

1. `alpha_fidelity` listede kalır → **`required_metric_unmeasured`** (`:394`)
2. `_render_alpha` hiç yazılmaz → `_alpha_comparison` **None** (`:346`) →
   **`alpha_stage_metrics_incomplete`** (`:407`)

`_decide` ise `alpha_fidelity`'yi **aşamadan bağımsız** zorunlu sayıyor (`:405`).

**Sonuç yapısal:** `alpha_fidelity` zorunlu metriklerdeyken
`restore_source_dimensions` dışındaki **her** journal aşaması, kalitesi ne
olursa olsun bu iki kodla reddedilir. Ara sıra olan bir ölçüm boşluğu değil,
her koşuda kesin.

Gözlenen çift kod tam da bu imzadır (madde 13):
`reasons=['required_metric_unmeasured', 'alpha_stage_metrics_incomplete']`.

### Renderer sağlam — yanlış alarma düşmedim

Önce PDF/EPS'teki gibi bir ortam eksiği olabileceğinden şüphelendim ve
kontrol ettim: `render_svg_to_rgba` üretilen SVG'de **çalışıyor**
(1001x538 ve 512x275'te doğru şekil). Yani sebep renderer yokluğu değil,
`measure_alpha` bayrağının aşamaya bağlı olması.

### ⚠️ Kasıtlı mı, kusur mu — KARAR VERİLMEDİ

- **Kusur lehine:** kod adları bir ölçüm başarısızlığını anlatıyor
  (`..._unmeasured`, `..._incomplete`), bir politika kararını değil. Politika
  olsaydı `alpha_stage_not_permitted` gibi bir kod beklenirdi.
- **Kasıt lehine:** alfa taşıyan artefaktı `edge_cleanup`'ın değiştirmemesi
  bilinçli bir güvenlik olabilir. `:178` yorumu fail-closed davranışı
  savunuyor (ama renderer yokluğu bağlamında).

### İki olası yön (ikisi de ölçülmeli)

1. **Alfayı `edge_cleanup` aşamasında da ölç.** Doğru sonucu verir ama
   aşama başına bir render daha ekler — boru hattı zaten 388 s ve madde 10.2'de
   render/kontur maliyetinin baskın olduğu ölçüldü. Süreyi kötüleştirebilir.
2. **`alpha_fidelity`'yi ölçülmediği aşamalarda zorunlu saymamak.** Ucuz, ama
   kapıyı gevşetir; "eşik gevşetmek çözüm değildir" kuralına (madde 5) yakın
   durduğu için sahibiyle konuşulmalı.

⚠️ Hiçbiri uygulanmadı. Bu, oturumdaki ilk **tamamen yerel** kök neden;
doğrulaması `class_reklam` ile 388 s'de tekrarlanabilir, CI gerekmez.

### 13.2 SEÇENEK 1 DENENDİ ve ÇÜRÜTÜLDÜ — `measure_alpha` ölçüm tabanını değiştiriyor

Uygulanan tek satır:

```python
measure_alpha = capture_render or ("alpha_fidelity" in self.required_metrics)
```

Sonuç: `class_reklam` **tamamen düştü** (önceden başarılıydı):

```
RuntimeError: source_alpha_candidate_knockout_iou_gate_failed:0.911869<0.995
```

Geri alındı; motor yine yalnız knockout tanılamasını taşıyor.

### Neden — `measure_alpha` masum bir bayrak değil (`transform_journal.py:136-148`)

```python
if alpha_required:
    render_rgba = _source_truth.render_svg_to_rgba(path, w, h)
    if render_rgba is not None:
        rnd = _source_truth.composite_rgba(render_rgba, 255)   # RGB buradan
    else:
        rnd = render_svg_to_rgb(path, w, h)
else:
    rnd = render_svg_to_rgb(path, w, h)                        # normalde buradan
```

`measure_alpha=True` yapıldığında **RGB görüntüsünün kaynağı değişiyor**:
`app.fidelity.render_svg_to_rgb` yerine
`source_truth.composite_rgba(render_svg_to_rgba(...), 255)`.

Bunlar **farklı renderer yolları**; pikselleri birebir aynı değil. Ve `ssim`,
`edge_f1_1px`, `seam_ratio`, `component_delta`, `hole_delta` kapılarının
**hepsi** bu `rnd` üzerinden hesaplanıyor. Yani tek satırlık "fazladan alfa
ölç" değişikliği, **her aşamadaki her görsel kapının ölçüm tabanını** sessizce
değiştiriyor. Aşağı akıştaki 0,9119 IoU çöküşü bununla tutarlı.

### Bunun anlamı

Seçenek 1 küçük bir düzeltme **değil**. `alpha_fidelity`yi her aşamada ölçmek
isteniyorsa, önce iki renderer yolunun aynı pikselleri verdiği (ya da tüm
eşiklerin yeni tabana göre yeniden kalibre edildiği) **gösterilmelidir**.
Aksi hâlde kapılar sessizce başka bir zeminde karar verir.

⚠️ Bu, madde 11'deki dersin aynısının başka bir kılığı: bir bayrağın
"yalnızca ölçüm ekler" göründüğü yerde, gerçekte üretim yolunu değiştiriyor.
Bu depoda bir bayrağı açmadan önce **onun tükettiği tüm dalları** okuyun.

### Geriye kalan yön

Seçenek 2 (`alpha_fidelity`yi ölçülmediği aşamalarda zorunlu saymamak) artık
tek makul aday, çünkü ölçüm tabanına dokunmuyor. Ama kapıyı gevşetme riski
taşıdığından madde 5 gereği sahibiyle konuşulmalı. Üçüncü bir yol da var ve
ölçülmedi: iki renderer yolunu **aynı** hâle getirmek (o zaman seçenek 1
güvenli olurdu).

### 13.3 İKİ RENDERER YOLU ÖLÇÜLDÜ — fark, kapı toleransının 3,6 katı

`app.fidelity.render_svg_to_rgb` ile
`source_truth.composite_rgba(render_svg_to_rgba(...), 255)` aynı SVG'de,
journal'ın kendi ölçeğinde (512 max_side) karşılaştırıldı. Dört ayrı çıktıda
tutarlı sonuç:

| SVG | farklı piksel | max kanal | **SSIM (iki render arası)** | edge_f1 |
|---|---|---|---|---|
| `e2e.svg` (267 KB) | 15 733 / 140 800 (%11,2) | 36 | **0,998152** | 1,000000 |
| `geo_clean.svg` | 13 651 (%9,7) | 43 | **0,998183** | 1,000000 |
| `geo_contour.svg` | 13 847 (%9,8) | 42 | **0,998155** | 1,000000 |
| `geo_detail.svg` | 14 218 (%10,1) | 43 | **0,998368** | 1,000000 |

### Karar veren sayı

```
iki renderer yolu arasindaki SSIM farki : ~0,0018
journal SSIM kapisi toleransi           :  0,0005   (transform_journal.py:452)
oran                                    :  3,6x
```

**Renderer yolunu değiştirmek, kapının tüm toleransının 3,6 katı büyüklüğünde
bir kayma üretiyor.** Bu yüzden `measure_alpha` bayrağını açmak (madde 13.2)
her aşamada sahte `ssim_regression` üretebilir ve `class_reklam`'ı düşürmesi
bununla tam tutarlı.

`edge_f1_1px` iki yolda **birebir 1,000000** — yani kenar geometrisi aynı;
fark tamamen kenar yumuşatma/iç tonlamada (ortalama kanal farkı ~0,53,
piksellerin ~%10'u).

### Üç seçeneğin güncel durumu

1. **Her aşamada alfa ölç** — ÇÜRÜTÜLDÜ (13.2). Ölçüm tabanını 3,6× tolerans
   kadar kaydırıyor.
2. **Ölçülmediği aşamalarda zorunlu sayma** — tek uygulanabilir aday. Ölçüm
   tabanına dokunmaz. Riski: kapıyı gevşetir (madde 5), sahibiyle konuşulmalı.
3. **İki renderer yolunu aynı hâle getir** — artık **nicelendi**: piksellerin
   ~%10'unda AA farkı, 0,0018 SSIM. Küçük bir uyum işi değil; kenarlar zaten
   aynı olduğundan fark yalnız AA/kompozit davranışında, ama büyüklüğü kapı
   toleransını aşıyor.

### ⚠️ DÜZELTME — "yollar karışık kullanılıyor" uyarım YANLIŞTI

İlk yazdığım genel uyarı ("aynı artefaktı biri parent diğeri candidate için
kullanan her karşılaştırma sahte regresyon üretebilir") **koddan denetlenince
çürüdü**:

`_measurement_stage_id` aşama başına bir kez set ediliyor
(`transform_journal.py:538-544`) ve parent ile candidate ölçümlerinin **ikisi
de** o pencerede yapılıyor. Dolayısıyla her karşılaştırmada iki taraf da aynı
`measure_alpha` bayrağını, yani **aynı renderer yolunu** kullanır. Sistematik
parent/candidate uyuşmazlığı **yoktur**.

Geriye yalnız dar bir açık kalır: `render_svg_to_rgba` taraflardan birinde
`None` dönerse o taraf `render_svg_to_rgb`'ye düşer ve karşılaştırma iki farklı
yoldan gelir. Bu **asimetrik renderer başarısızlığı** gerektirir; olağan durum
değildir.

**Bunun sonucu:** RFV-3B'de düşen vakaların (`public-05`, `public-15` vb.)
`seam_regression`'larını ölçüm tabanı tutarsızlığına bağlamak **yanlış olur**;
o redler büyük olasılıkla gerçektir. `public-04`'ün `edge_f1_regression`'ı
zaten bağımsız olarak bunu destekliyor: ölçtüğüm gibi `edge_f1` iki renderer
yolunda birebir 1,000000, yani bu metrik farka tamamen duyarsız.

**Geçerli kalan tek sonuç:** renderer yolunu *değiştirmek* (yani
`measure_alpha` bayrağını açmak) 0,0018 SSIM kayması üretir ve kapı toleransı
0,0005 olduğundan tehlikelidir. Seçenek 1'in neden `class_reklam`'ı düşürdüğü
budur — mevcut redleri açıklamaz.

### 13.4 `refine_cache` kapasitesi SUÇSUZ — ölçüldü, kapasite artırılmadı

`_DEFAULT_MAX_RENDERS = 2` (`refine_cache.py:38`) ve gözlenen isabet %2,8
olduğu için "önbellek thrash ediyor" hipotezini kurdum. **Kapasiteyi
artırmadan önce** anahtar izini çıkarıp aynı LRU politikasıyla simüle ettim.

`class_reklam`, 455,9 s koşu:

| ölçüm | değer |
|---|---|
| render çağrısı | 68 |
| benzersiz anahtar | **62** |
| tekrar (sonsuz kapasitede MAKS isabet) | **6** → tavan **%8,8** |
| aynı içeriğin birden fazla boyutta istenmesi | **0** |

| kapasite | 2 | 4 | 8 | 16 | 32 | 64 | sonsuz |
|---|---|---|---|---|---|---|---|
| isabet | 2 | 2 | 2 | 4 | **6** | 6 | 6 |

**Karar: kapasite artırılmadı.** Sonsuz önbellek bile 68 çağrının yalnız
6'sını kurtarıyor; 2→32 yalnız **4 ek isabet** getirir. Karşılığında 32 girdi
~13-51 MB bellek ve CLAUDE.md motor tepe kullanımının 826 MB olduğunu,
512 MB planların OOM olduğunu söylüyor. Belirsiz %6 render kazancı için
bellek riski alınmaz.

**Asıl bulgu:** yavaşlık "aynı şeyi tekrar render etmek" değil, **62 ayrı
render yapmak**. Önbellek doğru çalışıyor; iş yükü gerçekten benzersiz.

### Kaldıraç nerede

1. Render **sayısını** azaltmak — ama aday/varyant sayısı kalite kapılarına
   bağlı, dokunmak riskli.
2. Render'ın **kendisini** ucuzlatmak — madde 10.2'deki ölçümle uyumlu
   (`cv2.findContours` patolojik girdide 98 s).

⚠️ Sonraki oturum: "önbelleği büyütelim" fikrini tekrar açma, ölçüldü ve
tavanı %8,8. Ölçüm betiği deseni saklanmaya değer: anahtar izini çıkarıp
kapasiteleri simüle etmek, kod değiştirmeden tavanı verdi.

### 13.5 RENDER DARBOĞAZ DEĞİL — ölçüldü (%16), sürenin %84'ü başka yerde

Renderer fonksiyonlarının kendisi sarmalandı (`app.fidelity.render_svg_to_rgb`,
`app.source_truth.render_svg_to_rgba`), böylece mutator + journal + alfa
merdiveni dahil **tüm** render'lar yakalandı. `class_reklam`, 452,0 s:

| ölçüm | değer |
|---|---|
| boru hattı toplam | **452,0 s** |
| **render toplam** | **72,8 s → %16,1** |
| render çağrısı | **275** (`refine_cache` yalnız 68'ini görüyor) |
| medyan / p90 / maks | 55,2 ms / 150,8 ms / **21 564 ms** |

**Sürenin %84'ü (≈379 s) render DIŞINDA.** "Render'ı ucuzlatalım" yönü
(madde 13.4'te önerilmişti) bu ölçümle **çürüdü**.

### Render içindeki patoloji (küçük ama gerçek)

| süre | boyut | svg |
|---|---|---|
| 21 564 ms | 1001×538 | 254 032 B |
| 20 949 ms | 1001×538 | 253 983 B |
| 6 665 ms | 512×275 | 254 032 B |
| 6 638 ms | 512×275 | 253 983 B |
| 168 ms | 1001×538 | 246 472 B |

Render süresinin **%50'si 2 çağrıda** (çağrıların %1'i). Tipik render 55 ms
iken bunlar 21 s — **380×**. Aynı SVG 512'de 6,6 s, 1001'de 21 s. Bunlar alfa
maskeli ~254 KB çıktılar; dev maske geometrisi renderer'ı boğuyor.

⚠️ Ama bu ikisini tamamen ortadan kaldırmak bile 452 s'nin yalnız **%9**'unu
kazandırır. Tek başına hedef değil.

### Sıfır RGBA çağrısı

275 çağrının **tamamı** RGB yolundan geçti; `render_svg_to_rgba` hiç
çağrılmadı. Yani bu koşuda `measure_alpha` hiçbir yerde açılmamış —
madde 13.1'deki bulguyla (alfa yalnız `restore_source_dimensions`ta ölçülür)
tutarlı ve alfa yolunun üretimde ne kadar dar bir yerde çalıştığını gösteriyor.

### Sıradaki adım

**379 saniyenin nerede geçtiği bilinmiyor.** Tahmin edilmemeli;
`cProfile` ile tek koşuda fonksiyon bazında döküm alınmalı. Aday şüpheliler
(ölçülmedi): VTracer aday üretimi, shape fitting, `findContours` (madde 10.2'de
patolojik girdide 98 s ölçüldü), skorlama.

### 13.6 cProfile — 526 s nerede geçiyor (ve madde 13.5'in DÜZELTMESİ)

⚠️ **Madde 13.5'teki "render %16" rakamı YANLIŞTI.** Orada yalnız
`app.fidelity.render_svg_to_rgb` ve `source_truth.render_svg_to_rgba`
sarmalanmıştı (275 çağrı). cProfile gerçek sayıyı veriyor:
`resvg_py.svg_to_bytes` **695 çağrı, 171,4 s = %32,6**. Yaklaşık 420 çağrı
o iki sarmalayıcıyı atlayarak doğrudan resvg'ye gidiyor. Ders: bir maliyeti
sarmalayıcıyla ölçerken **tüm çağrı yollarının** oradan geçtiği doğrulanmalı.

`class_reklam`, profil altında 526,4 s (taban 452 s, ek yük ~%16):

**Kümülatif (çağrı ağacı):**

| aşama | çağrı | süre | pay |
|---|---|---|---|
| `apply_candidate_painter_reconstruction` | 2 | **237,2 s** | %45 |
| `evaluate_final_svg_bytes` | **37** | **191,3 s** | %36 |
| `_evaluate_phase` (painter) | 9 | 189,6 s | %36 |
| `validate_alpha_reconstruction_contract` | 11 | 149,4 s | %28 |

**Kendi süresi (tottime):**

| işlev | çağrı | süre |
|---|---|---|
| `resvg_py.svg_to_bytes` | 695 | **171,4 s** |
| `numpy.ufunc.reduce` | 56 569 | 51,3 s |
| `ndarray.nonzero` | 24 376 | 24,9 s |
| `_thread.lock.acquire` | 98 | 20,9 s |
| `scipy.ndimage.correlate1d` | 6 460 | 18,0 s |
| `palette_ops.classify_features` | 524 | 17,2 s (küm. 54,6) |
| `numpy.linalg.norm` | 11 600 | 15,0 s (küm. 45,5) |
| `ciede2000` | 74 | 9,5 s |
| `_ssim` | 646 | 6,7 s (küm. 26,5) |
| `alpha_mask_contour.add` | **3 959 860** | 5,3 s |

Toplam **250 308 803 fonksiyon çağrısı**.

### Yorum: yavaşlık hata değil, tasarımın bedeli

Alfa painter merdiveni 526 s'nin 237'sini tüketiyor. Maliyetin kaynağı:
her aday için resvg ile rasterleştirme (695 render) **ve tam final artifact
değerlendirmesinin 37 kez** koşturulması (ciede2000 + SSIM + kmeans + palet
sınıflandırma). Bu, "ölçüm-kapılı iyileştirme" felsefesinin doğrudan bedeli.

Hızlandırmanın iki yolu var ve ikisi de bedava değil:
1. **Aday sayısını azaltmak** — kalite kapılarına dokunur, riskli.
2. **Değerlendirmeyi ucuzlatmak** — `evaluate_final_svg` 37 kez tam suite
   koşuyor; erken eleme (ucuz metrikle önce ele, pahalıyı sonra) mümkün
   görünüyor ama **ölçülmedi** ve sonucu değiştirmemesi kanıtlanmalı.

⚠️ Profil dosyası saklandı: `scratchpad/pipeline.prof` (pstats ile açılabilir).
Bu konteyner geçici; kalıcı analiz gerekiyorsa yeniden üretilmeli.

### 13.7 ERKEN ELEME TAVANI = 0,0 s — fikir ölçümle öldü

`evaluate_final_svg_bytes` sarmalanıp her çağrının süresi + `hard_fail_codes`
kaydedildi (`class_reklam`, 450,5 s):

| ölçüm | değer |
|---|---|
| `evaluate_final_svg` | **134,4 s** (37 çağrı, boru hattının **%29,8**'i) |
| çağrı başına | 3 632 ms |
| hard fail **yok** (geçen) | **0** |
| hard fail var (düşen) | **37** |
| **yalnız ucuz kodla düşen** | **0** → **TAVAN 0,0 s** |

Her reddin içinde en az bir pahalı algısal metrik var. **Ucuz kapıları öne
almak hiçbir şey kazandırmaz.** Madde 13.6'da önerdiğim "erken eleme" yönü
kapandı; sonraki oturum tekrar açmasın.

### Beklenmedik gözlem: 37 değerlendirmenin TAMAMI düşüyor

```
37x  alpha_white_ssim_below_min      <- istisnasiz her cagri
37x  alpha_black_ssim_below_min      <- istisnasiz her cagri
37x  alpha_checker_ssim_below_min    <- istisnasiz her cagri
36x  topology_component_delta
31x  seam_gap
30x  ssim_below_min
23x  alpha_{white,black,checker}_mae_above_max
15x  topology_hole_delta
```

Hiçbir aday `evaluate_final_svg`'den temiz geçmiyor, buna rağmen boru hattı
**başarıyla** bitiyor (`source_alpha_vector_mask` kazanıyor). Yani bu
değerlendirmenin verdict'i aday seçiminde son söz değil.

⚠️ **Yorumu sınanmadı.** Üç alfa-SSIM kapısının her adayda düşmesi iki şeyden
biri olabilir: (a) gerçekten hiçbir aday yeterli değil, (b) bu üç kapıda
sistematik bir sorun var (ör. yanlış arka plan kompoziti ya da ölçek). İkisi
ayırt edilmedi. Ayırt etmenin ucuz yolu: tek bir adayın
`alpha_white_ssim` değerini ve eşiğini okuyup, aynı adayı gözle
render edip karşılaştırmak.

Bu, RFV-3B'de `public-04/05/15`'in `ssim_regression`/`seam_regression` ile
düşmesiyle **aynı aileden** görünüyor — ama madde 13.3'ün düzeltmesinden sonra
artık renderer tabanı tutarsızlığına bağlanamaz. Bağımsız bir sorgu gerektirir.

### Zaman haritasının özeti (madde 13.5-13.7 birlikte)

| kalem | süre | pay |
|---|---|---|
| `resvg_py.svg_to_bytes` (695 çağrı) | 171,4 s | %33 |
| `evaluate_final_svg` (37 çağrı) | 134,4 s | %30 |
| alfa painter merdiveni (kümülatif) | 237,2 s | %45 |

İlk ikisi büyük ölçüde üçüncünün **içinde**. Yani boru hattı süresinin
yarısına yakını alfa aday merdiveninde, ve orada da maliyet rasterleştirme +
tam değerlendirmenin tekrar tekrar koşturulması.
