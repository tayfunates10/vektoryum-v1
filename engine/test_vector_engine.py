"""Vektoryum API - motor doğrulama test scripti.

Çalıştırma (engine klasöründen):

    .\\.venv\\Scripts\\python.exe test_vector_engine.py

12 kontrol yapar ve başarısızlıkta çıkış kodu 1 döner.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# engine klasörünü import yoluna ekle (app paketi için)
ENGINE_DIR = Path(__file__).resolve().parent
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

import numpy as np
from PIL import Image

print("Python:", sys.version)

_results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    _results.append((name, bool(condition), detail))
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" :: {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Sentetik test görselleri
# ---------------------------------------------------------------------------
def make_geometric_logo() -> Image.Image:
    """Siyah-beyaz-kırmızı, sert kenarlı, az renkli geometrik logo."""
    w, h = 800, 600
    arr = np.full((h, w, 3), 255, dtype=np.uint8)  # beyaz zemin
    # siyah dış çerçeve (kalın, içi boş)
    arr[60:540, 60:90] = 0
    arr[60:540, 710:740] = 0
    arr[60:90, 60:740] = 0
    arr[510:540, 60:740] = 0
    # kırmızı dolu blok
    arr[160:360, 180:420] = (255, 0, 0)
    # siyah dolu blok (monogram benzeri)
    arr[160:440, 470:660] = 0
    return Image.fromarray(arr, "RGB")


def make_color_logo() -> Image.Image:
    """Çok renkli + gradyanlı AI logo benzeri görsel."""
    w, h = 800, 600
    yy, xx = np.mgrid[0:h, 0:w]
    r = (xx / w * 255).astype(np.uint8)
    g = (yy / h * 255).astype(np.uint8)
    b = ((xx + yy) / (w + h) * 255).astype(np.uint8)
    arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
    palette = [(220, 30, 30), (30, 160, 60), (40, 80, 200), (240, 200, 20), (150, 40, 160), (240, 130, 20)]
    for i, color in enumerate(palette):
        cy = 150 + (i % 2) * 250
        cx = 120 + (i % 3) * 250
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < 90 ** 2
        arr[mask] = color
    return Image.fromarray(arr, "RGB")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="vec_test_"))

    # 1. app.main import
    try:
        from app.main import ALLOWED_MODES, app  # noqa: F401
        check("1. app.main import", True)
    except Exception as e:  # noqa: BLE001
        check("1. app.main import", False, repr(e))
        return _summary()

    # 2. allowed_modes
    check("2. geometric_logo in ALLOWED_MODES", "geometric_logo" in ALLOWED_MODES, f"{ALLOWED_MODES}")

    src_png = tmp / "orig.png"
    make_geometric_logo().save(src_png)

    # 3 + 4. analyzer sınıflandırma
    try:
        from app.analyzer import analyze_image_from_mem
        geo_report = analyze_image_from_mem(make_geometric_logo())
        check("3. analyzer -> geometric_logo (b/w/red)",
              geo_report["recommended_mode"] == "geometric_logo",
              f"recommended={geo_report['recommended_mode']}, colors={geo_report['estimated_color_count']}, "
              f"edge={geo_report['edge_density']}, likely_geo={geo_report['likely_geometric_logo']}")

        color_report = analyze_image_from_mem(make_color_logo())
        check("4. analyzer -> logo_color (multi-color)",
              color_report["recommended_mode"] == "logo_color",
              f"recommended={color_report['recommended_mode']}, colors={color_report['estimated_color_count']}")
    except Exception as e:  # noqa: BLE001
        check("3. analyzer -> geometric_logo (b/w/red)", False, repr(e))
        check("4. analyzer -> logo_color (multi-color)", False, repr(e))

    # 5. build_vector_candidates
    try:
        from app.vector_engines import build_vector_candidates
        cands = build_vector_candidates("geometric_logo")
        check("5. build_vector_candidates('geometric_logo') >= 4", len(cands) >= 4, f"{list(cands.keys())}")
    except Exception as e:  # noqa: BLE001
        check("5. build_vector_candidates('geometric_logo') >= 4", False, repr(e))

    # 6 + 8. geometry_cleanup
    try:
        from app.geometry_cleanup import cleanup_svg_geometry
        check("6. geometry_cleanup import", True)
        check("8. cleanup_svg_geometry mevcut", callable(cleanup_svg_geometry))
    except Exception as e:  # noqa: BLE001
        check("6. geometry_cleanup import", False, repr(e))
        check("8. cleanup_svg_geometry mevcut", False, repr(e))

    # 7. _path_efficiency_score(22, 4, "geometric_logo") == 100
    try:
        from app.scoring import _path_efficiency_score
        val = _path_efficiency_score(22, 4, "geometric_logo")
        check("7. _path_efficiency_score(22,4,geometric_logo)==100", val == 100.0, f"={val}")
    except Exception as e:  # noqa: BLE001
        check("7. _path_efficiency_score(22,4,geometric_logo)==100", False, repr(e))

    # 9. vectorize_geometric_contours_to_svg mevcut
    try:
        from app.vector_engines import vectorize_geometric_contours_to_svg
        check("9. vectorize_geometric_contours_to_svg mevcut", callable(vectorize_geometric_contours_to_svg))
    except Exception as e:  # noqa: BLE001
        check("9. vectorize_geometric_contours_to_svg mevcut", False, repr(e))

    # 10. Algısal sadakat: render yoksa (CairoSVG eksik) sistem çökmez -> None döner
    try:
        from app.fidelity import score_svg_fidelity
        svg = tmp / "tiny.svg"
        svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
                       '<rect width="10" height="10" fill="#000"/></svg>', encoding="utf-8")
        fid = score_svg_fidelity(svg, src_png)
        ok = fid is None or (isinstance(fid, dict) and "fidelity_score" in fid)
        check("10. Sadakat: render yoksa çökmez (None döner)", ok,
              f"fidelity={'dict' if isinstance(fid, dict) else fid} (None = cairo yok, beklenen)")
    except Exception as e:  # noqa: BLE001
        check("10. Sadakat: render yoksa çökmez (None döner)", False, repr(e))

    # 11. Potrace yoksa fallback
    try:
        from app.vector_engines import get_potrace_path, run_candidate
        potrace = get_potrace_path()
        if potrace is None:
            raised = False
            try:
                run_candidate("potrace", src_png, tmp / "p.svg", {"params": {}})
            except FileNotFoundError as fe:
                raised = "potrace not found" in str(fe)
            check("11. Potrace yoksa düzgün fallback", raised, "FileNotFoundError('potrace not found') beklenir")
        else:
            check("11. Potrace yoksa düzgün fallback", True, f"potrace bulundu: {potrace}")
    except Exception as e:  # noqa: BLE001
        check("11. Potrace yoksa düzgün fallback", False, repr(e))

    # 12. AutoTrace yoksa fallback/warning + skeleton fallback çalışıyor
    try:
        from app.vector_engines import get_autotrace_path, run_candidate
        from app.preprocess import preprocess_for_mode
        autotrace = get_autotrace_path()
        if autotrace is None:
            raised = False
            try:
                run_candidate("autotrace", src_png, tmp / "a.svg", {"params": {"centerline": True}})
            except FileNotFoundError as fe:
                raised = "autotrace not found" in str(fe)
            pp, _ = preprocess_for_mode(src_png, "centerline", tmp)
            skel_ok = True
            try:
                run_candidate("opencv_skeleton", pp, tmp / "skel.svg", {"params": {}})
            except Exception:  # noqa: BLE001
                skel_ok = False
            check("12. AutoTrace yoksa fallback/warning", raised and skel_ok,
                  f"autotrace_error_raised={raised}, skeleton_fallback_ok={skel_ok}")
        else:
            check("12. AutoTrace yoksa fallback/warning", True, f"autotrace bulundu: {autotrace}")
    except Exception as e:  # noqa: BLE001
        check("12. AutoTrace yoksa fallback/warning", False, repr(e))

    # 13. HED derin kenar modeli OPSİYONEL: model yokken compute_edge_map
    # güvenle None döner (çökme yok); varken geçerli 0..1 haritası üretir
    try:
        import importlib
        import os
        import app.dl_segmentation as dl

        arr = np.asarray(make_geometric_logo())
        with_model = dl.compute_edge_map(arr)
        ok_with = with_model is None or (
            isinstance(with_model, np.ndarray)
            and with_model.shape == arr.shape[:2]
            and 0.0 <= float(with_model.min()) and float(with_model.max()) <= 1.0
        )

        old_proto = os.environ.get("HED_PROTO_PATH")
        os.environ["HED_PROTO_PATH"] = str(tmp / "yok.prototxt")
        importlib.reload(dl)
        without_model = dl.compute_edge_map(arr)
        if old_proto is None:
            os.environ.pop("HED_PROTO_PATH", None)
        else:
            os.environ["HED_PROTO_PATH"] = old_proto
        importlib.reload(dl)

        check("13. HED opsiyonel: yokken None, varken geçerli harita",
              ok_with and without_model is None,
              f"varken={'harita' if isinstance(with_model, np.ndarray) else with_model}, "
              f"yokken={without_model}")
    except Exception as e:  # noqa: BLE001
        check("13. HED opsiyonel: yokken None, varken geçerli harita", False, repr(e))

    # 14. Anlamsal foto imzası: düz beyaz zeminli fotoğraf (zemin-düzgünlüğü
    # kriterini geçtiği için eski tespitin kör noktası) photo_poster'a gitmeli.
    # HED modeli yoksa sinyal kapalıdır; test bilgi notuyla geçer.
    try:
        from app.analyzer import analyze_image_from_mem
        from app.dl_segmentation import is_available

        if is_available():
            h, w = 400, 500
            rng = np.random.default_rng(42)
            yy, xx = np.mgrid[0:h, 0:w]
            base = np.zeros((h, w, 3), np.uint8)
            base[..., 0] = (70 + 130 * xx / w)
            base[..., 1] = (80 + 120 * yy / h)
            base[..., 2] = (120 + 80 * np.sin((xx + yy) / 80))
            photo = np.clip(base.astype(np.float32) + rng.normal(0, 28, base.shape), 0, 255).astype(np.uint8)
            canvas = np.full((600, 800, 3), 255, np.uint8)
            canvas[100:500, 150:650] = photo
            rep = analyze_image_from_mem(Image.fromarray(canvas))
            check("14. HED: beyaz zeminli foto -> photo_poster",
                  rep["recommended_mode"] == "photo_poster" and rep["semantic_photo_like"],
                  f"mode={rep['recommended_mode']}, semantic_photo_like={rep['semantic_photo_like']}")
        else:
            check("14. HED: beyaz zeminli foto -> photo_poster", True,
                  "HED modeli yok; sinyal kapalı (models/fetch_hed.py ile etkinleşir)")
    except Exception as e:  # noqa: BLE001
        check("14. HED: beyaz zeminli foto -> photo_poster", False, repr(e))

    # 15. Curve fairing: küçük açılı C-C eklemi G1'e hizalanır, uç noktalar
    # sabit kalır; keskin köşe (>25 derece) korunur
    try:
        from app.curve_fairing import _parse_subpaths, _serialize_subpaths, count_curve_kinks, fair_subpath

        d = "M0 0 C10 0 20 0 30 0 C40 3.5 50 7 60 10 Z"
        sp = _parse_subpaths(d)[0]
        k_before, _ = count_curve_kinks(d)
        fair_subpath(sp)
        d2 = _serialize_subpaths([sp])
        k_after, _ = count_curve_kinks(d2)
        ends_fixed = sp["segs"][0][-1] == (30.0, 0.0) and sp["segs"][1][-1] == (60.0, 10.0)

        corner = "M0 0 C10 0 20 0 30 0 C30 10 30 20 30 30 Z"  # 90 derece: köşe
        sp_c = _parse_subpaths(corner)[0]
        corner_kept = fair_subpath(sp_c) == 0

        check("15. Curve fairing: kink hizalanır, köşe/uçlar korunur",
              k_before == 1 and k_after == 0 and ends_fixed and corner_kept,
              f"kink {k_before}->{k_after}, uçlar_sabit={ends_fixed}, köşe_korundu={corner_kept}")
    except Exception as e:  # noqa: BLE001
        check("15. Curve fairing: kink hizalanır, köşe/uçlar korunur", False, repr(e))

    # 16. Bütünsel şekil oturtma: daire/elips/dikdörtgen/roundrect tanınır,
    # L-poligon reddedilir (organik şekiller asla zorla değiştirilmez)
    try:
        import math
        from app.shape_fitting import try_fit_whole_shape

        rng = np.random.default_rng(3)
        t = np.linspace(0, 2 * math.pi, 120, endpoint=False)
        circle = np.c_[200 + 80 * np.cos(t), 150 + 80 * np.sin(t)] + rng.normal(0, 0.5, (120, 2))
        ellipse_x, ellipse_y = 120 * np.cos(t), 60 * np.sin(t)
        ca, sa = math.cos(math.radians(30)), math.sin(math.radians(30))
        ellipse = np.c_[300 + ellipse_x * ca - ellipse_y * sa,
                        200 + ellipse_x * sa + ellipse_y * ca] + rng.normal(0, 0.5, (120, 2))
        sq = []
        for (x0, y0), (x1, y1) in [((-100, -60), (100, -60)), ((100, -60), (100, 60)),
                                   ((100, 60), (-100, 60)), ((-100, 60), (-100, -60))]:
            for f in np.linspace(0, 1, 40, endpoint=False):
                sq.append((250 + x0 + (x1 - x0) * f, 250 + y0 + (y1 - y0) * f))
        rect = np.array(sq) + rng.normal(0, 0.4, (160, 2))
        lpts = []
        L = [(0, 0), (100, 0), (100, 40), (40, 40), (40, 100), (0, 100)]
        for i in range(len(L)):
            a, b = np.array(L[i], float), np.array(L[(i + 1) % len(L)], float)
            for f in np.linspace(0, 1, 30, endpoint=False):
                lpts.append(a + (b - a) * f)
        lshape = np.array(lpts)

        star_v = []
        for k in range(5):
            ao = 2 * math.pi * k / 5
            ai = 2 * math.pi * (k + 0.5) / 5
            star_v.append((300 + 150 * math.cos(ao), 300 + 150 * math.sin(ao)))
            star_v.append((300 + 60 * math.cos(ai), 300 + 60 * math.sin(ai)))
        spts = []
        for i in range(len(star_v)):
            a, b = np.array(star_v[i]), np.array(star_v[(i + 1) % len(star_v)])
            for f in np.linspace(0, 1, 14, endpoint=False):
                spts.append(a + (b - a) * f)
        star = np.array(spts) + rng.normal(0, 0.5, (140, 2))

        d_circle = try_fit_whole_shape(circle, True)
        d_ellipse = try_fit_whole_shape(ellipse, True)
        d_rect = try_fit_whole_shape(rect, True)
        d_star = try_fit_whole_shape(star, True)
        d_l = try_fit_whole_shape(lshape, True)
        ok = (
            d_circle is not None and "A" in d_circle
            and d_ellipse is not None and "A" in d_ellipse
            and d_rect is not None
            and d_star is not None and "A" not in d_star
            and d_l is None
        )
        check("16. Bütünsel şekil oturtma: daire/elips/rect/yıldız EVET, L-poligon HAYIR", ok,
              f"daire={bool(d_circle)}, elips={bool(d_ellipse)}, rect={bool(d_rect)}, "
              f"yıldız={bool(d_star)}, L={d_l is None}")
    except Exception as e:  # noqa: BLE001
        check("16. Bütünsel şekil oturtma: daire/elips/rect/yıldız EVET, L-poligon HAYIR", False, repr(e))

    return _summary()


def _summary() -> int:
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print("\n" + "=" * 52)
    print(f"SONUC: {passed}/{total} test gecti")
    print("=" * 52)
    failed = [name for name, ok, _ in _results if not ok]
    if failed:
        print("Basarisiz:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
