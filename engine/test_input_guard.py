"""P0: güvenli görüntü alımı — magic/format/boyut/piksel/bomb/animated/EXIF/ICC.

Doğrulanan canlı hatalar: istemci Content-Type'a güveniliyordu, byte/piksel sınırı
ve gerçek magic kontrolü yoktu, decompression bomb korumasızdı, EXIF/ICC normalize
edilmiyordu. Bu testler guard'ı ve /api/vectorize entegrasyonunu kilitler.

Çalıştırma::  .venv/bin/python test_input_guard.py   (~5 sn)
"""
from __future__ import annotations

import asyncio
import io
import json
import sys
from pathlib import Path

from PIL import Image

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


def _bytes(mode="RGB", size=(16, 16), fmt="PNG", color=(255, 0, 11), **kw):
    b = io.BytesIO()
    Image.new(mode, size, color).save(b, fmt, **kw)
    return b.getvalue()


def test_valid_png_jpeg_webp() -> None:
    print("== Geçerli PNG/JPEG/WEBP kabul ==")
    from app.input_guard import validate_and_load
    for fmt in ("PNG", "JPEG", "WEBP"):
        li = validate_and_load(_bytes(fmt=fmt))
        check(li.format == fmt, f"{fmt} kabul edildi (sha={li.sha256[:8]})")


def test_unsupported_format() -> None:
    print("== Desteklenmeyen format (GIF/BMP) → unsupported_format 415 ==")
    from app.input_guard import validate_and_load, InputError
    for fmt in ("GIF", "BMP"):
        try:
            validate_and_load(_bytes(fmt=fmt))
            check(False, f"{fmt} reddedilmedi")
        except InputError as e:
            check(e.code == "unsupported_format" and e.status == 415, f"{fmt} → {e.code} {e.status}")


def test_content_type_not_trusted() -> None:
    print("== Content-Type'a güvenilmez: gerçek magic kullanılır ==")
    from app.input_guard import validate_and_load, InputError
    # gerçekte GIF baytları (istemci 'image/png' dese bile) → 415
    try:
        validate_and_load(_bytes(fmt="GIF"), filename="fake.png")
        check(False, "sahte uzantı geçti")
    except InputError as e:
        check(e.code == "unsupported_format", "magic tespit etti (uzantı değil)")


def test_oversize_pixels_and_bytes() -> None:
    print("== Byte ve piksel sınırları ==")
    from app.input_guard import validate_and_load, InputError
    from app.settings import Settings
    s_px = Settings(10**9, 8, 100, frozenset({"PNG"}), 0)
    try:
        validate_and_load(_bytes(size=(64, 64)), settings=s_px)
        check(False, "büyük görsel geçti")
    except InputError as e:
        check(e.code == "image_too_large" and e.status == 413, f"piksel → {e.code} {e.status}")
    s_b = Settings(10, 99999, 10**9, frozenset({"PNG"}), 0)
    try:
        validate_and_load(_bytes(), settings=s_b)
        check(False, "büyük dosya geçti")
    except InputError as e:
        check(e.code == "file_too_large" and e.status == 413, f"byte → {e.code} {e.status}")


def test_corrupt_rejected() -> None:
    print("== Bozuk/kesik veri reddedilir ==")
    from app.input_guard import validate_and_load, InputError
    # geçerli PNG magic + kesik gövde
    good = _bytes()
    truncated = good[:len(good) // 2]
    try:
        validate_and_load(truncated)
        check(False, "kesik PNG geçti")
    except InputError as e:
        check(e.code in ("corrupt_image", "unsupported_format"), f"kesik → {e.code}")


def test_animated_rejected() -> None:
    print("== Animasyonlu (çok kareli) reddedilir ==")
    from app.input_guard import validate_and_load, InputError
    b = io.BytesIO()
    f0 = Image.new("RGB", (16, 16), (255, 0, 0))
    f1 = Image.new("RGB", (16, 16), (0, 255, 0))
    try:
        f0.save(b, "WEBP", save_all=True, append_images=[f1], duration=100)
    except Exception:  # noqa: BLE001 — WEBP animasyon kaydı yoksa GIF dene
        b = io.BytesIO()
        f0.save(b, "GIF", save_all=True, append_images=[f1], duration=100)
    try:
        validate_and_load(b.getvalue())
        check(False, "animasyon geçti")
    except InputError as e:
        check(e.code in ("animated_not_supported", "unsupported_format"),
              f"animasyon → {e.code} {e.status}")


def test_exif_transpose() -> None:
    print("== EXIF orientation uygulanır (boyut düzelir) ==")
    from app.input_guard import validate_and_load
    ex = Image.Exif()
    ex[0x0112] = 6   # 90° döndür → 20x10 görüntü 10x20 olur
    b = io.BytesIO()
    Image.new("RGB", (20, 10), (255, 0, 0)).save(b, "JPEG", exif=ex.tobytes())
    li = validate_and_load(b.getvalue())
    check((li.width, li.height) == (10, 20), f"EXIF transpose sonrası {li.width}x{li.height}")
    check(li.normalized, "normalized=True (EXIF değişti)")


def test_alpha_detected() -> None:
    print("== Alpha kanalı tespit edilir ==")
    from app.input_guard import validate_and_load
    li = validate_and_load(_bytes(mode="RGBA", color=(255, 0, 0, 100)))
    check(li.has_alpha, "RGBA has_alpha=True")
    li2 = validate_and_load(_bytes(mode="RGB"))
    check(not li2.has_alpha, "RGB has_alpha=False")


def test_endpoint_rejects_with_code() -> None:
    print("== /api/vectorize güvenli alım hatalarını kod ile döner ==")
    import app.main as M
    from starlette.datastructures import Headers, UploadFile

    M._require_user = lambda s: {"email": "t@t", "role": "user"}
    # GIF → 415 unsupported_format
    gb = _bytes(fmt="GIF")
    up = UploadFile(filename="x.gif", file=io.BytesIO(gb),
                    headers=Headers({"content-type": "image/png"}))  # sahte content-type
    resp = asyncio.run(M.vectorize_image(file=up, trace_mode="auto",
                                         shape_stacking="stacked", edge_cleanup="on", session="x"))
    body = json.loads(bytes(resp.body))
    check(resp.status_code == 415, f"GIF → 415 ({resp.status_code})")
    check(body.get("code") == "unsupported_format", f"kod unsupported_format ({body.get('code')})")


def main() -> int:
    test_valid_png_jpeg_webp()
    test_unsupported_format()
    test_content_type_not_trusted()
    test_oversize_pixels_and_bytes()
    test_corrupt_rejected()
    test_animated_rejected()
    test_exif_transpose()
    test_alpha_detected()
    test_endpoint_rejects_with_code()
    print("=" * 60)
    if FAILS:
        print(f"SONUC: {len(FAILS)} KONTROL BASARISIZ")
        for m in FAILS:
            print(" -", m)
        return 1
    print("SONUC: tum kontroller gecti")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
