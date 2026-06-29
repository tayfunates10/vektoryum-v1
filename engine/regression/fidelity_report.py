r"""Algısal sadakat ölçüm CLI'si.

Regression fixture'larını (veya verilen görselleri) gerçek pipeline'dan geçirir
ve her aday için **algısal sadakat** (SSIM + LAB ΔE + kenar-F1) raporlar. Amaç:
motorda yapılan her değişikliğin kaliteye etkisini SAYISAL olarak görebilmek.

Kullanım (engine dizininden):

    .\.venv\Scripts\python.exe regression\fidelity_report.py
    .\.venv\Scripts\python.exe regression\fidelity_report.py path\to\image.png
    .\.venv\Scripts\python.exe regression\fidelity_report.py --json > rapor.json

CairoSVG yoksa sadakat hesaplanamaz; CLI bunu açıkça belirtir (skorlar yapısal
tahmine düşer ve fidelity sütunu boş kalır).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# engine kökünü import yoluna ekle (CLI doğrudan çalıştırılabilsin)
_ENGINE_ROOT = Path(__file__).resolve().parent.parent
if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))

from PIL import Image  # noqa: E402

from app.pipeline import run_pipeline  # noqa: E402

_HERE = Path(__file__).resolve().parent
_MANIFEST = _HERE / "manifest.json"
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _discover_inputs(args_paths: list[str]) -> list[tuple[str, Path, str]]:
    """(id, görsel_yolu, trace_mode) üçlüleri döndürür.

    Argümanlar dosya veya KLASÖR olabilir; klasör verilirse içindeki tüm görseller
    taranır (gerçek görsel topluluğuyla toplu ölçüm için).
    """
    if args_paths:
        out = []
        for p in args_paths:
            path = Path(p)
            if path.is_dir():
                for img in sorted(path.iterdir()):
                    if img.suffix.lower() in _IMAGE_EXTS:
                        out.append((img.stem, img, "auto"))
            elif path.exists():
                out.append((path.stem, path, "auto"))
            else:
                print(f"uyarı: bulunamadı: {path}", file=sys.stderr)
        return out

    inputs: list[tuple[str, Path, str]] = []
    if _MANIFEST.exists():
        manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
        for case in manifest.get("cases", []):
            img = (_HERE / case["input"]).resolve()
            if img.exists():
                inputs.append((case["id"], img, case.get("trace_mode", "auto")))
    if not inputs:
        for img in sorted((_HERE / "fixtures").glob("*.png")):
            inputs.append((img.stem, img, "auto"))
    return inputs


def _evaluate(image_path: Path, trace_mode: str) -> dict:
    """Tek görseli pipeline'dan geçirir ve sadakat özeti döndürür."""
    with Image.open(image_path) as im:
        image = im.convert("RGBA")

    with tempfile.TemporaryDirectory(prefix="fidelity_") as tmp:
        job_dir = Path(tmp)
        original_path = job_dir / f"original{image_path.suffix or '.png'}"
        original_path.write_bytes(image_path.read_bytes())
        pipe = run_pipeline(image, original_path, trace_mode, job_dir)

        candidates = []
        for c in pipe["scored"]:
            sd = c.get("score_details") or {}
            candidates.append({
                "name": c["name"],
                "total_score": c.get("total_score"),
                "fidelity_score": c.get("fidelity_score"),
                "ssim": sd.get("ssim"),
                "mean_delta_e": sd.get("mean_delta_e"),
                "edge_f1": sd.get("edge_f1"),
                "rendered_ok": c.get("rendered_ok"),
                "path_count": sd.get("path_count"),
                "unique_colors": sd.get("unique_colors"),
            })
        candidates.sort(key=lambda x: (x["fidelity_score"] is None, -(x["fidelity_score"] or 0)))

        best = pipe["best"]
        return {
            "mode_used": pipe["mode_used"],
            "best_candidate": best["name"] if best else None,
            "best_fidelity": best.get("fidelity_score") if best else None,
            "selection_reason": pipe["selection_reason"],
            "candidates": candidates,
        }


def _fmt(value: object, width: int, nd: int = 1) -> str:
    if value is None:
        return "—".rjust(width)
    if isinstance(value, float):
        return f"{value:.{nd}f}".rjust(width)
    return str(value).rjust(width)


def main() -> int:
    parser = argparse.ArgumentParser(description="Algısal sadakat ölçüm raporu")
    parser.add_argument("paths", nargs="*", help="Görsel yolları (boşsa manifest/fixtures kullanılır)")
    parser.add_argument("--json", action="store_true", help="Makine-okur JSON çıktısı")
    args = parser.parse_args()

    # Türkçe Windows konsolu (cp1254) ΔE gibi karakterleri kodlayamaz -> UTF-8'e zorla
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    inputs = _discover_inputs(args.paths)
    if not inputs:
        print("Değerlendirilecek görsel bulunamadı (regression/fixtures boş?).", file=sys.stderr)
        return 1

    report = {}
    selected_fidelities = []
    for case_id, img_path, trace_mode in inputs:
        result = _evaluate(img_path, trace_mode)
        report[case_id] = result
        if result["best_fidelity"] is not None:
            selected_fidelities.append(result["best_fidelity"])

    if args.json:
        mean = round(sum(selected_fidelities) / len(selected_fidelities), 2) if selected_fidelities else None
        print(json.dumps({"cases": report, "mean_selected_fidelity": mean}, ensure_ascii=False, indent=2))
        return 0

    render_available = any(
        c["fidelity_score"] is not None
        for r in report.values() for c in r["candidates"]
    )
    if not render_available:
        print("UYARI: CairoSVG ile render yapılamadı; sadakat hesaplanamadı.")
        print("       Skorlar yapısal tahmine düşüyor (fidelity sütunları boş).\n")

    for case_id, result in report.items():
        print(f"=== {case_id}  (mode: {result['mode_used']}) ===")
        print(f"    seçilen: {result['best_candidate']}  "
              f"fidelity: {_fmt(result['best_fidelity'], 5)}  "
              f"({result['selection_reason']})")
        print(f"    {'aday':<22}{'fidelity':>9}{'ssim':>7}{'ΔE':>7}{'edgeF1':>8}{'total':>7}{'paths':>7}")
        for c in result["candidates"]:
            star = "*" if c["name"] == result["best_candidate"] else " "
            print(f"  {star} {c['name']:<22}"
                  f"{_fmt(c['fidelity_score'], 9)}"
                  f"{_fmt(c['ssim'], 7, 3)}"
                  f"{_fmt(c['mean_delta_e'], 7, 2)}"
                  f"{_fmt(c['edge_f1'], 8, 3)}"
                  f"{_fmt(c['total_score'], 7)}"
                  f"{_fmt(c['path_count'], 7)}")
        print()

    if selected_fidelities:
        mean = sum(selected_fidelities) / len(selected_fidelities)
        print(f"ORTALAMA seçilen-aday sadakati: {mean:.2f}  "
              f"({len(selected_fidelities)}/{len(inputs)} görsel render edildi)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
