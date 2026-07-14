"""Human-readable JSON/HTML summary for benchmark pipeline results."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

_HIGHER_IS_BETTER = ("fidelity", "ssim", "edge_f1", "alpha_iou")
_LOWER_IS_BETTER = ("delta_e00", "path_count", "svg_bytes", "render_ms", "peak_rss_mb")
_ALPHA_RELEVANT_CATEGORIES = {"transparent"}


def _category(case_id: str) -> str:
    parts = str(case_id).split("-", 3)
    return parts[3] if len(parts) == 4 else str(case_id)


def build_quality_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != "benchmark-results-v1":
        raise ValueError("unsupported benchmark results schema")
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("benchmark results are required")

    rows: list[dict[str, Any]] = []
    for item in results:
        metrics = dict(item.get("metrics") or {})
        category = _category(str(item.get("case_id", "")))
        missing = sorted(name for name in (*_HIGHER_IS_BETTER, *_LOWER_IS_BETTER) if metrics.get(name) is None)
        rows.append({
            "case_id": str(item.get("case_id", "")),
            "category": category,
            "alpha_applicable": category in _ALPHA_RELEVANT_CATEGORIES,
            "failure": item.get("failure"),
            "missing_metrics": missing,
            "metrics": metrics,
        })
    rows.sort(key=lambda row: row["case_id"])

    def worst(metric: str, reverse: bool) -> list[dict[str, Any]]:
        measured = [
            row for row in rows
            if row["metrics"].get(metric) is not None
            and (metric != "alpha_iou" or row["alpha_applicable"])
        ]
        measured.sort(key=lambda row: float(row["metrics"][metric]), reverse=reverse)
        return [{"case_id": row["case_id"], "value": row["metrics"][metric]} for row in measured[:3]]

    alerts: list[dict[str, Any]] = []
    for row in rows:
        m = row["metrics"]
        if row["alpha_applicable"] and m.get("alpha_iou") is not None and float(m["alpha_iou"]) < 0.95:
            alerts.append({"case_id": row["case_id"], "metric": "alpha_iou", "value": m["alpha_iou"]})
        if m.get("ssim") is not None and float(m["ssim"]) < 0.97:
            alerts.append({"case_id": row["case_id"], "metric": "ssim", "value": m["ssim"]})
        if m.get("render_ms") is not None and float(m["render_ms"]) > 15000:
            alerts.append({"case_id": row["case_id"], "metric": "render_ms", "value": m["render_ms"]})

    return {
        "schema_version": "benchmark-quality-summary-v1",
        "case_count": len(rows),
        "categories": sorted({row["category"] for row in rows}),
        "rows": rows,
        "alerts": sorted(alerts, key=lambda x: (x["case_id"], x["metric"])),
        "worst_cases": {
            **{name: worst(name, reverse=False) for name in _HIGHER_IS_BETTER},
            **{name: worst(name, reverse=True) for name in _LOWER_IS_BETTER},
        },
    }


def write_reports(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "quality_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    rows = []
    for row in summary["rows"]:
        m = row["metrics"]
        alpha_value = m.get("alpha_iou") if row["alpha_applicable"] else "n/a"
        rows.append("<tr>" + "".join([
            f"<td>{html.escape(row['case_id'])}</td>",
            f"<td>{html.escape(row['category'])}</td>",
            f"<td>{m.get('fidelity')}</td>", f"<td>{m.get('ssim')}</td>",
            f"<td>{m.get('edge_f1')}</td>", f"<td>{alpha_value}</td>",
            f"<td>{m.get('delta_e00')}</td>", f"<td>{m.get('path_count')}</td>",
            f"<td>{m.get('render_ms')}</td>",
        ]) + "</tr>")
    alert_rows = "".join(
        f"<li>{html.escape(a['case_id'])}: {html.escape(a['metric'])} = {a['value']}</li>"
        for a in summary["alerts"]
    ) or "<li>None</li>"
    page = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Vektoryum Benchmark Quality</title></head><body>"
        f"<h1>Benchmark Quality Summary</h1><p>Cases: {summary['case_count']}</p><h2>Alerts</h2><ul>{alert_rows}</ul>"
        "<table><thead><tr><th>Case</th><th>Category</th><th>Fidelity</th><th>SSIM</th><th>Edge F1</th>"
        "<th>Alpha IoU</th><th>Delta E00</th><th>Paths</th><th>Render ms</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></body></html>"
    )
    (output_dir / "quality_summary.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    summary = build_quality_summary(payload)
    write_reports(args.output, summary)
    print(json.dumps({"status": "ok", "case_count": summary["case_count"], "alert_count": len(summary["alerts"])}, sort_keys=True))


if __name__ == "__main__":
    main()
