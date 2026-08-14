"""Temporary generic renderer probe for Issue #152; removed before final."""
from __future__ import annotations

import numpy as np


def _fmt(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def pytest_sessionfinish(session, exitstatus):  # noqa: ANN001
    from app import semantic_region_geometry_impl as impl
    from app.scale_stable_primitive_geometry import FACTORS, model_mask

    colors = np.asarray([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    sw = sh = 256
    background = '<path fill="#ffffff" d="M0 0H256V256H0Z"/>'

    def bbox(mask):
        ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
        if not len(xs):
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    def measure(model, element):
        levels = []
        for factor in FACTORS:
            tw = th = max(16, int(round(256 * factor)))
            rendered = impl._render(sw, sh, [background, element], colors, tw, th)
            if rendered is None:
                return None
            actual = rendered == 0
            expected = model_mask(model, sw, sh, tw, th)
            inter = int(np.count_nonzero(actual & expected))
            union = int(np.count_nonzero(actual | expected))
            levels.append({
                "scale": f"{factor:g}x",
                "iou": 1.0 if union == 0 else inter / float(union),
                "actual_area": int(np.count_nonzero(actual)),
                "expected_area": int(np.count_nonzero(expected)),
                "actual_bbox": bbox(actual),
                "expected_bbox": bbox(expected),
                "components": int(impl._cc(actual)),
            })
        return levels

    ellipse_results = []
    for phase in range(4):
        cx = cy = 128.0 + float(phase)
        for radius in (1.0, 2.0, 3.0, 4.0):
            model = {"kind": "ellipse", "cx": cx, "cy": cy, "rx": radius, "ry": radius}
            candidates = []
            for delta in (-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5):
                rr = max(0.125, radius + delta)
                candidates.append((f"fill_d{delta:g}", f'<ellipse cx="{_fmt(cx)}" cy="{_fmt(cy)}" rx="{_fmt(rr)}" ry="{_fmt(rr)}" fill="#000000"/>'))
                for stroke_width in (0.5, 0.75, 1.0, 1.25, 1.5):
                    candidates.append((
                        f"ns_d{delta:g}_w{stroke_width:g}",
                        f'<ellipse cx="{_fmt(cx)}" cy="{_fmt(cy)}" rx="{_fmt(rr)}" ry="{_fmt(rr)}" fill="#000000" stroke="#000000" stroke-width="{_fmt(stroke_width)}" vector-effect="non-scaling-stroke"/>'
                    ))
            ranked = []
            for name, element in candidates:
                levels = measure(model, element)
                if not levels:
                    continue
                ranked.append((
                    min(float(level["iou"]) for level in levels),
                    sum(float(level["iou"]) for level in levels) / len(levels),
                    name,
                    levels,
                ))
            ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
            ellipse_results.append({
                "phase": phase,
                "radius": radius,
                "top": [
                    {"name": item[2], "min_iou": item[0], "mean_iou": item[1], "levels": item[3]}
                    for item in ranked[:4]
                ],
            })

    rect_results = []
    for phase in range(4):
        x = y = 128.0 + float(phase)
        for side in (1.0, 2.0, 3.0, 4.0):
            model = {"kind": "rect", "x": x, "y": y, "width": side, "height": side}
            candidates = [
                ("full", f'<rect x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(side)}" height="{_fmt(side)}" fill="#000000"/>'),
                ("full_m01", f'<rect x="{_fmt(x-.01)}" y="{_fmt(y-.01)}" width="{_fmt(side+.02)}" height="{_fmt(side+.02)}" fill="#000000"/>'),
                ("inclusive_ns", f'<rect x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(max(.001, side-1.0))}" height="{_fmt(max(.001, side-1.0))}" fill="#000000" stroke="#000000" stroke-width="1" vector-effect="non-scaling-stroke"/>'),
                ("center_point_ns", f'<path d="M{_fmt(x+(side-1)/2)} {_fmt(y+(side-1)/2)}h0.001" fill="none" stroke="#000000" stroke-width="1" vector-effect="non-scaling-stroke" stroke-linecap="square"/>'),
            ]
            ranked = []
            for name, element in candidates:
                levels = measure(model, element)
                if not levels:
                    continue
                ranked.append((
                    min(float(level["iou"]) for level in levels),
                    sum(float(level["iou"]) for level in levels) / len(levels),
                    name,
                    levels,
                ))
            ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
            rect_results.append({
                "phase": phase,
                "side": side,
                "top": [
                    {"name": item[2], "min_iou": item[0], "mean_iou": item[1], "levels": item[3]}
                    for item in ranked
                ],
            })

    print("\nISSUE152_GENERIC_ELLIPSE_PROBE=" + repr(ellipse_results))
    print("ISSUE152_GENERIC_RECT_PROBE=" + repr(rect_results))
