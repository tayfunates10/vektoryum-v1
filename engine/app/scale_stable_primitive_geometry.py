"""Source-derived primitive calibration against the real production renderer.

Only geometry inferred from connected source masks is considered.  Candidate
SVGs are rendered in isolation at 0.25x/0.5x/1x/2x/4x and compared with the
same Pillow primitive semantics used by normalized source geometry.  Nested
ring/hole regions are deliberately not rewritten as top-level shapes.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from app import semantic_region_geometry_impl as _impl

FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)


def _fmt(value: float) -> str:
    return _impl._fmt(float(value))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _pil_mask(width: int, height: int, draw_fn: Any) -> np.ndarray:
    image = Image.new("L", (int(width), int(height)), 0)
    draw_fn(ImageDraw.Draw(image))
    return np.asarray(image, dtype=np.uint8) > 0


def _scaled(value: float, target: int, source: int) -> int:
    return int(round(float(value) * float(target) / float(source)))


def _scaled_width(value: float, target: int, source: int) -> int:
    return max(1, _scaled(value, target, source))


def model_mask(model: dict[str, Any], sw: int, sh: int, tw: int, th: int) -> np.ndarray:
    kind = str(model["kind"])
    if kind == "ellipse":
        cx = _scaled(model["cx"], tw, sw)
        cy = _scaled(model["cy"], th, sh)
        rx = _scaled_width(model["rx"], tw, sw)
        ry = _scaled_width(model["ry"], th, sh)
        return _pil_mask(tw, th, lambda d: d.ellipse((cx-rx, cy-ry, cx+rx, cy+ry), fill=255))
    if kind == "line":
        x0 = _scaled(model["x0"], tw, sw)
        x1 = _scaled(model["x1"], tw, sw)
        y0 = _scaled(model["y0"], th, sh)
        y1 = _scaled(model["y1"], th, sh)
        axis = str(model["axis"])
        target = th if axis == "h" else tw
        source = sh if axis == "h" else sw
        stroke = _scaled_width(model["width"], target, source)
        return _pil_mask(tw, th, lambda d: d.line((x0, y0, x1, y1), fill=255, width=stroke))
    if kind == "rect":
        x = _scaled(model["x"], tw, sw)
        y = _scaled(model["y"], th, sh)
        width = _scaled_width(model["width"], tw, sw)
        height = _scaled_width(model["height"], th, sh)
        return _pil_mask(tw, th, lambda d: d.rectangle((x, y, x+width-1, y+height-1), fill=255))
    if kind == "rounded_rect":
        x0 = _scaled(model["x0"], tw, sw)
        y0 = _scaled(model["y0"], th, sh)
        x1 = _scaled(model["x1"], tw, sw)
        y1 = _scaled(model["y1"], th, sh)
        radius = _scaled_width(model["radius"], min(tw, th), min(sw, sh))
        return _pil_mask(tw, th, lambda d: d.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=255))
    raise ValueError(f"unsupported primitive {kind}")


def _infer_rounded(mask: np.ndarray) -> dict[str, Any] | None:
    source = np.asarray(mask, dtype=bool)
    box = _bbox(source)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    width = x1-x0+1
    height = y1-y0+1
    if min(width, height) < 8:
        return None
    outer = np.asarray(_impl._core._filled_external_region(source), dtype=bool)
    # Exact native reconstruction keeps this inference source-derived and avoids
    # arbitrary corner-radius guesses.
    for radius in range(1, max(1, min(width, height)//2)+1):
        candidate = _pil_mask(
            source.shape[1], source.shape[0],
            lambda d, r=radius: d.rounded_rectangle((x0, y0, x1, y1), radius=r, fill=255),
        )
        if np.array_equal(candidate, outer):
            return {
                "kind": "rounded_rect",
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "radius": radius,
            }
    return None


def _point(cx: float, cy: float, fill: str, *, cap: str = "square", width: float = 1.0) -> str:
    return (
        f'<path d="M{_fmt(cx)} {_fmt(cy)}h0.001" fill="none" stroke="{fill}" '
        f'stroke-width="{_fmt(width)}" vector-effect="non-scaling-stroke" stroke-linecap="{cap}"/>'
    )


def _candidate_elements(model: dict[str, Any], fill: str) -> list[list[str]]:
    kind = str(model["kind"])
    if kind == "ellipse":
        cx, cy, rx, ry = (float(model[k]) for k in ("cx", "cy", "rx", "ry"))
        return [
            [f'<ellipse cx="{_fmt(cx)}" cy="{_fmt(cy)}" rx="{_fmt(rx)}" ry="{_fmt(ry)}" fill="{fill}" stroke="{fill}" stroke-width="1" vector-effect="non-scaling-stroke"/>'],
            [f'<ellipse cx="{_fmt(cx)}" cy="{_fmt(cy)}" rx="{_fmt(rx)}" ry="{_fmt(ry)}" fill="{fill}" stroke="{fill}" stroke-width="1.25" vector-effect="non-scaling-stroke"/>'],
            [f'<ellipse cx="{_fmt(cx)}" cy="{_fmt(cy)}" rx="{_fmt(rx)}" ry="{_fmt(ry)}" fill="{fill}" stroke="{fill}" stroke-width="1.5" vector-effect="non-scaling-stroke"/>'],
            [f'<ellipse cx="{_fmt(cx)}" cy="{_fmt(cy)}" rx="{_fmt(rx+0.25)}" ry="{_fmt(ry+0.25)}" fill="{fill}"/>'],
            [f'<ellipse cx="{_fmt(cx)}" cy="{_fmt(cy)}" rx="{_fmt(max(.5,rx-.125))}" ry="{_fmt(max(.5,ry-.125))}" fill="{fill}" stroke="{fill}" stroke-width="1" vector-effect="non-scaling-stroke"/>'],
        ]
    if kind == "line":
        x0, y0, x1, y1, width = (float(model[k]) for k in ("x0", "y0", "x1", "y1", "width"))
        axis = str(model["axis"])
        centers = (width/2.0, max(0.0, (width-1.0)/2.0))
        output: list[list[str]] = []
        for offset in centers:
            path = (
                f'M{_fmt(x0)} {_fmt(y0+offset)}H{_fmt(x1)}'
                if axis == "h" else
                f'M{_fmt(x0+offset)} {_fmt(y0)}V{_fmt(y1)}'
            )
            base = f'<path d="{path}" fill="none" stroke="{fill}" stroke-width="{_fmt(width)}" stroke-linecap="butt"/>'
            output.append([base])
            output.append([base, f'<path d="{path}" fill="none" stroke="{fill}" stroke-width="1" vector-effect="non-scaling-stroke" stroke-linecap="butt"/>'])
        return output
    if kind == "rect":
        x, y, width, height = (float(model[k]) for k in ("x", "y", "width", "height"))
        cx = x+(width-1.0)/2.0
        cy = y+(height-1.0)/2.0
        point = _point(cx, cy, fill)
        specs = (
            (x-.01, y-.01, width+.02, height+.02),
            (x-.01, y-.01, width, height),
            (x+.01, y-.01, width, height),
            (x-.01, y+.01, width, height),
            (x, y, width, height),
        )
        output = [[f'<rect x="{_fmt(px)}" y="{_fmt(py)}" width="{_fmt(pw)}" height="{_fmt(ph)}" fill="{fill}"/>'] for px,py,pw,ph in specs]
        output.append([f'<rect x="{_fmt(x-.01)}" y="{_fmt(y-.01)}" width="{_fmt(width+.02)}" height="{_fmt(height+.02)}" fill="{fill}"/>', point])
        return output
    if kind == "rounded_rect":
        x0, y0, x1, y1, radius = (float(model[k]) for k in ("x0", "y0", "x1", "y1", "radius"))
        width = x1-x0+1.0
        height = y1-y0+1.0
        specs = (
            (x0, y0, width, height, radius),
            (x0-.01, y0-.01, width+.02, height+.02, radius),
            (x0-.01, y0-.01, width+.02, height+.02, max(.5, radius-.5)),
            (x0-.01, y0-.01, width+.02, height+.02, radius+.5),
            (x0, y0-.01, width, height+.02, max(.5, radius-.5)),
            (x0+.01, y0-.01, width, height+.02, radius),
        )
        return [[f'<rect x="{_fmt(px)}" y="{_fmt(py)}" width="{_fmt(pw)}" height="{_fmt(ph)}" rx="{_fmt(pr)}" ry="{_fmt(pr)}" fill="{fill}"/>'] for px,py,pw,ph,pr in specs]
    return []


def _blacken(elements: list[str], fill: str) -> list[str]:
    return [element.replace(fill, "#000000") for element in elements]


def _metrics(model: dict[str, Any], elements: list[str], fill: str, sw: int, sh: int) -> dict[str, Any] | None:
    colors = np.asarray([[0,0,0], [255,255,255]], dtype=np.uint8)
    background = f'<path fill="#ffffff" d="M0 0H{sw}V{sh}H0Z"/>'
    black = [background] + _blacken(elements, fill)
    levels: list[dict[str, Any]] = []
    for factor in FACTORS:
        tw = max(16, int(round(sw*factor)))
        th = max(16, int(round(sh*factor)))
        rendered = _impl._render(sw, sh, black, colors, tw, th)
        if rendered is None:
            return None
        actual = rendered == 0
        expected = model_mask(model, sw, sh, tw, th)
        component_count = _impl._cc(actual)
        intersection = int(np.count_nonzero(actual & expected))
        union = int(np.count_nonzero(actual | expected))
        iou = 1.0 if union == 0 else intersection/float(union)
        levels.append({
            "scale": f"{factor:g}x",
            "size": [tw, th],
            "component_count": int(component_count),
            "iou": float(iou),
            "mismatch": float(np.mean(actual != expected)),
        })
    return {
        "levels": levels,
        "component_error": int(sum(abs(int(level["component_count"])-1) for level in levels)),
        "min_iou": float(min(float(level["iou"]) for level in levels)),
        "mean_iou": float(np.mean([float(level["iou"]) for level in levels])),
        "max_mismatch": float(max(float(level["mismatch"]) for level in levels)),
        "native_iou": float(levels[2]["iou"]),
    }


def _score(metrics: dict[str, Any], element_count: int) -> tuple[float,float,float,float,int]:
    return (
        float(metrics["component_error"]),
        -float(metrics["min_iou"]),
        float(metrics["max_mismatch"]),
        -float(metrics["mean_iou"]),
        int(element_count),
    )


def calibrate_primitives(
    labels: np.ndarray,
    colors: np.ndarray,
    elements: list[str],
    regions: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]], dict[int,dict[str,Any]]]:
    node_map, nodes, root, _depth, parent = _impl._core._connected_region_graph(labels, len(colors))
    child_count = [0 for _ in nodes]
    for node_id, parent_id in enumerate(parent):
        if parent_id is not None and 0 <= int(parent_id) < len(child_count):
            child_count[int(parent_id)] += 1

    parts: list[list[str]] = []
    cursor = 0
    for region in regions:
        count = int(region.get("path_count") or 0)
        parts.append(list(elements[cursor:cursor+count]))
        cursor += count
    if cursor != len(elements):
        return list(elements), [], {}

    sh, sw = labels.shape
    reports: list[dict[str, Any]] = []
    repair_models: dict[int,dict[str,Any]] = {}
    for index, region in enumerate(regions):
        if len(parts[index]) != 1:
            continue
        node_id = int(region["node_id"])
        color_index = int(region["color_index"])
        strategy = str(region.get("strategy") or "")
        source_mask = node_map == node_id
        model = _impl._infer(source_mask, strategy)
        rounded = _infer_rounded(source_mask)
        is_top_level = parent[node_id] == root if 0 <= node_id < len(parent) else False
        is_leaf = child_count[node_id] == 0 if 0 <= node_id < len(child_count) else False

        if rounded is not None and is_top_level:
            model = rounded
        elif model is None or not is_leaf or str(model.get("kind")) not in {"ellipse", "line", "rect"}:
            continue

        red, green, blue = (int(value) for value in colors[color_index])
        fill = f"#{red:02x}{green:02x}{blue:02x}"
        candidates = _candidate_elements(model, fill)
        if not candidates:
            continue
        base_part = list(parts[index])
        base_metrics = _metrics(model, base_part, fill, sw, sh)
        if base_metrics is None:
            continue
        best_part = base_part
        best_metrics = base_metrics
        best_score = _score(base_metrics, len(base_part))
        for candidate in candidates:
            metrics = _metrics(model, candidate, fill, sw, sh)
            if metrics is None:
                continue
            if float(metrics["native_iou"])+1e-12 < float(_impl._NATIVE_MIN_IOU):
                continue
            score = _score(metrics, len(candidate))
            if score < best_score:
                best_score = score
                best_part = list(candidate)
                best_metrics = metrics

        changed = best_part != base_part
        if changed:
            parts[index] = best_part
        if str(model["kind"]) == "rect":
            repair_models[node_id] = dict(model)
        elif str(model["kind"]) in {"ellipse", "line"}:
            repair_models[node_id] = dict(model)
        reports.append({
            "node_id": node_id,
            "color_index": color_index,
            "strategy": strategy,
            "model": model,
            "top_level": bool(is_top_level),
            "leaf": bool(is_leaf),
            "changed": changed,
            "before": base_metrics,
            "after": best_metrics,
            "element_delta": len(best_part)-len(base_part),
        })
    return [element for part in parts for element in part], reports, repair_models


__all__ = ["FACTORS", "calibrate_primitives", "model_mask"]
