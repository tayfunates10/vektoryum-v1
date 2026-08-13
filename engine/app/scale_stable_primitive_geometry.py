"""Source-derived, production-renderer-verified scale-stable primitive geometry."""
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
    if kind == "small_rect":
        x = _scaled(model["x"], tw, sw)
        y = _scaled(model["y"], th, sh)
        width = _scaled_width(model["width"], tw, sw)
        height = _scaled_width(model["height"], th, sh)
        return _pil_mask(tw, th, lambda d: d.rectangle((x, y, x + width - 1, y + height - 1), fill=255))
    if kind == "line":
        x0 = _scaled(model["x0"], tw, sw)
        x1 = _scaled(model["x1"], tw, sw)
        y0 = _scaled(model["y0"], th, sh)
        y1 = _scaled(model["y1"], th, sh)
        target = th if str(model["axis"]) == "h" else tw
        source = sh if str(model["axis"]) == "h" else sw
        width = _scaled_width(model["width"], target, source)
        return _pil_mask(tw, th, lambda d: d.line((x0, y0, x1, y1), fill=255, width=width))
    if kind == "ellipse":
        cx = _scaled(model["cx"], tw, sw)
        cy = _scaled(model["cy"], th, sh)
        rx = _scaled_width(model["rx"], tw, sw)
        ry = _scaled_width(model["ry"], th, sh)
        return _pil_mask(tw, th, lambda d: d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255))
    if kind == "rounded_rect":
        x0 = _scaled(model["x0"], tw, sw)
        y0 = _scaled(model["y0"], th, sh)
        x1 = _scaled(model["x1"], tw, sw)
        y1 = _scaled(model["y1"], th, sh)
        radius = _scaled_width(model["radius"], min(tw, th), min(sw, sh))
        return _pil_mask(tw, th, lambda d: d.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=255))
    raise ValueError(f"unsupported semantic primitive: {kind}")


def infer_model(mask: np.ndarray, strategy: str) -> dict[str, Any] | None:
    source = np.asarray(mask, dtype=bool)
    box = _bbox(source)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    crop = source[y0 : y1 + 1, x0 : x1 + 1]
    occupancy = float(crop.mean())

    if occupancy >= 0.999:
        short = min(width, height)
        long = max(width, height)
        if short <= 4 and long >= max(5, short * 3):
            if width >= height:
                return {"kind": "line", "axis": "h", "x0": x0, "y0": y0, "x1": x1, "y1": y0, "width": height}
            return {"kind": "line", "axis": "v", "x0": x0, "y0": y0, "x1": x0, "y1": y1, "width": width}
        if max(width, height) <= 10:
            return {"kind": "small_rect", "x": x0, "y": y0, "width": width, "height": height}

    outer = np.asarray(_impl._core._filled_external_region(source), dtype=bool)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    rx = (width - 1) / 2.0
    ry = (height - 1) / 2.0
    if min(rx, ry) >= 1.0 and float(cx).is_integer() and float(cy).is_integer():
        ellipse = {"kind": "ellipse", "cx": int(cx), "cy": int(cy), "rx": int(round(rx)), "ry": int(round(ry))}
        if np.array_equal(model_mask(ellipse, source.shape[1], source.shape[0], source.shape[1], source.shape[0]), outer):
            return ellipse

    if min(width, height) >= 8:
        for radius in range(1, max(1, min(width, height) // 2) + 1):
            candidate = _pil_mask(
                source.shape[1],
                source.shape[0],
                lambda d, r=radius: d.rounded_rectangle((x0, y0, x1, y1), radius=r, fill=255),
            )
            if np.array_equal(candidate, outer):
                return {"kind": "rounded_rect", "x0": x0, "y0": y0, "x1": x1, "y1": y1, "radius": radius}
    return None


def _point(cx: float, cy: float, fill: str) -> str:
    return (
        f'<path d="M{_fmt(cx)} {_fmt(cy)}h0.001" fill="none" stroke="{fill}" '
        f'stroke-width="1" vector-effect="non-scaling-stroke" stroke-linecap="square"/>'
    )


def candidate_elements(model: dict[str, Any], fill: str) -> list[list[str]]:
    kind = str(model["kind"])
    output: list[list[str]] = []
    if kind == "small_rect":
        x = float(model["x"]); y = float(model["y"])
        width = float(model["width"]); height = float(model["height"])
        cx = x + (width - 1.0) / 2.0; cy = y + (height - 1.0) / 2.0
        for delta in (0.0, -0.25, 0.25, -0.49, 0.49):
            output.append([f'<rect x="{_fmt(x-delta/2)}" y="{_fmt(y-delta/2)}" width="{_fmt(width+delta)}" height="{_fmt(height+delta)}" fill="{fill}"/>'])
        output.append([f'<rect x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(width)}" height="{_fmt(height)}" fill="{fill}"/>', _point(cx, cy, fill)])
        output.append([f'<rect x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(max(0.001,width-1.0))}" height="{_fmt(max(0.001,height-1.0))}" fill="{fill}" stroke="{fill}" stroke-width="1" vector-effect="non-scaling-stroke"/>'])
        output.append([f'<path d="M{_fmt(cx)} {_fmt(cy)}h0.001" fill="none" stroke="{fill}" stroke-width="{_fmt(max(width,height))}" stroke-linecap="square"/>', _point(cx, cy, fill)])
        return output

    if kind == "line":
        x0=float(model["x0"]); y0=float(model["y0"]); x1=float(model["x1"]); y1=float(model["y1"]); width=float(model["width"])
        axis=str(model["axis"]); center=(y0+(width-1.0)/2.0) if axis=="h" else (x0+(width-1.0)/2.0)
        for end_delta in (0.0, 0.25, 0.5, 0.75, 1.0):
            path=(f'M{_fmt(x0)} {_fmt(center)}H{_fmt(x1+end_delta)}' if axis=="h" else f'M{_fmt(center)} {_fmt(y0)}V{_fmt(y1+end_delta)}')
            scalable=f'<path d="{path}" fill="none" stroke="{fill}" stroke-width="{_fmt(width)}" stroke-linecap="butt"/>'
            minimum=f'<path d="{path}" fill="none" stroke="{fill}" stroke-width="1" vector-effect="non-scaling-stroke" stroke-linecap="butt"/>'
            output.append([scalable]); output.append([scalable, minimum])
        return output

    if kind == "ellipse":
        cx=float(model["cx"]); cy=float(model["cy"]); rx=float(model["rx"]); ry=float(model["ry"])
        for delta in (-0.25, 0.0, 0.25, 0.5):
            erx=max(0.5,rx+delta); ery=max(0.5,ry+delta)
            output.append([f'<ellipse cx="{_fmt(cx)}" cy="{_fmt(cy)}" rx="{_fmt(erx)}" ry="{_fmt(ery)}" fill="{fill}"/>'])
            output.append([f'<ellipse cx="{_fmt(cx)}" cy="{_fmt(cy)}" rx="{_fmt(erx)}" ry="{_fmt(ery)}" fill="{fill}" stroke="{fill}" stroke-width="1" vector-effect="non-scaling-stroke"/>'])
        return output

    if kind == "rounded_rect":
        x0=float(model["x0"]); y0=float(model["y0"]); x1=float(model["x1"]); y1=float(model["y1"]); radius=float(model["radius"])
        span_x=x1-x0; span_y=y1-y0
        for grow in (0.0, 0.25, 0.5, 0.75, 1.0):
            for radius_delta in (-0.25, 0.0, 0.25):
                rr=max(0.5,radius+radius_delta)
                output.append([f'<rect x="{_fmt(x0)}" y="{_fmt(y0)}" width="{_fmt(span_x+grow)}" height="{_fmt(span_y+grow)}" rx="{_fmt(rr)}" ry="{_fmt(rr)}" fill="{fill}"/>'])
                if grow <= 0.5:
                    output.append([f'<rect x="{_fmt(x0)}" y="{_fmt(y0)}" width="{_fmt(span_x+grow)}" height="{_fmt(span_y+grow)}" rx="{_fmt(rr)}" ry="{_fmt(rr)}" fill="{fill}" stroke="{fill}" stroke-width="1" vector-effect="non-scaling-stroke"/>'])
        return output
    return output


def _blacken(elements: list[str], fill: str) -> list[str]:
    return [element.replace(fill, "#000000") for element in elements]


def primitive_metrics(model: dict[str, Any], elements: list[str], fill: str, sw: int, sh: int) -> dict[str, Any] | None:
    synthetic = np.asarray([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    white = f'<path fill="#ffffff" d="M0 0H{sw}V{sh}H0Z"/>'
    black = [white] + _blacken(elements, fill)
    levels: list[dict[str, Any]] = []
    for factor in FACTORS:
        tw=max(16,int(round(sw*factor))); th=max(16,int(round(sh*factor)))
        rendered=_impl._render(sw,sh,black,synthetic,tw,th)
        if rendered is None:
            return None
        actual=rendered==0; expected=model_mask(model,sw,sh,tw,th)
        component_count=_impl._cc(actual)
        intersection=int(np.count_nonzero(actual & expected)); union=int(np.count_nonzero(actual | expected))
        iou=1.0 if union==0 else intersection/float(union)
        levels.append({"scale":f"{factor:g}x","size":[tw,th],"component_count":int(component_count),"iou":float(iou),"mismatch":float(np.mean(actual!=expected))})
    return {
        "levels":levels,
        "component_error":int(sum(abs(int(level["component_count"])-1) for level in levels)),
        "min_iou":float(min(float(level["iou"]) for level in levels)),
        "mean_iou":float(np.mean([float(level["iou"]) for level in levels])),
        "max_mismatch":float(max(float(level["mismatch"]) for level in levels)),
        "native_iou":float(levels[2]["iou"]),
    }


def _score(metrics: dict[str, Any], element_count: int) -> tuple[float,float,float,float,int]:
    return (float(metrics["component_error"]),-float(metrics["min_iou"]),float(metrics["max_mismatch"]),-float(metrics["mean_iou"]),int(element_count))


def calibrate_primitives(labels: np.ndarray, colors: np.ndarray, elements: list[str], regions: list[dict[str, Any]]) -> tuple[list[str],list[dict[str, Any]],dict[int,dict[str,Any]]]:
    node_map,_nodes,_root,_depth,_parent=_impl._core._connected_region_graph(labels,len(colors))
    parts=[]; cursor=0
    for region in regions:
        count=int(region.get("path_count") or 0); parts.append(list(elements[cursor:cursor+count])); cursor+=count
    if cursor != len(elements):
        return list(elements),[],{}
    sh,sw=labels.shape; reports=[]; repair_models={}
    for index,region in enumerate(regions):
        node_id=int(region["node_id"]); color_index=int(region["color_index"]); strategy=str(region.get("strategy") or "")
        model=infer_model(node_map==node_id,strategy)
        if model is None:
            continue
        if str(model["kind"])=="small_rect":
            repair_models[node_id]={"kind":"rect","x":model["x"],"y":model["y"],"width":model["width"],"height":model["height"]}
        elif str(model["kind"]) in {"line","ellipse"}:
            repair_models[node_id]=dict(model)
        red,green,blue=(int(value) for value in colors[color_index]); fill=f"#{red:02x}{green:02x}{blue:02x}"
        best_part=list(parts[index]); best_metrics=primitive_metrics(model,best_part,fill,sw,sh)
        best_score=_score(best_metrics,len(best_part)) if best_metrics is not None else None
        for candidate in candidate_elements(model,fill):
            metrics=primitive_metrics(model,candidate,fill,sw,sh)
            if metrics is None or float(metrics["native_iou"])+1e-12 < float(_impl._NATIVE_MIN_IOU):
                continue
            score=_score(metrics,len(candidate))
            if best_score is None or score < best_score:
                best_part=list(candidate); best_metrics=metrics; best_score=score
        changed=best_part!=parts[index]; old_count=len(parts[index]); parts[index]=best_part
        reports.append({"node_id":node_id,"color_index":color_index,"strategy":strategy,"model":model,"changed":changed,"selected":best_metrics,"element_delta":len(best_part)-old_count})
    return [element for part in parts for element in part],reports,repair_models


__all__=["FACTORS","calibrate_primitives","candidate_elements","infer_model","model_mask","primitive_metrics"]
