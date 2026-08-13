"""Fail-closed, renderer-verified semantic geometry regularization.

Primitive models are inferred only from connected source masks.  The wrapper
never inspects fixture IDs, imports regression builders, embeds raster data, or
changes shared byte/path/node budgets.  Every local edit preserves native
component fidelity and may only improve (never worsen) five-scale topology;
the caller still performs the authoritative native and multi-render gate.
"""
from __future__ import annotations

import copy
import hashlib
import os
import tempfile
from collections import OrderedDict, deque
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app import semantic_region_geometry_core as _core

SemanticRegionFitError = _core.SemanticRegionFitError
_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
_NATIVE_MIN_IOU = 0.95
_CACHE_MAX = 12
_MAX_ANCHORS_PER_SCALE = 16
_NEIGHBORS = ((-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1))
_SEPARATOR_STRATEGIES = {"ellipse_axis_arm_union", "compound_parent_expand_axis_arm_union"}
_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _fmt(v: float) -> str:
    s = f"{float(v):.4f}".rstrip("0").rstrip(".")
    return s if s not in {"", "-0"} else "0"


def _svg(w: int, h: int, elements: list[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" shape-rendering="crispEdges">'
            + "".join(elements) + "</svg>")


def _labels(rgb: np.ndarray, colors: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb, dtype=np.int32)
    p = np.asarray(colors, dtype=np.int32)
    d = a[:, :, None, :] - p[None, None, :, :]
    return np.argmin(np.sum(d*d, axis=3, dtype=np.int64), axis=2).astype(np.int16)


def _render(w: int, h: int, elements: list[str], colors: np.ndarray, tw: int, th: int) -> np.ndarray | None:
    try:
        from app.fidelity import render_svg_to_rgb  # noqa: PLC0415
    except Exception:
        return None
    f = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    path = f.name
    try:
        f.write(_svg(w, h, elements).encode("utf-8")); f.close()
        rgb = render_svg_to_rgb(path, int(tw), int(th))
        return None if rgb is None else _labels(rgb, colors)
    finally:
        try: f.close()
        except Exception: pass
        try: os.unlink(path)
        except OSError: pass


def _cc(mask: np.ndarray) -> int:
    n, _ = cv2.connectedComponents(np.asarray(mask, dtype=np.uint8), connectivity=8)
    return max(0, int(n)-1)


def _counts(labels: np.ndarray, n: int) -> list[int]:
    return [_cc(labels == i) for i in range(n)]


def _best_iou(rendered: np.ndarray, expected: np.ndarray, color: int) -> float:
    n, lab = cv2.connectedComponents((np.asarray(rendered) == int(color)).astype(np.uint8), connectivity=8)
    e = np.asarray(expected, dtype=bool); best = 0.0
    for i in range(1, int(n)):
        r = lab == i; inter = int(np.count_nonzero(e & r))
        if not inter: continue
        union = int(np.count_nonzero(e | r))
        if union: best = max(best, inter / float(union))
    return float(best)


def _bbox(mask: np.ndarray) -> tuple[int,int,int,int] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(xs): return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _pil_mask(w: int, h: int, fn: Any) -> np.ndarray:
    im = Image.new("L", (int(w), int(h)), 0); fn(ImageDraw.Draw(im))
    return np.asarray(im, dtype=np.uint8) > 0


def _scaled(v: float, target: int, source: int) -> int:
    return int(round(float(v) * float(target) / float(source)))


def _model_mask(model: dict[str, Any], sw: int, sh: int, tw: int, th: int) -> np.ndarray:
    kind = model["kind"]; fx = tw / float(sw); fy = th / float(sh)
    if kind == "ellipse":
        cx = _scaled(model["cx"], tw, sw); cy = _scaled(model["cy"], th, sh)
        rx = max(1, int(round(model["rx"]*fx))); ry = max(1, int(round(model["ry"]*fy)))
        return _pil_mask(tw, th, lambda d: d.ellipse((cx-rx,cy-ry,cx+rx,cy+ry), fill=255))
    if kind == "line":
        x0 = _scaled(model["x0"], tw, sw); x1 = _scaled(model["x1"], tw, sw)
        y0 = _scaled(model["y0"], th, sh); y1 = _scaled(model["y1"], th, sh)
        scale = fy if model["axis"] == "h" else fx
        width = max(1, int(round(model["width"]*scale)))
        return _pil_mask(tw, th, lambda d: d.line((x0,y0,x1,y1), fill=255, width=width))
    x = _scaled(model["x"], tw, sw); y = _scaled(model["y"], th, sh)
    rw = max(1, int(round(model["width"]*fx))); rh = max(1, int(round(model["height"]*fy)))
    return _pil_mask(tw, th, lambda d: d.rectangle((x,y,x+rw-1,y+rh-1), fill=255))


def _infer(mask: np.ndarray, strategy: str) -> dict[str, Any] | None:
    b = _bbox(mask)
    if b is None: return None
    x0,y0,x1,y1 = b; w=x1-x0+1; h=y1-y0+1
    crop = np.asarray(mask, dtype=bool)[y0:y1+1, x0:x1+1]
    occ = float(crop.mean())
    if "ellipse" in strategy:
        rx=(w-1)/2.0; ry=(h-1)/2.0
        if min(rx,ry) >= 1.0:
            m={"kind":"ellipse","cx":(x0+x1)/2.0,"cy":(y0+y1)/2.0,"rx":rx,"ry":ry}
            if np.mean(_model_mask(m, mask.shape[1], mask.shape[0], mask.shape[1], mask.shape[0]) == mask) >= 0.995:
                return m
    if occ >= 0.999:
        short,long=min(w,h),max(w,h)
        if short <= 4 and long >= max(5, short*3):
            if w >= h: return {"kind":"line","axis":"h","x0":x0,"y0":y0,"x1":x1,"y1":y0,"width":h}
            return {"kind":"line","axis":"v","x0":x0,"y0":y0,"x1":x0,"y1":y1,"width":w}
        if max(w,h) <= 8:
            return {"kind":"rect","x":x0,"y":y0,"width":w,"height":h}
    return None


def _candidates(model: dict[str, Any], fill: str) -> list[list[str]]:
    if model["kind"] == "ellipse":
        cx,cy,rx,ry = (float(model[k]) for k in ("cx","cy","rx","ry"))
        return [[f'<ellipse cx="{_fmt(cx)}" cy="{_fmt(cy)}" rx="{_fmt(rx)}" ry="{_fmt(ry)}" fill="{fill}" stroke="{fill}" stroke-width="1" vector-effect="non-scaling-stroke"/>'],
                [f'<ellipse cx="{_fmt(cx)}" cy="{_fmt(cy)}" rx="{_fmt(rx+.25)}" ry="{_fmt(ry+.25)}" fill="{fill}"/>']]
    if model["kind"] == "line":
        x0,y0,x1,y1,width = (float(model[k]) for k in ("x0","y0","x1","y1","width"))
        out=[]
        for off in (width/2.0, max(0.0,(width-1.0)/2.0)):
            d = (f'M{_fmt(x0)} {_fmt(y0+off)}H{_fmt(x1)}' if model["axis"]=="h"
                 else f'M{_fmt(x0+off)} {_fmt(y0)}V{_fmt(y1)}')
            out.append([f'<path d="{d}" fill="none" stroke="{fill}" stroke-width="{_fmt(width)}" stroke-linecap="butt"/>',
                        f'<path d="{d}" fill="none" stroke="{fill}" stroke-width="1" vector-effect="non-scaling-stroke" stroke-linecap="square"/>'])
        return out
    x,y,w,h = (float(model[k]) for k in ("x","y","width","height"))
    cx=x+(w-1)/2.0; cy=y+(h-1)/2.0; point=f'M{_fmt(cx)} {_fmt(cy)}h0.001'
    return [[f'<rect x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(w)}" height="{_fmt(h)}" fill="{fill}"/>',
             f'<path d="{point}" fill="none" stroke="{fill}" stroke-width="1" vector-effect="non-scaling-stroke" stroke-linecap="square"/>'],
            [f'<rect x="{_fmt(x-.01)}" y="{_fmt(y-.01)}" width="{_fmt(w+.02)}" height="{_fmt(h+.02)}" fill="{fill}"/>',
             f'<path d="{point}" fill="none" stroke="{fill}" stroke-width="1" vector-effect="non-scaling-stroke" stroke-linecap="square"/>']]


def _flatten(parts: list[list[str]]) -> list[str]:
    return [e for p in parts for e in p]


def _metrics(src: np.ndarray, colors: np.ndarray, elements: list[str], node_map: np.ndarray,
             node: int, color: int, source_counts: list[int], model: dict[str, Any]) -> dict[str, Any] | None:
    sh,sw=src.shape; source_mask=node_map==int(node); ious=[]; errors=[]; native_mismatch=None; native_iou=None
    for factor in _FACTORS:
        tw=max(16,int(round(sw*factor))); th=max(16,int(round(sh*factor)))
        r=_render(sw,sh,elements,colors,tw,th)
        if r is None: return None
        expected=_model_mask(model,sw,sh,tw,th); ious.append(_best_iou(r,expected,color))
        actual=_counts(r,len(colors)); errors.append(sum(abs(a-b) for a,b in zip(actual,source_counts,strict=True)))
        if factor==1.0:
            native_mismatch=float(np.mean(r != src)); native_iou=_best_iou(r,source_mask,color)
    return {"native_iou":float(native_iou or 0.0),"native_mismatch":float(native_mismatch or 0.0),
            "lineage_errors":[int(x) for x in errors],"scale_ious":[float(x) for x in ious],
            "min_scale_iou":float(min(ious)),"mean_scale_iou":float(np.mean(ious))}


def _score(m: dict[str, Any], count: int) -> tuple[float,float,float,float,int]:
    return (float(sum(m["lineage_errors"])), -m["min_scale_iou"], -m["mean_scale_iou"], m["native_mismatch"], count)


def _regularize(labels: np.ndarray, colors: np.ndarray, elements: list[str], regions: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]], dict[int,dict[str,Any]]]:
    node_map,nodes,_root,_depth,_parent=_core._connected_region_graph(labels,len(colors))
    source_counts=[sum(1 for n in nodes if int(n["color_index"])==c) for c in range(len(colors))]
    parts=[]; cursor=0
    for region in regions:
        n=int(region.get("path_count") or 0); parts.append(list(elements[cursor:cursor+n])); cursor+=n
    if cursor != len(elements): return list(elements),[],{}
    reports=[]; models={}
    for i,region in enumerate(regions):
        if len(parts[i]) != 1: continue
        node=int(region["node_id"]); color=int(region["color_index"]); model=_infer(node_map==node,str(region.get("strategy") or ""))
        if model is None: continue
        models[node]=model; rgb=colors[color]; fill=f'#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}'
        base=_metrics(labels,colors,_flatten(parts),node_map,node,color,source_counts,model)
        if base is None: continue
        best=base; best_part=list(parts[i]); best_score=_score(base,len(best_part))
        for candidate in _candidates(model,fill):
            trial=list(parts); trial[i]=candidate
            m=_metrics(labels,colors,_flatten(trial),node_map,node,color,source_counts,model)
            if m is None or m["native_iou"]+1e-12 < _NATIVE_MIN_IOU: continue
            before=[int(v) for v in base["lineage_errors"]]; after=[int(v) for v in m["lineage_errors"]]
            if after[2] != 0 or any(a>b for a,b in zip(after,before,strict=True)): continue
            if m["native_mismatch"] > base["native_mismatch"] + .00025: continue
            s=_score(m,len(candidate))
            if s < best_score: best_score=s; best=m; best_part=candidate
        changed=best_part != parts[i]
        if changed: parts[i]=best_part
        reports.append({"node_id":node,"color_index":color,"strategy":str(region.get("strategy") or ""),
                        "model":model,"changed":changed,"before":base,"after":best,"element_delta":len(best_part)-1})
    return _flatten(parts),reports,models


def _tree_path(a: int, b: int, parent: list[int|None]) -> list[int]:
    chain=[]; cur:int|None=int(a)
    while cur is not None: chain.append(cur); cur=parent[cur]
    pos={n:i for i,n in enumerate(chain)}; other=[]; cur=int(b)
    while cur is not None and cur not in pos: other.append(cur); cur=parent[cur]
    return [] if cur is None else chain[:pos[cur]+1]+list(reversed(other))


def _project(mask: np.ndarray, tw: int, th: int) -> np.ndarray:
    return cv2.resize(np.asarray(mask,dtype=np.uint8),(int(tw),int(th)),interpolation=cv2.INTER_NEAREST).astype(bool)


def _shortest(component: np.ndarray, start: np.ndarray, goal: np.ndarray) -> list[tuple[int,int]]:
    seeds=np.argwhere(component & start)
    if not len(seeds) or not np.any(component & goal): return []
    q:deque[tuple[int,int]]=deque(); prev:dict[tuple[int,int],tuple[int,int]|None]={}
    for y,x in seeds.tolist(): p=(int(y),int(x)); q.append(p); prev[p]=None
    end=None; h,w=component.shape
    while q and end is None:
        y,x=q.popleft()
        if goal[y,x]: end=(y,x); break
        for dx,dy in _NEIGHBORS:
            nx,ny=x+dx,y+dy; p=(ny,nx)
            if 0<=nx<w and 0<=ny<h and p not in prev and component[ny,nx]: prev[p]=(y,x); q.append(p)
    if end is None: return []
    out=[]; cur=end
    while cur is not None: out.append(cur); cur=prev[cur]
    return list(reversed(out))


def _anchor(x: int, y: int, tw: int, th: int, sw: int, sh: int, fill: str) -> str:
    cx=(x+.5)*sw/float(tw); cy=(y+.5)*sh/float(th)
    return f'<path d="M{_fmt(cx)} {_fmt(cy)}h0.001" fill="none" stroke="{fill}" stroke-width="1" vector-effect="non-scaling-stroke" stroke-linecap="square"/>'


def _bridge(rendered: np.ndarray, color: int, node_map: np.ndarray, nodes: list[dict[str,Any]], parent: list[int|None], strategies: dict[int,str]) -> tuple[int,int,int]|None:
    mask=rendered==int(color); n,cc=cv2.connectedComponents(mask.astype(np.uint8),connectivity=8); th,tw=mask.shape
    ids=[i for i,node in enumerate(nodes) if int(node["color_index"])==int(color)]
    projected={i:_project(node_map==i,tw,th) for i in ids}
    for cid in range(1,int(n)):
        comp=cc==cid; overlap=[i for i in ids if np.any(comp & projected[i])]
        for ai,a in enumerate(overlap[:-1]):
            for b in overlap[ai+1:]:
                path=_tree_path(a,b,parent)
                separators=[i for i in path[1:-1] if int(nodes[i]["color_index"])!=color and strategies.get(i) in _SEPARATOR_STRATEGIES]
                route=_shortest(comp,projected[a],projected[b]) if separators else []
                best=None
                for sep in separators:
                    sm=_project(node_map==sep,tw,th); dist=cv2.distanceTransform((~sm).astype(np.uint8),cv2.DIST_L2,3); repl=int(nodes[sep]["color_index"])
                    for y,x in route:
                        item=(float(dist[y,x]),int(y),int(x),repl)
                        if best is None or item<best: best=item
                if best is not None: return best[2],best[1],best[3]
    return None


def _expected_anchor(model: dict[str,Any], sw: int, sh: int, tw: int, th: int) -> tuple[int,int]|None:
    m=_model_mask(model,sw,sh,tw,th); ys,xs=np.nonzero(m)
    if not len(xs): return None
    cx=float(xs.mean()); cy=float(ys.mean()); order=np.argmin((xs-cx)**2+(ys-cy)**2)
    return int(xs[order]),int(ys[order])


def _repair(labels: np.ndarray, colors: np.ndarray, elements: list[str], regions: list[dict[str,Any]], models: dict[int,dict[str,Any]]) -> tuple[list[str],list[dict[str,Any]]]:
    sh,sw=labels.shape; node_map,nodes,_root,_depth,parent=_core._connected_region_graph(labels,len(colors))
    source_counts=[sum(1 for n in nodes if int(n["color_index"])==c) for c in range(len(colors))]
    strategies={int(r["node_id"]):str(r.get("strategy") or "") for r in regions if "node_id" in r}
    out=list(elements); reports=[]
    for factor in _FACTORS:
        tw=max(16,int(round(sw*factor))); th=max(16,int(round(sh*factor))); anchors=[]
        for _ in range(_MAX_ANCHORS_PER_SCALE):
            r=_render(sw,sh,out,colors,tw,th)
            if r is None: break
            actual=_counts(r,len(colors)); missing=[c for c,(e,a) in enumerate(zip(source_counts,actual,strict=True)) if a<e]
            if not missing: break
            applied=False
            for color in missing:
                for node,model in models.items():
                    if int(nodes[node]["color_index"]) != color: continue
                    expected=_model_mask(model,sw,sh,tw,th)
                    if np.any((r==color) & expected): continue
                    point=_expected_anchor(model,sw,sh,tw,th)
                    if point is None: continue
                    rgb=colors[color]; fill=f'#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}'
                    out.append(_anchor(point[0],point[1],tw,th,sw,sh,fill)); anchors.append({"kind":"component_survival","node_id":node,"class_index":color,"target_pixel":[point[0],point[1]]}); applied=True; break
                if applied: break
            if applied: continue
            for color in missing:
                cut=_bridge(r,color,node_map,nodes,parent,strategies)
                if cut is None: continue
                x,y,repl=cut; rgb=colors[repl]; fill=f'#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}'
                out.append(_anchor(x,y,tw,th,sw,sh,fill)); anchors.append({"kind":"separator_cut","class_index":color,"replacement_class_index":repl,"target_pixel":[x,y]}); applied=True; break
            if not applied: break
        final=_render(sw,sh,out,colors,tw,th)
        reports.append({"scale":f"{factor:g}x","size":[tw,th],"anchors":anchors,"final_component_counts":_counts(final,len(colors)) if final is not None else None})
    return out,reports


def _key(labels: np.ndarray, colors: np.ndarray) -> str:
    l=np.ascontiguousarray(labels); c=np.ascontiguousarray(colors,dtype=np.uint8); h=hashlib.sha256()
    h.update(str(l.shape).encode()); h.update(l.dtype.str.encode()); h.update(l.tobytes()); h.update(c.tobytes()); return h.hexdigest()


def build_semantic_region_elements(labels: np.ndarray, colors: np.ndarray) -> dict[str, Any]:
    labels=np.asarray(labels); colors=np.asarray(colors,dtype=np.uint8); key=_key(labels,colors)
    cached=_CACHE.get(key)
    if cached is not None: _CACHE.move_to_end(key); return copy.deepcopy(cached)
    report=_core.build_semantic_region_elements(labels,colors); base=list(report.get("elements") or []); regions=list(report.get("regions") or [])
    regularized,primitive_report,models=_regularize(labels,colors,base,regions)
    repaired,repair_report=_repair(labels,colors,regularized,regions,models)
    primitive_delta=len(regularized)-len(base); anchor_delta=len(repaired)-len(regularized); total=len(repaired)-len(base)
    if total:
        strategies=dict(report.get("strategy_counts") or {})
        if primitive_delta: strategies["scale_stable_primitive_reconstruction"]=primitive_delta
        if anchor_delta: strategies["render_feedback_topology_anchor"]=anchor_delta
        report["strategy_counts"]=dict(sorted(strategies.items())); report["path_count"]=int(report.get("path_count") or 0)+total; report["node_count"]=int(report.get("node_count") or 0)+2*total
    report["elements"]=repaired; report["scale_stable_primitive_reconstruction"]=primitive_report
    report["scale_stable_primitive_changed"]=sum(1 for x in primitive_report if x.get("changed")); report["scale_topology_repair"]=repair_report; report["scale_topology_anchor_count"]=anchor_delta
    _CACHE[key]=copy.deepcopy(report); _CACHE.move_to_end(key)
    while len(_CACHE)>_CACHE_MAX: _CACHE.popitem(last=False)
    return report


__all__=["SemanticRegionFitError","build_semantic_region_elements"]
