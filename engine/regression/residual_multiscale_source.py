"""Verified analytic multi-scale source contract for residual diagnostics.

Reference pixels are generated directly at the requested size from normalized
geometry. The source path never resizes a canonical raster. This intentionally
replaces the three rejected strategies: legacy fixed-coordinate builder replay,
AREA/Lanczos interpolation, and nearest-neighbour raster resizing.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from engine.regression.residual_error_metrics import SCALES, _finite, measure_residual_error
from app.source_truth import render_svg_to_rgba

SOURCE_SCALE_POLICY = "normalized_analytic_geometry_v1"
SCALE_SIZES = (64, 128, 256, 512, 1024)


@dataclass(frozen=True)
class AnalyticScene:
    image: Image.Image
    labels: np.ndarray
    geometry_sha256: str
    palette_mode: str
    source_basis: str


def _stable_sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scale(value: float, size: int, basis: int = 256) -> int:
    return int(round(float(value) * float(size) / float(basis)))


def _width(value: float, size: int, basis: int = 256) -> int:
    return max(1, _scale(value, size, basis))


def _canvas(size: int, rgba=(255, 255, 255, 255)) -> tuple[Image.Image, np.ndarray]:
    return Image.new("RGBA", (size, size), rgba), np.zeros((size, size), dtype=np.uint8)


def _draw_mask(size: int, command: str, args: tuple[Any, ...], **kwargs: Any) -> np.ndarray:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    getattr(draw, command)(*args, fill=255, **kwargs)
    return np.asarray(mask, dtype=np.uint8) > 0


def _paint(image: Image.Image, labels: np.ndarray, mask: np.ndarray, rgba: tuple[int, int, int, int], label: int) -> None:
    arr = np.asarray(image, dtype=np.uint8).copy()
    arr[mask] = np.asarray(rgba, dtype=np.uint8)
    image.paste(Image.fromarray(arr, mode="RGBA"))
    labels[mask] = np.uint8(label)


def _scene(name: str, image: Image.Image, labels: np.ndarray, *, palette_mode: str = "discrete", source_basis: str = "normalized_256") -> AnalyticScene:
    spec = {
        "fixture": name,
        "policy": SOURCE_SCALE_POLICY,
        "basis": source_basis,
        "semantic_geometry_version": 1,
    }
    return AnalyticScene(image=image, labels=labels, geometry_sha256=_stable_sha(spec), palette_mode=palette_mode, source_basis=source_basis)


def _gray_border_counter(size: int) -> AnalyticScene:
    image, labels = _canvas(size)
    draw = ImageDraw.Draw(image)
    ldraw = ImageDraw.Draw(Image.fromarray(labels, mode="L"))
    # Draw image and labels separately because labels encode semantic regions.
    gray_box = tuple(_scale(v, size) for v in (24, 36, 232, 220))
    black_box = tuple(_scale(v, size) for v in (38, 50, 218, 206))
    hole = tuple(_scale(v, size) for v in (86, 78, 170, 174))
    stem = tuple(_scale(v, size) for v in (122, 168, 144, 220))
    draw.rounded_rectangle(gray_box, radius=_width(38, size), fill=(150,150,150,255))
    draw.rounded_rectangle(black_box, radius=_width(30, size), fill=(15,15,15,255))
    draw.ellipse(hole, fill=(255,255,255,255)); draw.rectangle(stem, fill=(255,255,255,255))
    labimg = Image.new("L", (size,size), 0); ld=ImageDraw.Draw(labimg)
    ld.rounded_rectangle(gray_box, radius=_width(38,size), fill=1)
    ld.rounded_rectangle(black_box, radius=_width(30,size), fill=2)
    ld.ellipse(hole, fill=0); ld.rectangle(stem, fill=0)
    return _scene("_draw_gray_border_counter", image, np.asarray(labimg,dtype=np.uint8))


def _shared_boundary(size: int) -> AnalyticScene:
    image, labels = _canvas(size)
    draw=ImageDraw.Draw(image); lab=Image.new("L",(size,size),0); ld=ImageDraw.Draw(lab)
    left=tuple(_scale(v,size) for v in (24,32,128,224)); right=tuple(_scale(v,size) for v in (128,32,232,224)); circ=tuple(_scale(v,size) for v in (92,92,164,164))
    draw.rectangle(left,fill=(220,35,45,255)); ld.rectangle(left,fill=1)
    draw.rectangle(right,fill=(30,90,220,255)); ld.rectangle(right,fill=2)
    draw.ellipse(circ,fill=(245,195,25,255)); ld.ellipse(circ,fill=3)
    return _scene("_draw_shared_boundary", image, np.asarray(lab,dtype=np.uint8))


def _ring_holes(size: int) -> AnalyticScene:
    image, labels=_canvas(size); draw=ImageDraw.Draw(image); lab=Image.new("L",(size,size),0); ld=ImageDraw.Draw(lab)
    ops=[("ellipse",(20,20,116,116),1), ("ellipse",(46,46,90,90),0), ("rounded_rectangle",(134,22,232,116),1), ("ellipse",(158,44,208,94),0), ("ellipse",(44,142,212,230),1), ("ellipse",(78,162,178,212),0)]
    for kind, box, label in ops:
        b=tuple(_scale(v,size) for v in box); fill=(18,18,18,255) if label else (255,255,255,255)
        if kind=="rounded_rectangle":
            draw.rounded_rectangle(b,radius=_width(20,size),fill=fill); ld.rounded_rectangle(b,radius=_width(20,size),fill=label)
        else:
            draw.ellipse(b,fill=fill); ld.ellipse(b,fill=label)
    return _scene("_draw_ring_holes",image,np.asarray(lab,dtype=np.uint8))


def _monoline(size: int) -> AnalyticScene:
    image, labels=_canvas(size); draw=ImageDraw.Draw(image); lab=Image.new("L",(size,size),0); ld=ImageDraw.Draw(lab)
    def line(points,width):
        pts=tuple(_scale(v,size) for v in points); w=_width(width,size); draw.line(pts,fill=(0,0,0,255),width=w); ld.line(pts,fill=1,width=w)
    line((20,128,236,128),3); line((128,20,128,236),3)
    box=tuple(_scale(v,size) for v in (48,48,208,208)); w=_width(3,size); draw.rectangle(box,outline=(0,0,0,255),width=w); ld.rectangle(box,outline=1,width=w)
    ell=tuple(_scale(v,size) for v in (76,76,180,180)); draw.ellipse(ell,outline=(0,0,0,255),width=w); ld.ellipse(ell,outline=1,width=w)
    arc=tuple(_scale(v,size) for v in (92,92,164,164)); w2=_width(2,size); draw.arc(arc,20,310,fill=(0,0,0,255),width=w2); ld.arc(arc,20,310,fill=1,width=w2)
    return _scene("_draw_monoline",image,np.asarray(lab,dtype=np.uint8))


def _small_details(size: int) -> AnalyticScene:
    image, labels=_canvas(size); draw=ImageDraw.Draw(image); lab=Image.new("L",(size,size),0); ld=ImageDraw.Draw(lab)
    box=tuple(_scale(v,size) for v in (20,36,236,220)); w=_width(5,size); draw.rounded_rectangle(box,radius=_width(24,size),outline=(20,20,20,255),width=w); ld.rounded_rectangle(box,radius=_width(24,size),outline=1,width=w)
    for index,radius in enumerate((2,3,4,5,6,7)):
        x=48+index*30; y=88 if index%2==0 else 156
        cx=_scale(x,size); cy=_scale(y,size); rr=_width(radius,size); b=(cx-rr,cy-rr,cx+rr,cy+rr)
        draw.ellipse(b,fill=(20,20,20,255)); ld.ellipse(b,fill=1)
    for index in range(6):
        x=44+index*32; pts=tuple(_scale(v,size) for v in (x,112,x+14,112)); ww=_width(2,size); draw.line(pts,fill=(215,35,45,255),width=ww); ld.line(pts,fill=2,width=ww)
    return _scene("_draw_small_details",image,np.asarray(lab,dtype=np.uint8))


def _transparent_overlap(size: int) -> AnalyticScene:
    image=Image.new("RGBA",(size,size),(0,0,0,0)); lab=Image.new("L",(size,size),0)
    draw=ImageDraw.Draw(image); ld=ImageDraw.Draw(lab)
    b1=tuple(_scale(v,size) for v in (24,40,168,208)); b2=tuple(_scale(v,size) for v in (92,28,228,220)); hole=tuple(_scale(v,size) for v in (100,88,156,160))
    draw.ellipse(b1,fill=(30,150,245,150)); ld.ellipse(b1,fill=1)
    draw.rounded_rectangle(b2,radius=_width(28,size),fill=(250,45,85,190)); ld.rounded_rectangle(b2,radius=_width(28,size),fill=2)
    draw.ellipse(hole,fill=(255,255,255,0)); ld.ellipse(hole,fill=0)
    return _scene("_draw_transparent_overlap",image,np.asarray(lab,dtype=np.uint8),palette_mode="alpha_discrete")


def _lowres_badge(size: int) -> AnalyticScene:
    # 32x32 is the fixture's logical geometry grid. Each logical cell is scaled
    # as a normalized rectangle; no raster-resize operation is used.
    tiny=Image.new("RGBA",(32,32),(255,255,255,255)); d=ImageDraw.Draw(tiny)
    d.ellipse((2,2,29,29),fill=(25,25,25,255)); d.ellipse((7,7,24,24),fill=(235,235,235,255)); d.rectangle((14,5,17,27),fill=(190,25,35,255)); d.rectangle((5,14,27,17),fill=(190,25,35,255))
    src=np.asarray(tiny,dtype=np.uint8); image=Image.new("RGBA",(size,size),(255,255,255,255)); draw=ImageDraw.Draw(image); lab=Image.new("L",(size,size),0); ld=ImageDraw.Draw(lab)
    colors={(25,25,25,255):1,(235,235,235,255):2,(190,25,35,255):3,(255,255,255,255):0}
    for y in range(32):
        x=0
        while x<32:
            color=tuple(int(v) for v in src[y,x]); x1=x+1
            while x1<32 and np.array_equal(src[y,x1],src[y,x]): x1+=1
            left=int(round(x*size/32)); right=int(round(x1*size/32))-1; top=int(round(y*size/32)); bottom=int(round((y+1)*size/32))-1
            if right>=left and bottom>=top:
                draw.rectangle((left,top,right,bottom),fill=color); ld.rectangle((left,top,right,bottom),fill=colors[color])
            x=x1
    return _scene("_draw_lowres_badge",image,np.asarray(lab,dtype=np.uint8),source_basis="normalized_32_cell_geometry")


def _micro_component_ladder(size:int)->AnalyticScene:
    image,labels=_canvas(size); draw=ImageDraw.Draw(image); lab=Image.new("L",(size,size),0); ld=ImageDraw.Draw(lab); scale=size/256.0
    box=tuple(round(v*scale) for v in (24,34,232,150)); r=max(1,round(18*scale)); draw.rounded_rectangle(box,radius=r,fill=(24,24,24,255)); ld.rounded_rectangle(box,radius=r,fill=1)
    y=round(202*scale)
    for index,nominal_side in enumerate(range(1,9)):
        side=max(1,round(nominal_side*scale)); x=round((34+index*26)*scale); b=(x,y,x+side-1,y+side-1); draw.rectangle(b,fill=(24,24,24,255)); ld.rectangle(b,fill=1)
    return _scene("_draw_micro_component_ladder",image,np.asarray(lab,dtype=np.uint8))


def _thin_negative_space(size:int)->AnalyticScene:
    image,labels=_canvas(size); draw=ImageDraw.Draw(image); lab=Image.new("L",(size,size),0); ld=ImageDraw.Draw(lab); scale=size/256.0
    box=tuple(round(v*scale) for v in (20,28,236,228)); r=max(1,round(22*scale)); draw.rounded_rectangle(box,radius=r,fill=(18,18,18,255)); ld.rounded_rectangle(box,radius=r,fill=1)
    for nominal_width,center in enumerate((62,106,150,194),start=1):
        width=max(1,round(nominal_width*scale)); cx=round(center*scale); x0=cx-width//2; x1=x0+width-1; b=(x0,round(58*scale),x1,round(198*scale)); draw.rectangle(b,fill=(255,255,255,255)); ld.rectangle(b,fill=0)
    b=tuple(round(v*scale) for v in (86,90,170,174)); draw.ellipse(b,fill=(255,255,255,255)); ld.ellipse(b,fill=0)
    b=tuple(round(v*scale) for v in (91,95,165,169)); draw.ellipse(b,fill=(18,18,18,255)); ld.ellipse(b,fill=1)
    return _scene("_draw_thin_negative_space",image,np.asarray(lab,dtype=np.uint8))


def _hard_stop_gradient(size:int)->AnalyticScene:
    rgba=np.full((size,size,4),255,dtype=np.uint8); labels=np.zeros((size,size),dtype=np.uint8)
    y0=max(1,round(size*48/256)); y1=min(size,round(size*208/256)); x0=max(1,round(size*24/256)); stop=min(size-1,round(size*128/256)); x1=min(size,round(size*232/256)); span=max(1,stop-x0)
    start=np.asarray((225,40,35),dtype=np.float64); end=np.asarray((245,188,35),dtype=np.float64)
    for x in range(x0,stop):
        t=0.0 if span<=1 else (x-x0)/float(span-1); rgba[y0:y1,x,:3]=np.rint(start*(1-t)+end*t).astype(np.uint8)
    rgba[y0:y1,stop:x1,:3]=np.asarray((35,92,220),dtype=np.uint8); labels[y0:y1,x0:stop]=1; labels[y0:y1,stop:x1]=2
    return _scene("_draw_hard_stop_gradient",Image.fromarray(rgba,mode="RGBA"),labels,palette_mode="continuous_analytic")


def _soft_alpha_shadow(size:int)->AnalyticScene:
    yy,xx=np.mgrid[0:size,0:size].astype(np.float64); cx=size*.50; cy=size*.66; rx=max(1.0,size*.36); ry=max(1.0,size*.18); distance=((xx-cx)/rx)**2+((yy-cy)/ry)**2; shadow_alpha=np.rint(np.clip(1.0-distance,0.0,1.0)*105.0).astype(np.uint8)
    shadow=np.zeros((size,size,4),dtype=np.uint8); shadow[:,:,:3]=28; shadow[:,:,3]=shadow_alpha; image=Image.fromarray(shadow,mode="RGBA"); labels=np.zeros((size,size),dtype=np.uint8); labels[shadow_alpha>0]=1
    blue=Image.new("RGBA",(size,size),(0,0,0,0)); bd=ImageDraw.Draw(blue); b=tuple(round(size*v/256) for v in (24,48,168,208)); bd.ellipse(b,fill=(28,148,242,172)); image=Image.alpha_composite(image,blue); lm=Image.new("L",(size,size),0); ImageDraw.Draw(lm).ellipse(b,fill=1); labels[np.asarray(lm)>0]=2
    red=Image.new("RGBA",(size,size),(0,0,0,0)); rd=ImageDraw.Draw(red); b=tuple(round(size*v/256) for v in (92,34,228,214)); r=max(1,round(size*28/256)); rd.rounded_rectangle(b,radius=r,fill=(248,48,82,198)); image=Image.alpha_composite(image,red); lm=Image.new("L",(size,size),0); ImageDraw.Draw(lm).rounded_rectangle(b,radius=r,fill=1); labels[np.asarray(lm)>0]=3
    return _scene("_draw_soft_alpha_shadow",image,labels,palette_mode="continuous_alpha_analytic")


def _neutral_tone_steps(size:int)->AnalyticScene:
    image,labels=_canvas(size); draw=ImageDraw.Draw(image); lab=Image.new("L",(size,size),0); ld=ImageDraw.Draw(lab); scale=size/256.0
    layers=(((18,24,238,232),244),((34,40,222,216),230),((52,58,204,198),208),((76,82,180,174),142),((104,108,152,148),24))
    for idx,(box,tone) in enumerate(layers,1):
        b=tuple(round(v*scale) for v in box); r=max(1,round(12*scale)); draw.rounded_rectangle(b,radius=r,fill=(tone,tone,tone,255)); ld.rounded_rectangle(b,radius=r,fill=idx)
    return _scene("_draw_neutral_tone_steps",image,np.asarray(lab,dtype=np.uint8))


_ANALYTIC_BUILDERS: dict[str, Callable[[int], AnalyticScene]] = {
    "_draw_gray_border_counter": _gray_border_counter,
    "_draw_shared_boundary": _shared_boundary,
    "_draw_ring_holes": _ring_holes,
    "_draw_monoline": _monoline,
    "_draw_small_details": _small_details,
    "_draw_transparent_overlap": _transparent_overlap,
    "_draw_lowres_badge": _lowres_badge,
    "_draw_micro_component_ladder": _micro_component_ladder,
    "_draw_thin_negative_space": _thin_negative_space,
    "_draw_hard_stop_gradient": _hard_stop_gradient,
    "_draw_soft_alpha_shadow": _soft_alpha_shadow,
    "_draw_neutral_tone_steps": _neutral_tone_steps,
}


def build_analytic_scene(source_builder: Callable[[int], Image.Image], size: int) -> AnalyticScene:
    name = getattr(source_builder, "__name__", "")
    builder = _ANALYTIC_BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"unverified fixture builder: {name or '<anonymous>'}")
    if int(size) not in SCALE_SIZES:
        raise ValueError(f"unsupported verified scale size: {size}")
    return builder(int(size))


def _hole_count(mask: np.ndarray) -> int:
    contours, hierarchy = cv2.findContours(np.asarray(mask,dtype=np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None: return 0
    return sum(1 for item in hierarchy[0] if int(item[3]) >= 0)


def _topology_signature(labels: np.ndarray) -> dict[str, Any]:
    lab=np.asarray(labels,dtype=np.uint8); components=0; holes=0; adjacency:set[tuple[int,int]]=set()
    for cls in sorted(int(v) for v in np.unique(lab) if int(v)!=0):
        count,_=cv2.connectedComponents((lab==cls).astype(np.uint8),connectivity=8); components += max(0,count-1); holes += _hole_count(lab==cls)
    left,right=lab[:,:-1],lab[:,1:]; up,down=lab[:-1,:],lab[1:,:]
    for a,b in zip(left[left!=right].tolist(),right[left!=right].tolist()):
        if a and b: adjacency.add(tuple(sorted((int(a),int(b)))))
    for a,b in zip(up[up!=down].tolist(),down[up!=down].tolist()):
        if a and b: adjacency.add(tuple(sorted((int(a),int(b)))))
    return {"component_count":components,"hole_count":holes,"adjacency_pairs":[list(p) for p in sorted(adjacency)]}


def validate_analytic_scale_contract(source_builder: Callable[[int], Image.Image], sizes: Iterable[int] = SCALE_SIZES) -> dict[str, Any]:
    levels=[]; failures=[]; geometry_hashes=set(); topology_signatures=[]; discrete_palettes=[]
    for size in tuple(int(v) for v in sizes):
        scene=build_analytic_scene(source_builder,size); rgba=np.asarray(scene.image.convert("RGBA"),dtype=np.uint8); geometry_hashes.add(scene.geometry_sha256); topo=_topology_signature(scene.labels); topology_signatures.append(topo)
        palette=np.unique(rgba.reshape(-1,4),axis=0) if scene.palette_mode in {"discrete","alpha_discrete"} else None
        if palette is not None: discrete_palettes.append(_stable_sha(palette.tolist()))
        levels.append({"size":size,"shape":[int(v) for v in rgba.shape],"geometry_sha256":scene.geometry_sha256,"palette_mode":scene.palette_mode,"palette_sha256":(_stable_sha(palette.tolist()) if palette is not None else None),"topology":topo})
    if len(geometry_hashes)!=1: failures.append("geometry_identity_drift")
    topology_required = bool(levels) and levels[0]["palette_mode"] != "continuous_alpha_analytic"
    if topology_required and topology_signatures and any(item!=topology_signatures[0] for item in topology_signatures[1:]): failures.append("semantic_topology_drift")
    if discrete_palettes and any(item!=discrete_palettes[0] for item in discrete_palettes[1:]): failures.append("palette_class_drift")
    return {"policy":SOURCE_SCALE_POLICY,"verified":not failures,"failures":failures,"geometry_sha256":next(iter(geometry_hashes),None),"levels":levels}


def measure_multiscale_svg_from_base(svg_path: Any, source_builder: Callable[[int], Image.Image], *, base_size: int = 256, scales: Iterable[float] = SCALES) -> dict[str, Any]:
    requested_scales=tuple(float(s) for s in scales); expected_sizes=tuple(max(16,int(round(base_size*s))) for s in requested_scales)
    contract=validate_analytic_scale_contract(source_builder,expected_sizes); levels=[]; missing=[]
    for scale,size in zip(requested_scales,expected_sizes,strict=True):
        scene=build_analytic_scene(source_builder,size); source=np.asarray(scene.image.convert("RGBA"),dtype=np.uint8); rendered=render_svg_to_rgba(svg_path,size,size); label=f"{scale:g}x"
        if rendered is None: missing.append(label); continue
        metrics=measure_residual_error(source,rendered); boundary=metrics.get("boundary_p95_px"); allowance=max(math.sqrt(2.0),0.75*scale)
        levels.append({"scale":label,"size":size,"source_geometry_sha256":scene.geometry_sha256,"source_topology":_topology_signature(scene.labels),"de00_p95":metrics["de00_p95"],"visible_error_ratio":metrics["visible_error_ratio"],"boundary_p95_px":boundary,"boundary_quantization_allowance_px":allowance,"normalized_boundary_excess_px":(max(0.0,float(boundary)-allowance)/scale if _finite(boundary) else None),"source_component_recall":metrics["source_component_recall"],"render_component_precision":metrics["render_component_precision"],"small_component_recall":metrics["small_component_recall"],"min_component_iou":metrics["min_component_iou"],"unmatched_source_count":metrics["unmatched_source_count"],"unmatched_render_count":metrics["unmatched_render_count"],"topology":metrics.get("topology")})
    def vals(key): return [float(level[key]) for level in levels if _finite(level.get(key))]
    recalls=vals("source_component_recall"); small=vals("small_component_recall"); excess=vals("normalized_boundary_excess_px"); visible=vals("visible_error_ratio")
    return {"scales_requested":[f"{s:g}x" for s in requested_scales],"missing_scales":missing,"levels":levels,"min_source_component_recall":min(recalls) if recalls else None,"min_small_component_recall":min(small) if small else None,"max_normalized_boundary_excess_px":max(excess) if excess else None,"max_visible_error_ratio":max(visible) if visible else None,"all_scales_measured":not missing and len(levels)==len(requested_scales),"source_scale_policy":SOURCE_SCALE_POLICY,"source_contract_verified":contract["verified"],"source_contract_failures":contract["failures"],"source_geometry_sha256":contract["geometry_sha256"],"source_contract":contract}
