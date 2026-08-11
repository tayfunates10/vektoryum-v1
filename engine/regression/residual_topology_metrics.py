"""Topology-aware residual hard stops for diagnostic evidence.

This module is production-neutral. It measures semantic topology that global
SSIM/boundary aggregates can hide: hole/ring loss, adjacency loss, and common
shared-boundary gap/double-line/overlap/drift defects. Missing measurements fail
closed when appended to the near-zero diagnostic contract.

Shared-boundary measurements explicitly tolerate one raster-quantization pixel
so an otherwise native-PASS vector is not failed merely because antialiasing
moves a classified boundary by one pixel. Defects must exceed that allowance;
raw drift/depth are retained as evidence.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

import cv2
import numpy as np

from app.graph_source import canonical_segmentation
from app.palette_ops import classify_rgb
from app.source_truth import composite_rgba

TOPOLOGY_POLICY_VERSION = "vektoryum-topology-hard-stop-v2"
SHARED_BOUNDARY_RASTER_ALLOWANCE_PX = 1.0
SHARED_BOUNDARY_OVERLAP_ALLOWANCE_PX = math.sqrt(2.0)
TOPOLOGY_BLOCKER_CODES = (
    "topology_metric_missing",
    "topology_hole_loss",
    "topology_ring_break",
    "topology_adjacency_loss",
    "shared_boundary_metric_missing",
    "shared_boundary_gap",
    "shared_boundary_double_line",
    "shared_boundary_overlap",
    "shared_boundary_drift",
)


def _background_label(labels: np.ndarray) -> int:
    lab=np.asarray(labels,dtype=np.uint8); h,w=lab.shape; sy=max(1,round(h*.04)); sx=max(1,round(w*.04))
    corners=np.concatenate((lab[:sy,:sx].ravel(),lab[:sy,-sx:].ravel(),lab[-sy:,:sx].ravel(),lab[-sy:,-sx:].ravel()))
    return int(np.argmax(np.bincount(corners.astype(np.int64))))


def _class_stats(labels: np.ndarray, background: int) -> dict[int, dict[str,int]]:
    result={}
    for cls in sorted(int(v) for v in np.unique(labels) if int(v)!=background):
        mask=(labels==cls).astype(np.uint8); count,_=cv2.connectedComponents(mask,connectivity=8)
        _contours,hierarchy=cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        holes=0 if hierarchy is None else sum(1 for item in hierarchy[0] if int(item[3])>=0)
        result[cls]={"components":max(0,int(count)-1),"holes":int(holes),"area":int(mask.sum())}
    return result


def _adjacency_counts(labels: np.ndarray, background: int) -> Counter[tuple[int,int]]:
    lab=np.asarray(labels,dtype=np.uint8); counts:Counter[tuple[int,int]]=Counter()
    left,right=lab[:,:-1],lab[:,1:]
    ys,xs=np.nonzero((left!=right)&(left!=background)&(right!=background))
    for y,x in zip(ys.tolist(),xs.tolist(),strict=True): counts[tuple(sorted((int(left[y,x]),int(right[y,x]))))]+=1
    up,down=lab[:-1,:],lab[1:,:]
    ys,xs=np.nonzero((up!=down)&(up!=background)&(down!=background))
    for y,x in zip(ys.tolist(),xs.tolist(),strict=True): counts[tuple(sorted((int(up[y,x]),int(down[y,x]))))]+=1
    return counts


def _compress(values: Iterable[int]) -> list[int]:
    out=[]
    for value in values:
        value=int(value)
        if not out or out[-1]!=value: out.append(value)
    return out


def _runs(values: Iterable[int]) -> list[tuple[int,int,int,int]]:
    seq=[int(v) for v in values]
    if not seq: return []
    out=[]; start=0; current=seq[0]
    for index,value in enumerate(seq[1:],1):
        if value!=current:
            out.append((current,start,index-1,index-start)); current=value; start=index
    out.append((current,start,len(seq)-1,len(seq)-start))
    return out


def _scan_interface(source: np.ndarray, rendered: np.ndarray, a: int, b: int, background: int) -> dict[str, Any]:
    h,w=source.shape; radius=max(3,round(min(h,w)*3/256)); total=gap=double=0; raw_drifts=[]; drift_excess=[]; matched=0

    def inspect(seq: list[int], expected_left: int, expected_right: int, seam_index: int) -> None:
        nonlocal total,gap,double,matched
        total+=1
        # A one-pixel non-semantic seam is raster quantization, not a real gap.
        for value,start,end,length in _runs(seq):
            if value not in (a,b) and start<=seam_index+1 and end>=seam_index and length>SHARED_BOUNDARY_RASTER_ALLOWANCE_PX:
                gap+=1; break
        # Ignore one-pixel A/B islands caused by antialias classification.
        pair_runs=_runs(v for v in seq if v in (a,b))
        if len(pair_runs)>=3 and any(length>SHARED_BOUNDARY_RASTER_ALLOWANCE_PX for _value,_start,_end,length in pair_runs[1:-1]):
            double+=1

        best=None
        for i in range(len(seq)-1):
            if seq[i]==expected_left and seq[i+1]==expected_right:
                d=abs((i+0.5)-(seam_index+0.5)); best=d if best is None else min(best,d)
        # Tolerate a single classified background/third-color bridge at the seam.
        if best is None:
            for i in range(len(seq)):
                if seq[i]!=expected_left: continue
                for j in range(i+1,min(len(seq),i+3)):
                    if seq[j]!=expected_right: continue
                    middle=seq[i+1:j]
                    if len(middle)<=SHARED_BOUNDARY_RASTER_ALLOWANCE_PX and all(v not in (a,b) for v in middle):
                        d=abs(((i+j)/2.0)-(seam_index+0.5)); best=d if best is None else min(best,d)
        if best is not None:
            matched+=1; raw_drifts.append(float(best)); drift_excess.append(max(0.0,float(best)-SHARED_BOUNDARY_RASTER_ALLOWANCE_PX))

    left,right=source[:,:-1],source[:,1:]
    mask=((left==a)&(right==b))|((left==b)&(right==a))
    ys,xs=np.nonzero(mask)
    for y,x in zip(ys.tolist(),xs.tolist(),strict=True):
        lo=max(0,x-radius+1); hi=min(w,x+radius+1)
        seq=rendered[y,lo:hi].tolist(); seam=x-lo
        if seam<0 or seam+1>=len(seq): continue
        inspect(seq,int(source[y,x]),int(source[y,x+1]),seam)

    up,down=source[:-1,:],source[1:,:]
    mask=((up==a)&(down==b))|((up==b)&(down==a))
    ys,xs=np.nonzero(mask)
    for y,x in zip(ys.tolist(),xs.tolist(),strict=True):
        lo=max(0,y-radius+1); hi=min(h,y+radius+1)
        seq=rendered[lo:hi,x].tolist(); seam=y-lo
        if seam<0 or seam+1>=len(seq): continue
        inspect(seq,int(source[y,x]),int(source[y+1,x]),seam)

    # Overlap uses Euclidean penetration depth instead of a 1-D seam sample.
    # A diagonal/curved antialiased boundary can look two pixels deep in a row
    # while still being only sqrt(2) from the true source boundary.
    source_a=(source==a).astype(np.uint8); source_b=(source==b).astype(np.uint8)
    dist_a=cv2.distanceTransform(source_a,cv2.DIST_L2,5); dist_b=cv2.distanceTransform(source_b,cv2.DIST_L2,5)
    wrong_b_in_a=(rendered==b)&(source==a); wrong_a_in_b=(rendered==a)&(source==b)
    deep_overlap=(wrong_b_in_a&(dist_a>SHARED_BOUNDARY_OVERLAP_ALLOWANCE_PX+1e-6))|(wrong_a_in_b&(dist_b>SHARED_BOUNDARY_OVERLAP_ALLOWANCE_PX+1e-6))
    max_overlap_depth=max(
        float(dist_a[wrong_b_in_a].max()) if np.any(wrong_b_in_a) else 0.0,
        float(dist_b[wrong_a_in_b].max()) if np.any(wrong_a_in_b) else 0.0,
    )

    return {
        "pair":[int(a),int(b)],
        "sample_count":int(total),
        "raster_allowance_px":SHARED_BOUNDARY_RASTER_ALLOWANCE_PX,
        "overlap_allowance_px":SHARED_BOUNDARY_OVERLAP_ALLOWANCE_PX,
        "gap_ratio":(float(gap/total) if total else None),
        "double_line_ratio":(float(double/total) if total else None),
        "overlap_ratio":(float(int(deep_overlap.sum())/total) if total else None),
        "max_overlap_depth_px":float(max_overlap_depth),
        "raw_drift_p95_px":(float(np.percentile(np.asarray(raw_drifts,dtype=np.float64),95)) if raw_drifts else None),
        "drift_p95_px":(float(np.percentile(np.asarray(drift_excess,dtype=np.float64),95)) if drift_excess else None),
        "matched_transition_ratio":(float(matched/total) if total else None),
    }


def measure_label_topology(source_labels: np.ndarray, render_labels: np.ndarray) -> dict[str, Any]:
    source=np.asarray(source_labels,dtype=np.uint8); rendered=np.asarray(render_labels,dtype=np.uint8)
    if source.shape!=rendered.shape: raise ValueError("topology label shapes must match")
    background=_background_label(source); source_stats=_class_stats(source,background); render_stats=_class_stats(rendered,background)
    source_adj=_adjacency_counts(source,background); render_adj=_adjacency_counts(rendered,background)
    hole_loss=[]; ring_break=[]
    for cls,stats in source_stats.items():
        got=render_stats.get(cls,{"components":0,"holes":0,"area":0})
        if got["holes"]<stats["holes"]: hole_loss.append({"class_id":cls,"source_holes":stats["holes"],"render_holes":got["holes"]})
        if stats["holes"]>0 and got["components"]>stats["components"]: ring_break.append({"class_id":cls,"source_components":stats["components"],"render_components":got["components"]})
    min_contact=max(2,round(min(source.shape)*.02)); adjacency_loss=[]; scans=[]
    for pair,count in sorted(source_adj.items()):
        if count<min_contact: continue
        if render_adj.get(pair,0)==0: adjacency_loss.append({"pair":list(pair),"source_contact_px":int(count)})
        scans.append(_scan_interface(source,rendered,pair[0],pair[1],background))
    def finite_vals(key): return [float(x[key]) for x in scans if isinstance(x.get(key),(int,float)) and math.isfinite(float(x[key]))]
    gaps=finite_vals("gap_ratio"); doubles=finite_vals("double_line_ratio"); overlaps=finite_vals("overlap_ratio"); drifts=finite_vals("drift_p95_px"); raw_drifts=finite_vals("raw_drift_p95_px"); overlap_depths=finite_vals("max_overlap_depth_px"); matched=finite_vals("matched_transition_ratio")
    shared_applicable=bool(scans)
    shared={
        "applicable":shared_applicable,
        "pair_count":len(scans),
        "raster_allowance_px":SHARED_BOUNDARY_RASTER_ALLOWANCE_PX,
        "overlap_allowance_px":SHARED_BOUNDARY_OVERLAP_ALLOWANCE_PX,
        "pairs":scans,
        "max_gap_ratio":max(gaps) if gaps else (0.0 if not shared_applicable else None),
        "max_double_line_ratio":max(doubles) if doubles else (0.0 if not shared_applicable else None),
        "max_overlap_ratio":max(overlaps) if overlaps else (0.0 if not shared_applicable else None),
        "max_overlap_depth_px":max(overlap_depths) if overlap_depths else (0.0 if not shared_applicable else None),
        "max_raw_drift_p95_px":max(raw_drifts) if raw_drifts else (0.0 if not shared_applicable else None),
        "max_drift_p95_px":max(drifts) if drifts else (0.0 if not shared_applicable else None),
        "min_matched_transition_ratio":min(matched) if matched else (1.0 if not shared_applicable else None),
    }
    complete=all(isinstance(shared[k],(int,float)) for k in ("max_gap_ratio","max_double_line_ratio","max_overlap_ratio","max_drift_p95_px","min_matched_transition_ratio"))
    return {
        "policy_version":TOPOLOGY_POLICY_VERSION,
        "complete":bool(complete),
        "background_class_id":background,
        "source_classes":source_stats,
        "render_classes":render_stats,
        "source_adjacency":[{"pair":list(pair),"contact_px":int(count)} for pair,count in sorted(source_adj.items())],
        "render_adjacency":[{"pair":list(pair),"contact_px":int(count)} for pair,count in sorted(render_adj.items())],
        "hole_loss_count":len(hole_loss),"hole_loss":hole_loss,
        "ring_break_count":len(ring_break),"ring_break":ring_break,
        "adjacency_loss_count":len(adjacency_loss),"adjacency_loss":adjacency_loss,
        "shared_boundary":shared,
    }


def measure_topology_residual(source_rgba: np.ndarray, render_rgba: np.ndarray, *, palette_size: int=8) -> dict[str, Any]:
    source=np.asarray(source_rgba,dtype=np.uint8); rendered=np.asarray(render_rgba,dtype=np.uint8)
    if source.shape!=rendered.shape: raise ValueError("topology RGBA shapes must match")
    source_rgb=composite_rgba(source,255); render_rgb=composite_rgba(rendered,255); unique_count=int(np.unique(source_rgb.reshape(-1,3),axis=0).shape[0]); k=max(1,min(int(palette_size),unique_count))
    source_labels,palette=canonical_segmentation(source_rgb,k=k); render_labels=classify_rgb(render_rgb,palette.astype(np.float32)).astype(np.uint8)
    result=measure_label_topology(source_labels,render_labels); result["palette_size"]=int(len(palette)); return result


def _check(code: str, value: Any, direction: str, target: float|bool) -> dict[str, Any]:
    measured=isinstance(value,(int,float,bool)) and not (isinstance(value,float) and not math.isfinite(value))
    if direction=="equals": passed=measured and value==target
    elif direction=="max": passed=measured and float(value)<=float(target)
    else: passed=measured and float(value)>=float(target)
    return {"code":code,"measured":bool(measured),"value":value if measured else None,"direction":direction,"target":target,"gap":None,"passed":bool(passed)}


def extend_near_zero_contract(contract: dict[str, Any], residual: dict[str, Any]|None, multi_scale: dict[str, Any]|None=None) -> dict[str, Any]:
    result=dict(contract); checks=list(result.get("checks") or []); topology=(residual or {}).get("topology") or {}
    checks.extend([
        _check("topology_metric_missing",topology.get("complete"),"equals",True),
        _check("topology_hole_loss",topology.get("hole_loss_count"),"max",0),
        _check("topology_ring_break",topology.get("ring_break_count"),"max",0),
        _check("topology_adjacency_loss",topology.get("adjacency_loss_count"),"max",0),
    ])
    shared=topology.get("shared_boundary") or {}
    if shared.get("applicable") is True:
        shared_complete = all(
            isinstance(shared.get(key), (int, float))
            and math.isfinite(float(shared[key]))
            for key in (
                "max_gap_ratio",
                "max_double_line_ratio",
                "max_overlap_ratio",
                "max_drift_p95_px",
                "min_matched_transition_ratio",
            )
        )
        checks.extend([
            _check("shared_boundary_metric_missing", shared_complete, "equals", True),
            _check("shared_boundary_gap",shared.get("max_gap_ratio"),"max",0.0),
            _check("shared_boundary_double_line",shared.get("max_double_line_ratio"),"max",0.0),
            _check("shared_boundary_overlap",shared.get("max_overlap_ratio"),"max",0.01),
            _check("shared_boundary_drift",shared.get("max_drift_p95_px"),"max",0.75),
            _check("shared_boundary_transition_recall",shared.get("min_matched_transition_ratio"),"min",1.0),
        ])
    if multi_scale is not None:
        checks.append(_check("multi_scale_source_contract_verified",multi_scale.get("source_contract_verified"),"equals",True))
        levels=multi_scale.get("levels") or []
        topology_levels=[level.get("topology") for level in levels]
        checks.append(_check("multi_scale_topology_complete",bool(levels) and all((item or {}).get("complete") is True for item in topology_levels),"equals",True))
    blockers=[str(item["code"]) for item in checks if not item.get("passed")]
    result["checks"]=checks; result["blocker_codes"]=blockers; result["ready"]=not blockers; result["topology_policy_version"]=TOPOLOGY_POLICY_VERSION; result["production_gate"]=False
    return result
