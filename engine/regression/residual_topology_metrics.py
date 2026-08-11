"""Topology-aware residual hard stops for diagnostic evidence.

This module is production-neutral. It measures semantic topology that global
SSIM/boundary aggregates can hide: hole/ring loss, adjacency loss, and common
shared-boundary gap/double-line/overlap/drift defects. Missing measurements fail
closed when appended to the near-zero diagnostic contract.
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

TOPOLOGY_POLICY_VERSION = "vektoryum-topology-hard-stop-v1"
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


def _scan_interface(source: np.ndarray, rendered: np.ndarray, a: int, b: int, background: int) -> dict[str, Any]:
    h,w=source.shape; radius=max(3,round(min(h,w)*3/256)); total=gap=double=overlap=0; drifts=[]

    def inspect(seq: list[int], expected_left: int, expected_right: int, seam_index: int) -> None:
        nonlocal total,gap,double,overlap
        total+=1
        core_left=seq[seam_index]; core_right=seq[seam_index+1]
        if core_left==background or core_right==background or (core_left not in (a,b) or core_right not in (a,b)):
            gap+=1
        if core_left==expected_right or core_right==expected_left:
            overlap+=1
        pair_only=[v for v in seq if v in (a,b)]
        compact=_compress(pair_only)
        if len(compact)>=3:
            double+=1
        best=None
        for i in range(len(seq)-1):
            if {seq[i],seq[i+1]}=={a,b}:
                # transition lies between i and i+1; seam is between seam_index and +1
                d=abs((i+0.5)-(seam_index+0.5))
                best=d if best is None else min(best,d)
        if best is not None: drifts.append(float(best))

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

    return {
        "pair":[int(a),int(b)],
        "sample_count":int(total),
        "gap_ratio":(float(gap/total) if total else None),
        "double_line_ratio":(float(double/total) if total else None),
        "overlap_ratio":(float(overlap/total) if total else None),
        "drift_p95_px":(float(np.percentile(np.asarray(drifts,dtype=np.float64),95)) if drifts else None),
        "matched_transition_ratio":(float(len(drifts)/total) if total else None),
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
    gaps=finite_vals("gap_ratio"); doubles=finite_vals("double_line_ratio"); overlaps=finite_vals("overlap_ratio"); drifts=finite_vals("drift_p95_px"); matched=finite_vals("matched_transition_ratio")
    shared_applicable=bool(scans)
    shared={
        "applicable":shared_applicable,
        "pair_count":len(scans),
        "pairs":scans,
        "max_gap_ratio":max(gaps) if gaps else (0.0 if not shared_applicable else None),
        "max_double_line_ratio":max(doubles) if doubles else (0.0 if not shared_applicable else None),
        "max_overlap_ratio":max(overlaps) if overlaps else (0.0 if not shared_applicable else None),
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
