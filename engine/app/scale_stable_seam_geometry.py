"""Five-scale stabilization for nested semantic paint boundaries."""
from __future__ import annotations
from typing import Any
import numpy as np
from PIL import Image, ImageDraw
from app import semantic_region_geometry_impl as _impl

FACTORS=(0.25,0.5,1.0,2.0,4.0)

def _fmt(v:float)->str: return _impl._fmt(float(v))
def _bbox(mask:np.ndarray):
    ys,xs=np.nonzero(np.asarray(mask,dtype=bool))
    return None if not len(xs) else (int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max()))
def _pil(w:int,h:int,fn:Any)->np.ndarray:
    im=Image.new('L',(int(w),int(h)),0); fn(ImageDraw.Draw(im)); return np.asarray(im,dtype=np.uint8)>0
def _iou(a:np.ndarray,b:np.ndarray)->float:
    aa=np.asarray(a,dtype=bool); bb=np.asarray(b,dtype=bool); u=int(np.count_nonzero(aa|bb))
    return 1.0 if u==0 else float(np.count_nonzero(aa&bb)/u)

def _shape(mask:np.ndarray)->dict[str,Any]|None:
    src=np.asarray(mask,dtype=bool); outer=np.asarray(_impl._core._filled_external_region(src),dtype=bool); box=_bbox(outer)
    if box is None:return None
    x0,y0,x1,y1=box; w=x1-x0+1; h=y1-y0+1
    if min(w,h)<5:return None
    best=None
    for r in range(1,max(1,min(w,h)//2)+1):
        cand=_pil(src.shape[1],src.shape[0],lambda d,rr=r:d.rounded_rectangle((x0,y0,x1,y1),radius=rr,fill=255)); s=_iou(cand,outer)
        if best is None or s>best[0]:best=(s,r)
        if s==1.0:break
    if best is not None and best[0]>=0.995:
        return {'kind':'rounded_rect','x0':x0,'y0':y0,'x1':x1,'y1':y1,'radius':int(best[1]),'source_fit_iou':float(best[0])}
    cx=(x0+x1)/2.0; cy=(y0+y1)/2.0; rx=(x1-x0)/2.0; ry=(y1-y0)/2.0
    if min(rx,ry)<1:return None
    cand=_pil(src.shape[1],src.shape[0],lambda d:d.ellipse((cx-rx,cy-ry,cx+rx,cy+ry),fill=255)); s=_iou(cand,outer)
    return {'kind':'ellipse','cx':cx,'cy':cy,'rx':rx,'ry':ry,'source_fit_iou':float(s)} if s>=0.995 else None

def _seam(model:dict[str,Any],stroke:str,width:float,non_scaling:bool)->str:
    ve=' vector-effect="non-scaling-stroke"' if non_scaling else ''
    common=f'fill="none" stroke="{stroke}" stroke-width="{_fmt(width)}"{ve} stroke-linejoin="round"'
    if model['kind']=='rounded_rect':
        x0,y0,x1,y1,r=(float(model[k]) for k in ('x0','y0','x1','y1','radius'))
        return f'<rect x="{_fmt(x0)}" y="{_fmt(y0)}" width="{_fmt(x1-x0+1)}" height="{_fmt(y1-y0+1)}" rx="{_fmt(r)}" ry="{_fmt(r)}" {common}/>'
    cx,cy,rx,ry=(float(model[k]) for k in ('cx','cy','rx','ry'))
    return f'<ellipse cx="{_fmt(cx)}" cy="{_fmt(cy)}" rx="{_fmt(rx)}" ry="{_fmt(ry)}" {common}/>'

def _metrics(labels:np.ndarray,colors:np.ndarray,elements:list[str],source_counts:list[int],node_map=None,node_id=None,color_index=None):
    sh,sw=labels.shape; errors=[]; counts=[]; native_mismatch=None; native_iou=None
    for factor in FACTORS:
        tw=max(16,int(round(sw*factor))); th=max(16,int(round(sh*factor))); rendered=_impl._render(sw,sh,elements,colors,tw,th)
        if rendered is None:return None
        actual=[int(v) for v in _impl._counts(rendered,len(colors))]; counts.append(actual); errors.append(int(sum(abs(a-b) for a,b in zip(actual,source_counts,strict=True))))
        if factor==1.0:
            native_mismatch=float(np.mean(rendered!=labels))
            if node_map is not None and node_id is not None and color_index is not None:native_iou=_impl._best_iou(rendered,node_map==int(node_id),int(color_index))
    return {'lineage_errors':errors,'class_component_counts':counts,'native_mismatch':float(native_mismatch or 0.0),'native_iou':native_iou}

def stabilize_nested_seams(labels:np.ndarray,colors:np.ndarray,elements:list[str])->tuple[list[str],list[dict[str,Any]]]:
    labels=np.asarray(labels); colors=np.asarray(colors,dtype=np.uint8); node_map,nodes,_root,depth,parent=_impl._core._connected_region_graph(labels,len(colors))
    children=[[] for _ in nodes]
    for node_id,parent_id in enumerate(parent):
        if parent_id is not None:children[int(parent_id)].append(node_id)
    source_counts=[sum(1 for n in nodes if int(n['color_index'])==c) for c in range(len(colors))]; out=list(elements); base=_metrics(labels,colors,out,source_counts)
    if base is None or not any(base['lineage_errors']):return out,[]
    reports=[]
    for node_id,node in enumerate(nodes):
        parent_id=parent[node_id]
        if parent_id is None or int(depth[node_id])<2 or not children[node_id]:continue
        color=int(node['color_index']); pcolor=int(nodes[int(parent_id)]['color_index'])
        if color==pcolor:continue
        model=_shape(node_map==node_id)
        if model is None:continue
        prgb=colors[pcolor]; pfill=f'#{int(prgb[0]):02x}{int(prgb[1]):02x}{int(prgb[2]):02x}'; before=_metrics(labels,colors,out,source_counts,node_map,node_id,color)
        if before is None:continue
        before_errors=[int(v) for v in before['lineage_errors']]
        if not any(before_errors):break
        best=before; best_element=None; best_key=(sum(before_errors),max(before_errors),float(before['native_mismatch']))
        for width,non_scaling in ((0.5,True),(0.75,True),(1.0,True),(1.25,True),(1.0,False)):
            element=_seam(model,pfill,width,non_scaling); metrics=_metrics(labels,colors,out+[element],source_counts,node_map,node_id,color)
            if metrics is None:continue
            errors=[int(v) for v in metrics['lineage_errors']]
            if any(a>b for a,b in zip(errors,before_errors,strict=True)):continue
            if metrics['class_component_counts'][2]!=source_counts:continue
            ni=metrics.get('native_iou')
            if ni is None or float(ni)+1e-12<float(_impl._NATIVE_MIN_IOU):continue
            key=(sum(errors),max(errors),float(metrics['native_mismatch']))
            if key<best_key:best_key=key;best=metrics;best_element=element
        changed=best_element is not None
        if changed:out.append(best_element)
        reports.append({'node_id':node_id,'color_index':color,'parent_node_id':int(parent_id),'parent_color_index':pcolor,'model':model,'changed':changed,'before':before,'after':best})
        if changed and not any(best['lineage_errors']):break
    return out,reports

__all__=['stabilize_nested_seams']
