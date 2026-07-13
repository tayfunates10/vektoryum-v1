"""HG-4 — Shadow graph cut-out face serializer (SHADOW).

Half-edge graph yüzlerinden gerçek SVG face path'leri üretir. Bu SVG YALNIZ
shadow artifact'tır: debug, regression karşılaştırması, kalite/komut/bellek
ölçümü için. Production preview/download/API'ye BAĞLANMAZ.

Kurallar (şartname):
- Her görünür yüz: bir outer cycle + sıfır/çok inner cycle (evenodd).
- Twin ortak sınır koordinatı canonical curve'den; yeniden hesap YOK.
- Background renkli eraser path YOK; delik = gerçek evenodd inner cycle.
- Curve'ler tekrar poligonize EDİLMEZ (fitted Bézier kullanılır).
- Transform YOK; source coordinate space; 0.01 serializer hassasiyeti.
- Determinist yüz sırası (z-order); metadata graph stats + shadow=true.
"""
from __future__ import annotations

from typing import Any

from app.canonical_curve import BezierSegment
from app.half_edge_graph import SharedBoundaryHalfEdgeGraph


def _fmt(v: float) -> str:
    """0.01 hassasiyet, gereksiz sıfırsız."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


def _oriented_segments(graph: SharedBoundaryHalfEdgeGraph, he_id: str) -> list[BezierSegment]:
    """Bir half-edge'in yönlü fitted segmentleri (reversed ise ters + segment ters)."""
    he = graph.half_edges[he_id]
    cur = graph.curves[he.curve_id]
    segs: list[BezierSegment] = list(cur.fitted_segments)
    if not segs:
        # fit yoksa polyline'dan düz segment zinciri (fallback)
        poly = cur.polyline
        segs = []
        for i in range(len(poly) - 1):
            p0 = (float(poly[i][0]), float(poly[i][1]))
            p1 = (float(poly[i + 1][0]), float(poly[i + 1][1]))
            c1 = (p0[0] + (p1[0] - p0[0]) / 3, p0[1] + (p1[1] - p0[1]) / 3)
            c2 = (p0[0] + 2 * (p1[0] - p0[0]) / 3, p0[1] + 2 * (p1[1] - p0[1]) / 3)
            segs.append(BezierSegment(p0, c1, c2, p1, is_line=True))
    if he.reversed:
        segs = [s.reversed() for s in reversed(segs)]
    return segs


def _cycle_path_d(graph: SharedBoundaryHalfEdgeGraph, cycle_id: str) -> tuple[str, int]:
    """Bir cycle'ın 'd' alt-yolu (M...Z) + komut sayısı. Twin koordinatı canonical."""
    cyc = graph.cycles[cycle_id]
    parts: list[str] = []
    cmds = 0
    started = False
    for he_id in cyc.half_edge_ids:
        segs = _oriented_segments(graph, he_id)
        for s in segs:
            if not started:
                parts.append(f"M{_fmt(s.p0[0])} {_fmt(s.p0[1])}")
                started = True
            if s.is_line:
                parts.append(f"L{_fmt(s.p1[0])} {_fmt(s.p1[1])}")
            else:
                parts.append(f"C{_fmt(s.c1[0])} {_fmt(s.c1[1])} "
                             f"{_fmt(s.c2[0])} {_fmt(s.c2[1])} "
                             f"{_fmt(s.p1[0])} {_fmt(s.p1[1])}")
            cmds += 1
    parts.append("Z")
    return "".join(parts), cmds


def _background_face(graph: SharedBoundaryHalfEdgeGraph) -> str | None:
    """Exterior'a komşu, en büyük alanlı görünür yüz = background."""
    ext = [f.face_id for f in graph.faces.values() if f.is_exterior]
    if not ext:
        return None
    ext_set = set(ext)
    bg = None
    best = -1.0
    for c in graph.curves.values():
        fa, fb = c.adjacent_face_ids
        for f in (fa, fb):
            if f and f not in ext_set:
                face = graph.faces.get(f)
                other = fb if f == fa else fa
                if face and face.visible and other in ext_set and face.area > best:
                    best, bg = face.area, f
    return bg


def serialize_graph_svg(graph: SharedBoundaryHalfEdgeGraph, width: int, height: int,
                        source_hash: str = "") -> tuple[str, dict[str, Any]]:
    """Graph'ı shadow SVG'ye serialize eder. Döner: (svg_str, metrics)."""
    bg_id = _background_face(graph)
    bg_face = graph.faces.get(bg_id) if bg_id else None

    # çizim sırası (z-order): background hariç, alan azalan + face_id determinist
    faces = [f for f in graph.faces.values()
             if f.visible and not f.is_exterior and f.face_id != bg_id]
    faces.sort(key=lambda f: (-f.area, f.face_id))

    total_cmds = 0
    visible_paths = 0
    lines: list[str] = []
    stats = graph.stats()

    # metadata
    lines.append("  <metadata>")
    lines.append(f'    <shadow-graph version="{graph.geometry_version}" '
                 f'shadow="true" source-hash="{source_hash}"/>')
    lines.append(f'    <graph-stats visible-faces="{stats["visible_faces"]}" '
                 f'curves="{stats["curves"]}" half-edges="{stats["half_edges"]}" '
                 f'junctions="{stats["junctions"]}" '
                 f'inner-cycles="{stats["inner_cycles"]}"/>')
    lines.append("  </metadata>")

    # background: basit tam-tuval rect (renkli eraser değil)
    lines.append('  <g id="shadow-background">')
    if bg_face is not None:
        fill = bg_face.fill_color or "#ffffff"
        lines.append(f'    <rect x="0" y="0" width="{width}" height="{height}" '
                     f'fill="{fill}" data-face-id="{bg_face.face_id}" '
                     f'data-region-id="{bg_face.source_region_id}"/>')
    lines.append("  </g>")

    # half-edge faces (outer + inner evenodd)
    lines.append('  <g id="shadow-half-edge-faces">')
    for f in faces:
        if f.outer_cycle_id is None:
            continue
        d_outer, c0 = _cycle_path_d(graph, f.outer_cycle_id)
        d = d_outer
        cmds = c0
        for inner in f.inner_cycle_ids:
            di, ci = _cycle_path_d(graph, inner)
            d += di
            cmds += ci
        fill = f.fill_color or "#000000"
        lines.append(f'    <path data-face-id="{f.face_id}" '
                     f'data-region-id="{f.source_region_id}" '
                     f'data-shadow-graph="true" fill="{fill}" '
                     f'fill-rule="evenodd" clip-rule="evenodd" d="{d}"/>')
        total_cmds += cmds
        visible_paths += 1
    lines.append("  </g>")

    body = "\n".join(lines)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
           f'height="{height}" viewBox="0 0 {width} {height}">\n{body}\n</svg>\n')

    metrics = {
        "visible_paths": visible_paths,
        "total_commands": total_cmds,
        "svg_bytes": len(svg.encode("utf-8")),
        "has_background_rect": bg_face is not None,
        "inner_cycles": stats["inner_cycles"],
        "has_transform": "transform=" in svg,
    }
    return svg, metrics
