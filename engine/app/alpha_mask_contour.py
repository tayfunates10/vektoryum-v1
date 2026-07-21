"""Deterministic cell-boundary contour tracing for alpha-mask geometry.

Traces the exact union boundary of a boolean raster cell set: shared internal
edges cancel, the remaining directed unit edges are stitched into closed
axis-aligned loops and collinear points are removed. Identical input produces
identical loops: loops start at their lexicographically smallest ``(y, x)``
corner, are ordered by that corner, and ambiguous checkerboard vertices always
resolve with the same right-turn rule so loops never cross.
"""
from __future__ import annotations

import numpy as np

# Sağ-el kuralı: iç bölge yürüyüş yönünün sağında kalır; checkerboard tepe
# noktalarında her zaman en keskin sağ dönüş seçilir (döngüler kesişmez).
_RIGHT_TURN = {(1, 0): (0, 1), (0, 1): (-1, 0), (-1, 0): (0, -1), (0, -1): (1, 0)}


def _directed_edges(mask: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """All boundary unit edges with the inside on the right; internal edges cancel."""
    cells = np.asarray(mask, dtype=bool)
    height, width = cells.shape
    padded = np.zeros((height + 2, width + 2), dtype=bool)
    padded[1:-1, 1:-1] = cells

    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def add(x0: int, y0: int, x1: int, y1: int) -> None:
        edges.setdefault((x0, y0), []).append((x1, y1))

    for y, x in zip(*np.nonzero(cells & ~padded[:-2, 1:-1])):  # üst sınır
        add(int(x), int(y), int(x) + 1, int(y))
    for y, x in zip(*np.nonzero(cells & ~padded[2:, 1:-1])):  # alt sınır
        add(int(x) + 1, int(y) + 1, int(x), int(y) + 1)
    for y, x in zip(*np.nonzero(cells & ~padded[1:-1, :-2])):  # sol sınır
        add(int(x), int(y) + 1, int(x), int(y))
    for y, x in zip(*np.nonzero(cells & ~padded[1:-1, 2:])):  # sağ sınır
        add(int(x) + 1, int(y), int(x) + 1, int(y) + 1)

    for candidates in edges.values():
        candidates.sort()
    return edges


def trace_cell_contours(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Closed corner loops covering exactly the ``True`` cells of ``mask``."""
    edges = _directed_edges(mask)
    loops: list[list[tuple[int, int]]] = []

    starts = sorted(edges, key=lambda vertex: (vertex[1], vertex[0]))
    for start in starts:
        while edges.get(start):
            current = start
            outgoing = edges[current].pop(0)
            if not edges[current]:
                del edges[current]
            direction = (outgoing[0] - current[0], outgoing[1] - current[1])
            corners = [current]
            while outgoing != corners[0]:
                candidates = edges.get(outgoing)
                if not candidates:
                    raise RuntimeError("source_alpha_contour_open_boundary")
                if len(candidates) == 1:
                    following = candidates.pop(0)
                else:
                    preferred = (
                        outgoing[0] + _RIGHT_TURN[direction][0],
                        outgoing[1] + _RIGHT_TURN[direction][1],
                    )
                    following = (
                        candidates.pop(candidates.index(preferred))
                        if preferred in candidates
                        else candidates.pop(0)
                    )
                if not candidates:
                    del edges[outgoing]
                next_direction = (
                    following[0] - outgoing[0],
                    following[1] - outgoing[1],
                )
                if next_direction != direction:
                    corners.append(outgoing)
                    direction = next_direction
                outgoing = following

            pivot = min(
                range(len(corners)), key=lambda i: (corners[i][1], corners[i][0])
            )
            loops.append(corners[pivot:] + corners[:pivot])

    loops.sort(key=lambda loop: (loop[0][1], loop[0][0]))
    return loops


def loop_signed_area(corners: list[tuple[int, int]]) -> float:
    """Shoelace area; sign distinguishes outer loops from enclosed hole loops."""
    total = 0
    for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
        total += x0 * y1 - x1 * y0
    return total / 2.0
