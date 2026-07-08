"""Next-generation Vektoryum.ai vectorization worker.

The class below implements the first production-oriented v2 pipeline slice:
Bilateral Filter -> optional resize -> K-Means color segmentation -> Canny edge
baseline -> contour extraction -> clean SVG path generation.

Future iterations can replace individual stages with HED, SLIC, OCR and a true
least-squares Bézier optimizer without changing the API Gateway or worker
contracts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from services.vectorizer_worker.contracts import RasterAnalysis, VectorizerArtifacts, VectorizerConfig


@dataclass(slots=True)
class _ColorRegion:
    color: tuple[int, int, int]
    mask: np.ndarray
    area: int


class PerfectVectorizer:
    """Measurement-first vectorizer for clean, production SVG output.

    The current implementation intentionally avoids bitmap embedding. It creates
    mathematically scalable SVG primitives from color regions and contours.
    """

    def __init__(self, config: VectorizerConfig | None = None) -> None:
        self.config = config or VectorizerConfig()

    def vectorize(self, input_path: Path, output_dir: Path) -> VectorizerArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        image = self._load_rgb(input_path)
        image = self._fit_processing_size(image)
        analysis = self.analyze(image)
        preprocessed = self.preprocess(image)
        edge_map = self.detect_edges(preprocessed)
        segment_map, regions = self.segment_colors(preprocessed, analysis)
        svg_text = self.fit_svg(preprocessed, edge_map=edge_map, analysis=analysis, regions=regions)

        preprocessed_path = output_dir / "preprocessed.png"
        edge_path = output_dir / "edges.png"
        segment_path = output_dir / "segments.png"
        svg_path = output_dir / "output.svg"
        Image.fromarray(preprocessed).save(preprocessed_path)
        Image.fromarray(edge_map).save(edge_path)
        Image.fromarray(segment_map).save(segment_path)
        svg_path.write_text(svg_text, encoding="utf-8")
        return VectorizerArtifacts(
            preprocessed=preprocessed_path,
            edge_map=edge_path,
            segment_map=segment_path,
            svg_path=svg_path,
        )

    def analyze(self, image: np.ndarray) -> RasterAnalysis:
        pixels = image.reshape(-1, 3)
        sample = pixels[:: max(1, len(pixels) // 12000)]
        quantized = np.unique((sample // 16).astype(np.uint8), axis=0)
        estimated_colors = int(len(quantized))
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        gradient_energy = float(np.mean(np.sqrt(np.sum(gx * gx + gy * gy, axis=2))))
        has_gradient = estimated_colors > 24 and gradient_energy < 38.0
        profile = "photo_poster" if estimated_colors > 96 else "logo" if estimated_colors > 8 else "icon"
        return RasterAnalysis(
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            estimated_colors=estimated_colors,
            has_gradient=has_gradient,
            profile=profile,
        )

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # Bilateral filtering: noise is reduced without destroying contour edges.
        return cv2.bilateralFilter(image, d=7, sigmaColor=45, sigmaSpace=45)

    def segment_colors(self, image: np.ndarray, analysis: RasterAnalysis) -> tuple[np.ndarray, list[_ColorRegion]]:
        """Posterize image into meaningful color regions with K-Means.

        The result is both a diagnostic segment map and a list of masks that are
        later converted into SVG paths. This is the v2 replacement for the old
        placeholder rectangle output.
        """
        h, w = image.shape[:2]
        estimated = max(1, int(analysis.estimated_colors))
        if analysis.profile == "photo_poster":
            k = min(18, max(8, int(round(np.sqrt(estimated)))))
        else:
            k = min(14, max(2, int(round(np.sqrt(estimated) * 1.35))))
        if estimated <= 1:
            color = tuple(int(v) for v in np.mean(image.reshape(-1, 3), axis=0).round())
            mask = np.full((h, w), 255, dtype=np.uint8)
            segment = np.zeros_like(image)
            segment[:, :] = color
            return segment, [_ColorRegion(color=color, mask=mask, area=h * w)]

        pixels = image.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 0.8)
        _compactness, labels, centers = cv2.kmeans(pixels, k, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
        centers_u8 = np.clip(centers, 0, 255).astype(np.uint8)
        label_img = labels.reshape((h, w))
        segment = centers_u8[label_img].reshape((h, w, 3))

        regions: list[_ColorRegion] = []
        kernel = np.ones((3, 3), np.uint8)
        for idx, center in enumerate(centers_u8):
            mask = np.where(label_img == idx, 255, 0).astype(np.uint8)
            area = int(cv2.countNonZero(mask))
            if area < max(6, int(0.00008 * h * w)):
                continue
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            area = int(cv2.countNonZero(mask))
            if area:
                regions.append(_ColorRegion(color=tuple(int(v) for v in center), mask=mask, area=area))
        regions.sort(key=lambda r: r.area, reverse=True)
        return segment, regions

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # HED hook will be added here; Canny is the deterministic baseline.
        return cv2.Canny(gray, threshold1=60, threshold2=160)

    def fit_svg(
        self,
        image: np.ndarray,
        *,
        edge_map: np.ndarray,
        analysis: RasterAnalysis,
        regions: list[_ColorRegion] | None = None,
    ) -> str:
        """Convert segmented color regions into clean SVG paths.

        Paths are emitted from largest to smallest region. The largest region is
        represented as the background rectangle; subsequent regions are traced as
        closed paths. No bitmap, empty group or metadata is emitted.
        """
        h, w = image.shape[:2]
        regions = regions or self.segment_colors(image, analysis)[1]
        background = regions[0].color if regions else self._dominant_rgb(image)
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
            f'  <rect width="{w}" height="{h}" fill="{self._hex(background)}"/>',
        ]
        min_area = max(8.0, 0.00012 * w * h)
        for region in regions[1:]:
            for contour in self._contours(region.mask):
                area = float(cv2.contourArea(contour))
                if area < min_area:
                    continue
                path = self._contour_to_path(contour, smooth=analysis.profile != "icon")
                if path:
                    lines.append(f'  <path d="{path}" fill="{self._hex(region.color)}"/>')
        lines.append("</svg>")
        return "\n".join(lines) + "\n"

    def _contours(self, mask: np.ndarray) -> list[np.ndarray]:
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours

    def _contour_to_path(self, contour: np.ndarray, *, smooth: bool) -> str | None:
        peri = cv2.arcLength(contour, True)
        if peri <= 0:
            return None
        epsilon = max(0.65, 0.0028 * peri)
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(float)
        if len(approx) < 3:
            return None
        if not smooth or len(approx) < 7:
            coords = [f"M {approx[0,0]:.2f} {approx[0,1]:.2f}"]
            coords += [f"L {x:.2f} {y:.2f}" for x, y in approx[1:]]
            coords.append("Z")
            return " ".join(coords)
        return self._catmull_rom_closed_path(approx)

    def _catmull_rom_closed_path(self, pts: np.ndarray) -> str:
        commands = [f"M {pts[0,0]:.2f} {pts[0,1]:.2f}"]
        n = len(pts)
        for i in range(n):
            p0 = pts[(i - 1) % n]
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            p3 = pts[(i + 2) % n]
            c1 = p1 + (p2 - p0) / 6.0
            c2 = p2 - (p3 - p1) / 6.0
            commands.append(
                f"C {c1[0]:.2f} {c1[1]:.2f} {c2[0]:.2f} {c2[1]:.2f} {p2[0]:.2f} {p2[1]:.2f}"
            )
        commands.append("Z")
        return " ".join(commands)

    def _load_rgb(self, path: Path) -> np.ndarray:
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"))

    def _fit_processing_size(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        max_side = max(h, w)
        if max_side <= self.config.max_processing_side:
            return image
        scale = self.config.max_processing_side / float(max_side)
        return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)

    def _dominant_rgb(self, image: np.ndarray) -> tuple[int, int, int]:
        pixels = image.reshape(-1, 3)
        mean = np.mean(pixels, axis=0).round().astype(int)
        return tuple(int(v) for v in mean)

    def _hex(self, color: tuple[int, int, int]) -> str:
        return "#%02x%02x%02x" % tuple(int(np.clip(v, 0, 255)) for v in color)

    def quality_contract(self, metrics: dict[str, Any]) -> bool:
        q = self.config.quality
        return (
            float(metrics.get("fidelity", 0.0)) >= q.min_fidelity
            and float(metrics.get("edge_f1", 0.0)) >= q.min_edge_f1
            and float(metrics.get("mean_delta_e", 999.0)) <= q.max_delta_e
            and float(metrics.get("banding_ratio", 999.0)) <= q.max_banding_ratio
        )
