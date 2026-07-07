"""Next-generation Vektoryum.ai vectorization worker.

The class below is the new pipeline skeleton requested by the SRS:
Bilateral Filter -> optional Super-Resolution -> K-Means/SLIC planning ->
Canny/HED edge map -> Marching Squares/Potrace/Bezier fitting -> clean SVG.

It is intentionally separated from the legacy engine. Each stage has a clear
contract so we can replace placeholders with production-grade implementations
incrementally without changing the API Gateway.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from services.vectorizer_worker.contracts import RasterAnalysis, VectorizerArtifacts, VectorizerConfig


class PerfectVectorizer:
    """Measurement-first vectorizer skeleton for clean, production SVG output."""

    def __init__(self, config: VectorizerConfig | None = None) -> None:
        self.config = config or VectorizerConfig()

    def vectorize(self, input_path: Path, output_dir: Path) -> VectorizerArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        image = self._load_rgb(input_path)
        image = self._fit_processing_size(image)
        analysis = self.analyze(image)
        preprocessed = self.preprocess(image)
        edge_map = self.detect_edges(preprocessed)
        svg_text = self.fit_svg(preprocessed, edge_map=edge_map, analysis=analysis)

        preprocessed_path = output_dir / "preprocessed.png"
        edge_path = output_dir / "edges.png"
        svg_path = output_dir / "output.svg"
        Image.fromarray(preprocessed).save(preprocessed_path)
        Image.fromarray(edge_map).save(edge_path)
        svg_path.write_text(svg_text, encoding="utf-8")
        return VectorizerArtifacts(preprocessed=preprocessed_path, edge_map=edge_path, svg_path=svg_path)

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
        denoised = cv2.bilateralFilter(image, d=7, sigmaColor=45, sigmaSpace=45)
        return denoised

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # HED hook will be added here; Canny is the deterministic baseline.
        return cv2.Canny(gray, threshold1=60, threshold2=160)

    def fit_svg(self, image: np.ndarray, *, edge_map: np.ndarray, analysis: RasterAnalysis) -> str:
        """Create a clean SVG placeholder with mathematically sharp primitives.

        The next implementation step will replace the rectangle fallback with
        Marching Squares + cubic Bézier least-squares fitting. The contract is
        already strict: no embedded bitmap, explicit viewBox, no empty groups.
        """
        h, w = image.shape[:2]
        bg = self._dominant_hex(image)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
            f'  <rect width="{w}" height="{h}" rx="0" fill="{bg}"/>\n'
            f'</svg>\n'
        )

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

    def _dominant_hex(self, image: np.ndarray) -> str:
        pixels = image.reshape(-1, 3)
        mean = np.mean(pixels, axis=0).round().astype(int)
        return "#%02x%02x%02x" % tuple(int(v) for v in mean)

    def quality_contract(self, metrics: dict[str, Any]) -> bool:
        q = self.config.quality
        return (
            float(metrics.get("fidelity", 0.0)) >= q.min_fidelity
            and float(metrics.get("edge_f1", 0.0)) >= q.min_edge_f1
            and float(metrics.get("mean_delta_e", 999.0)) <= q.max_delta_e
            and float(metrics.get("banding_ratio", 999.0)) <= q.max_banding_ratio
        )
