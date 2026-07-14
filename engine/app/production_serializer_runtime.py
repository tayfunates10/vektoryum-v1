"""Runtime bridge for controlled canonical SVG publication.

The bridge reads the production feature flag and approved digest, delegates all
selection to ``production_svg_selector``, and atomically publishes the selected
SVG. Canonical output is never selected from environment state alone: a valid,
promoted HG-8 cutover report is still mandatory.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping

from .controlled_svg_cutover import ControlledSvgCutoverReport
from .production_svg_selector import (
    ProductionSvgSelection,
    atomic_publish_svg,
    select_production_svg,
)


@dataclass(frozen=True)
class ProductionSerializerRuntimeReport:
    selection: ProductionSvgSelection
    published: bool
    destination: str
    enabled: bool
    approved_sha256: str


def _enabled(env: Mapping[str, str]) -> bool:
    return env.get("VEKTORYUM_CANONICAL_SVG_ENABLED", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def publish_runtime_svg(
    *,
    legacy_svg: Path,
    destination: Path,
    cutover: ControlledSvgCutoverReport | None,
    environ: Mapping[str, str] | None = None,
) -> ProductionSerializerRuntimeReport:
    """Select and publish the production SVG, preserving legacy on any failure."""
    env = os.environ if environ is None else environ
    enabled = _enabled(env)
    approved = env.get("VEKTORYUM_CANONICAL_SVG_SHA256", "").strip().lower()

    legacy_path = Path(legacy_svg)
    legacy_bytes = legacy_path.read_bytes() if legacy_path.is_file() else b""

    gated_cutover = cutover
    if cutover is not None and cutover.output_sha256 != approved:
        gated_cutover = None

    selection = select_production_svg(
        legacy_svg_bytes=legacy_bytes,
        cutover=gated_cutover,
        enabled=enabled,
    )

    published = False
    if selection.svg_bytes:
        atomic_publish_svg(selection, Path(destination))
        published = True

    return ProductionSerializerRuntimeReport(
        selection=selection,
        published=published,
        destination=str(Path(destination)),
        enabled=enabled,
        approved_sha256=approved,
    )
