"""Entry point that applies canonical-source scaling to the residual lab."""
from __future__ import annotations

from engine.regression import output_quality_residual_root_cause as root_cause
from engine.regression import output_quality_residual_suite as suite
from engine.regression.residual_multiscale_source import measure_multiscale_svg_from_base


if __name__ == "__main__":
    suite.measure_multiscale_svg = measure_multiscale_svg_from_base
    root_cause.main()
