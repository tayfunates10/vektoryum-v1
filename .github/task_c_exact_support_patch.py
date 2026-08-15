from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, got {count}")
    return text.replace(old, new, 1)


def patch_deficit() -> None:
    path = Path("engine/app/alpha_candidate_paint_deficit.py")
    text = path.read_text(encoding="utf-8")

    start = text.index("\ndef _coarsen_deficit_palette(\n")
    end = text.index("\ndef _paint_deficit_labels(\n", start)
    text = text[:start] + "\n" + text[end:]

    text = replace_once(
        text,
        '''def _paint_deficit_labels(
    source_rgba: np.ndarray,
    artwork_rgba: np.ndarray,
    *,
    palette_limit: int = _PALETTE_LIMIT,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:''',
        '''def _paint_deficit_labels(
    source_rgba: np.ndarray,
    artwork_rgba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:''',
        "restore label signature",
    )
    text = replace_once(
        text,
        '''    labels = np.zeros(deficit.shape, dtype=np.int32)
    labels[deficit] = nearest[deficit] + 1
    if int(palette_limit) < len(palette):
        labels, palette = _coarsen_deficit_palette(
            labels,
            palette,
            int(palette_limit),
        )
    alpha_residual = (source[:, :, 3] > 0) & (artwork[:, :, 3] < 255)''',
        '''    labels = np.zeros(deficit.shape, dtype=np.int32)
    labels[deficit] = nearest[deficit] + 1
    alpha_residual = (source[:, :, 3] > 0) & (artwork[:, :, 3] < 255)''',
        "remove palette mutation",
    )

    helper_marker = "\ndef build_paint_deficit_reconstruction_tree(\n"
    helper = r'''
def _build_paint_deficit_support(
    labels: np.ndarray,
    palette: np.ndarray,
    transform: str,
    transaction_id: str,
    *,
    support_encoding: str = "per_label",
) -> tuple[ET.Element, dict[str, Any]]:
    """Serialize deficit support without changing its visible palette assignment.

    ``per_label`` is the byte-identical legacy representation: one colour group
    per source-palette label. ``base_delta`` is the Task C fallback. It paints the
    entire binary deficit once with one *existing* palette colour, then overwrites
    every non-base label with that label's original colour. The final colour of
    every deficit cell is therefore identical to ``per_label``; only repeated
    rectangle geometry is factored out.

    At most eight base labels are possible. Build all exact-colour candidates and
    deterministically choose the smallest serialized support (then fewer rects,
    then the lower base label). If factoring is not actually smaller, fail closed.
    """
    from app.alpha_svg_mask import _merged_rectangles_by_level  # noqa: PLC0415

    qname = lambda name: f"{{{_SVG_NS}}}{name}"
    label_array = np.asarray(labels, dtype=np.int32)
    palette_array = np.asarray(palette, dtype=np.uint8)
    rectangles = _merged_rectangles_by_level(label_array)
    active_labels = [
        int(label)
        for label in sorted(rectangles)
        if 0 < int(label) <= len(palette_array) and rectangles[label]
    ]
    if not active_labels:
        raise RuntimeError("source_alpha_candidate_painter_paint_deficit_empty_support")

    def new_support() -> ET.Element:
        node = ET.Element(
            qname("g"),
            {
                "transform": transform,
                "data-vektoryum-paint-deficit": "source-palette-v1",
            },
        )
        tag_transform_node(node, ROLE_MASK_GEOMETRY, transaction_id)
        return node

    def append_colour(
        support: ET.Element,
        label: int,
        rects: list[tuple[int, int, int, int]],
    ) -> int:
        if not rects:
            return 0
        color = palette_array[label - 1]
        group = ET.SubElement(
            support,
            qname("g"),
            {
                "fill": (
                    f"rgb({int(color[0])},{int(color[1])},"
                    f"{int(color[2])})"
                )
            },
        )
        for x, y, width, height in rects:
            ET.SubElement(
                group,
                qname("rect"),
                {
                    "x": str(x),
                    "y": str(y),
                    "width": str(width),
                    "height": str(height),
                },
            )
        return len(rects)

    legacy = new_support()
    legacy_rect_count = 0
    for label in active_labels:
        legacy_rect_count += append_colour(legacy, label, rectangles[label])
    legacy_bytes = len(ET.tostring(legacy, encoding="utf-8"))

    if support_encoding == "per_label":
        return legacy, {
            "paint_deficit_palette_count": int(len(active_labels)),
            "paint_deficit_support_rect_count": int(legacy_rect_count),
            "paint_deficit_support_serialized_bytes": int(legacy_bytes),
            "paint_deficit_support_encoding": "per_label",
            "paint_deficit_support_base_label": 0,
            "paint_deficit_support_legacy_serialized_bytes": int(legacy_bytes),
            "paint_deficit_support_saved_bytes": 0,
        }
    if support_encoding != "base_delta":
        raise RuntimeError(
            "source_alpha_candidate_painter_paint_deficit_unknown_support_encoding:"
            f"{support_encoding}"
        )

    binary = (label_array > 0).astype(np.int32)
    coverage = _merged_rectangles_by_level(binary).get(1, [])
    if not coverage:
        raise RuntimeError("source_alpha_candidate_painter_paint_deficit_empty_support")

    best: tuple[int, int, int, ET.Element] | None = None
    for base_label in active_labels:
        candidate = new_support()
        rect_count = append_colour(candidate, base_label, coverage)
        for label in active_labels:
            if label == base_label:
                continue
            rect_count += append_colour(candidate, label, rectangles[label])
        serialized_bytes = len(ET.tostring(candidate, encoding="utf-8"))
        key = (int(serialized_bytes), int(rect_count), int(base_label))
        if best is None or key < best[:3]:
            best = (key[0], key[1], key[2], candidate)

    if best is None or int(best[0]) >= int(legacy_bytes):
        raise RuntimeError(
            "source_alpha_candidate_painter_paint_deficit_base_delta_not_smaller"
        )
    compact_bytes, compact_rect_count, base_label, support = best
    return support, {
        "paint_deficit_palette_count": int(len(active_labels)),
        "paint_deficit_support_rect_count": int(compact_rect_count),
        "paint_deficit_support_serialized_bytes": int(compact_bytes),
        "paint_deficit_support_encoding": "base_delta",
        "paint_deficit_support_base_label": int(base_label),
        "paint_deficit_support_legacy_serialized_bytes": int(legacy_bytes),
        "paint_deficit_support_saved_bytes": int(legacy_bytes - compact_bytes),
    }
'''
    if helper_marker not in text:
        raise SystemExit("builder marker missing")
    text = text.replace(helper_marker, "\n" + helper + helper_marker, 1)

    text = replace_once(
        text,
        '''    mask_encoding: str = "polygon",
    levels: int = _ALPHA_LEVELS,
    palette_limit: int = _PALETTE_LIMIT,
) -> tuple[ET.Element, dict[str, Any]]:''',
        '''    mask_encoding: str = "polygon",
    levels: int = _ALPHA_LEVELS,
    support_encoding: str = "per_label",
) -> tuple[ET.Element, dict[str, Any]]:''',
        "builder support signature",
    )
    text = replace_once(
        text,
        '''    labels, palette, deficit_stats = _paint_deficit_labels(
        source,
        artwork_rgba,
        palette_limit=palette_limit,
    )''',
        '''    labels, palette, deficit_stats = _paint_deficit_labels(
        source,
        artwork_rgba,
    )''',
        "restore labels call",
    )

    support_start = text.index("    support = ET.SubElement(\n        layer,\n        qname(\"g\"),")
    support_end_marker = "\n    layer.set(\"mask\", f\"url(#{mask_id})\")"
    support_end = text.index(support_end_marker, support_start)
    replacement = '''    support, support_stats = _build_paint_deficit_support(
        labels,
        palette,
        (
            f"translate({view_x:.12g} {view_y:.12g}) "
            f"scale({view_width / grid_width:.12g} "
            f"{view_height / grid_height:.12g})"
        ),
        transaction_id,
        support_encoding=support_encoding,
    )
    layer.append(support)
'''
    text = text[:support_start] + replacement + text[support_end:]

    text = replace_once(
        text,
        '''        "paint_deficit_palette_count": int(used_palette_count),
        "paint_deficit_support_rect_count": int(support_rect_count),
        "paint_deficit_support_serialized_bytes": int(support_serialized_bytes),''',
        '''        **support_stats,''',
        "support report stats",
    )
    path.write_text(text, encoding="utf-8")


def patch_painter() -> None:
    path = Path("engine/app/alpha_candidate_painter.py")
    text = path.read_text(encoding="utf-8")
    text = text.replace("_paint_deficit_palette_retry_eligible", "_paint_deficit_support_retry_eligible")
    text = text.replace('"paint_deficit_palette"', '"paint_deficit_support_compact"')
    text = text.replace("palette_limit: int | None = None", "support_encoding: str | None = None")
    text = text.replace("if palette_limit is None", "if support_encoding is None")
    text = text.replace('entry["paint_deficit_palette_limit"] = int(palette_limit)', 'entry["paint_deficit_support_encoding"] = str(support_encoding)')
    text = text.replace("if palette_limit is not None:", "if support_encoding is not None:")
    text = text.replace("palette_limit=int(palette_limit),", "support_encoding=str(support_encoding),")

    text = replace_once(
        text,
        '''                for stat_key in (
                    "paint_deficit_palette_count",
                    "paint_deficit_support_rect_count",
                    "paint_deficit_support_serialized_bytes",
                ):''',
        '''                for stat_key in (
                    "paint_deficit_palette_count",
                    "paint_deficit_support_rect_count",
                    "paint_deficit_support_serialized_bytes",
                    "paint_deficit_support_base_label",
                    "paint_deficit_support_legacy_serialized_bytes",
                    "paint_deficit_support_saved_bytes",
                ):''',
        "compact support telemetry",
    )

    old_phase_start = text.index("        legacy_paint_attempts = attempts[paint_deficit_attempt_start:]\n")
    old_phase_end = text.index("        return winner\n", old_phase_start)
    old_phase_end += len("        return winner\n")
    new_phase = '''        legacy_paint_attempts = attempts[paint_deficit_attempt_start:]
        if winner is None and _paint_deficit_support_retry_eligible(
            legacy_paint_attempts
        ):
            # Task C: public-15's q-ladder proves an alpha-independent byte floor.
            # Keep the smallest existing q8 cumulative mask and every original
            # source-palette colour unchanged. Factor only support geometry as
            # one binary base fill plus exact-colour correction rectangles.
            fallback_levels = min(_PAINT_DEFICIT_CUMULATIVE_LEVELS)
            winner = _try(
                "cumulative",
                f"paint-deficit-cumulative-q{fallback_levels}-base-delta",
                levels=fallback_levels,
                support_encoding="base_delta",
            )
        return winner
'''
    text = text[:old_phase_start] + new_phase + text[old_phase_end:]
    path.write_text(text, encoding="utf-8")


def patch_tests() -> None:
    path = Path("engine/test_alpha_painter_paint_deficit.py")
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        '''    _paint_deficit_labels,
    build_paint_deficit_reconstruction_tree,
)\nfrom app.alpha_candidate_painter import _paint_deficit_palette_retry_eligible\nfrom app.alpha_svg_mask import _merged_rectangles_by_level\n'''.replace('\\n', '\n'),
        '''    _build_paint_deficit_support,
    _paint_deficit_labels,
    build_paint_deficit_reconstruction_tree,
)\nfrom app.alpha_candidate_painter import _paint_deficit_support_retry_eligible\n'''.replace('\\n', '\n'),
    )

    start = text.index("    def test_palette_compaction_preserves_deficit_and_reduces_rectangles(self):\n")
    end = text.index("    def test_builder_is_vector_only_deterministic_and_repairs_paint(self):\n", start)
    replacement = '''    def test_base_delta_support_is_render_exact_and_smaller(self):
        labels = np.fromfunction(
            lambda y, x: ((x + y) % 2) + 1,
            (8, 8),
            dtype=int,
        ).astype(np.int32)
        palette = np.asarray([[32, 32, 32], [224, 32, 32]], dtype=np.uint8)
        transform = "translate(0 0) scale(1 1)"
        legacy, legacy_stats = _build_paint_deficit_support(
            labels,
            palette,
            transform,
            "txn-support-exact",
            support_encoding="per_label",
        )
        compact, compact_stats = _build_paint_deficit_support(
            labels,
            palette,
            transform,
            "txn-support-exact",
            support_encoding="base_delta",
        )

        self.assertEqual(compact_stats["paint_deficit_palette_count"], 2)
        self.assertEqual(compact_stats["paint_deficit_support_encoding"], "base_delta")
        self.assertGreater(compact_stats["paint_deficit_support_saved_bytes"], 0)
        self.assertLess(
            compact_stats["paint_deficit_support_serialized_bytes"],
            legacy_stats["paint_deficit_support_serialized_bytes"],
        )
        self.assertLess(
            compact_stats["paint_deficit_support_rect_count"],
            legacy_stats["paint_deficit_support_rect_count"],
        )

        legacy_root = ET.Element(
            qname("svg"),
            {"viewBox": "0 0 8 8", "width": "8", "height": "8"},
        )
        legacy_root.append(copy.deepcopy(legacy))
        compact_root = ET.Element(
            qname("svg"),
            {"viewBox": "0 0 8 8", "width": "8", "height": "8"},
        )
        compact_root.append(copy.deepcopy(compact))
        legacy_rgba = _render_root(legacy_root, 8, 8)
        compact_rgba = _render_root(compact_root, 8, 8)
        self.assertIsNotNone(legacy_rgba)
        self.assertIsNotNone(compact_rgba)
        np.testing.assert_array_equal(legacy_rgba, compact_rgba)

    def test_explicit_default_support_encoding_is_byte_identical(self):
        root, canvas = self._root()
        source = self._source()
        default_tree, default_report = build_paint_deficit_reconstruction_tree(
            root,
            canvas,
            source,
            "txn-default-support",
            mask_encoding="cumulative",
            levels=8,
        )
        root2 = copy.deepcopy(root)
        canvas2 = list(root2)[0]
        explicit_tree, explicit_report = build_paint_deficit_reconstruction_tree(
            root2,
            canvas2,
            source,
            "txn-default-support",
            mask_encoding="cumulative",
            levels=8,
            support_encoding="per_label",
        )
        self.assertEqual(ET.tostring(default_tree), ET.tostring(explicit_tree))
        self.assertEqual(default_report, explicit_report)

    def test_support_retry_requires_complete_byte_rejected_legacy_ladder(self):
        labels = [
            "paint-deficit-q24",
            "paint-deficit-cumulative",
            "paint-deficit-cumulative-q20",
            "paint-deficit-cumulative-q16",
            "paint-deficit-cumulative-q12",
            "paint-deficit-cumulative-q8",
        ]
        attempts = [
            {
                "encoding_label": label,
                "exact_or_quantized": "paint_deficit",
                "status": "byte_rejected",
            }
            for label in labels
        ]
        self.assertTrue(_paint_deficit_support_retry_eligible(attempts))
        self.assertFalse(_paint_deficit_support_retry_eligible(attempts[:-1]))

        quality_rejected = [dict(entry) for entry in attempts]
        quality_rejected[-1]["status"] = "evaluator_rejected"
        self.assertFalse(_paint_deficit_support_retry_eligible(quality_rejected))

        mixed_family = [dict(entry) for entry in attempts]
        mixed_family[-1]["exact_or_quantized"] = "quantized"
        self.assertFalse(_paint_deficit_support_retry_eligible(mixed_family))

'''
    text = text[:start] + replacement + text[end:]
    path.write_text(text, encoding="utf-8")


patch_deficit()
patch_painter()
patch_tests()
