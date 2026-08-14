from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, got {count}")
    return text.replace(old, new, 1)


def patch_deficit() -> None:
    path = Path("engine/app/alpha_candidate_paint_deficit.py")
    text = path.read_text(encoding="utf-8")

    marker = "\ndef _paint_deficit_labels(\n"
    helper = '''
def _coarsen_deficit_palette(
    labels: np.ndarray,
    palette: np.ndarray,
    target_limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce label fragmentation without changing deficit coverage.

    The original source-derived palette remains authoritative.  This helper is
    used only by the byte-pressure fallback: it chooses at most ``target_limit``
    existing palette entries as pixel-count-weighted medoids and remaps every
    positive deficit label to its nearest retained colour.  No deficit pixel is
    added or removed.  With at most eight colours exhaustive medoid selection is
    deterministic and tiny (max C(8, 4) == 70).
    """
    from itertools import combinations

    label_array = np.asarray(labels, dtype=np.int32)
    palette_array = np.asarray(palette, dtype=np.uint8)
    limit = max(1, int(target_limit))
    positive = label_array > 0
    if not bool(positive.any()) or len(palette_array) <= limit:
        return label_array.copy(), palette_array.copy()

    referenced = label_array[positive].astype(np.int64) - 1
    if bool(np.any(referenced < 0)) or bool(np.any(referenced >= len(palette_array))):
        raise ValueError("paint deficit labels reference palette out of range")

    weights = np.bincount(referenced, minlength=len(palette_array)).astype(np.int64)
    active = np.flatnonzero(weights > 0)
    if len(active) <= limit:
        return label_array.copy(), palette_array.copy()

    active_colors = palette_array[active].astype(np.int64)
    active_weights = weights[active].astype(np.float64)
    best_subset: tuple[int, ...] | None = None
    best_cost: float | None = None
    for subset in combinations(range(len(active)), limit):
        centers = active_colors[list(subset)]
        delta = active_colors[:, None, :] - centers[None, :, :]
        distance_sq = np.sum(delta * delta, axis=2)
        cost = float(np.sum(np.min(distance_sq, axis=1) * active_weights))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_subset = tuple(int(index) for index in subset)

    if best_subset is None:
        return label_array.copy(), palette_array.copy()

    selected_active = active[list(best_subset)]
    selected_colors = palette_array[selected_active].astype(np.int64)
    old_colors = palette_array.astype(np.int64)
    delta = old_colors[:, None, :] - selected_colors[None, :, :]
    nearest_selected = np.argmin(np.sum(delta * delta, axis=2), axis=1)

    remap = np.zeros(len(palette_array) + 1, dtype=np.int32)
    for old_index in active:
        remap[int(old_index) + 1] = int(nearest_selected[int(old_index)]) + 1
    result_labels = remap[label_array]
    if not np.array_equal(result_labels > 0, positive):
        raise AssertionError("paint deficit palette fallback changed coverage")
    return result_labels, palette_array[selected_active].copy()
'''
    if marker not in text:
        raise SystemExit("paint deficit helper marker missing")
    text = text.replace(marker, "\n" + helper + marker, 1)

    text = replace_once(
        text,
        '''def _paint_deficit_labels(
    source_rgba: np.ndarray,
    artwork_rgba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:''',
        '''def _paint_deficit_labels(
    source_rgba: np.ndarray,
    artwork_rgba: np.ndarray,
    *,
    palette_limit: int = _PALETTE_LIMIT,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:''',
        "labels signature",
    )
    text = replace_once(
        text,
        '''    labels = np.zeros(deficit.shape, dtype=np.int32)
    labels[deficit] = nearest[deficit] + 1
    alpha_residual = (source[:, :, 3] > 0) & (artwork[:, :, 3] < 255)''',
        '''    labels = np.zeros(deficit.shape, dtype=np.int32)
    labels[deficit] = nearest[deficit] + 1
    if int(palette_limit) < len(palette):
        labels, palette = _coarsen_deficit_palette(
            labels,
            palette,
            int(palette_limit),
        )
    alpha_residual = (source[:, :, 3] > 0) & (artwork[:, :, 3] < 255)''',
        "labels palette hook",
    )
    text = replace_once(
        text,
        '''    mask_encoding: str = "polygon",
    levels: int = _ALPHA_LEVELS,
) -> tuple[ET.Element, dict[str, Any]]:''',
        '''    mask_encoding: str = "polygon",
    levels: int = _ALPHA_LEVELS,
    palette_limit: int = _PALETTE_LIMIT,
) -> tuple[ET.Element, dict[str, Any]]:''',
        "builder signature",
    )
    text = replace_once(
        text,
        '''    labels, palette, deficit_stats = _paint_deficit_labels(
        source,
        artwork_rgba,
    )''',
        '''    labels, palette, deficit_stats = _paint_deficit_labels(
        source,
        artwork_rgba,
        palette_limit=palette_limit,
    )''',
        "builder labels call",
    )
    text = replace_once(
        text,
        '''        used_palette_count += 1

    layer.set("mask", f"url(#{mask_id})")
    return root, {''',
        '''        used_palette_count += 1

    support_serialized_bytes = len(ET.tostring(support, encoding="utf-8"))
    layer.set("mask", f"url(#{mask_id})")
    return root, {''',
        "support byte measurement",
    )
    text = replace_once(
        text,
        '''        "paint_deficit_palette_count": int(used_palette_count),
        "paint_deficit_support_rect_count": int(support_rect_count),''',
        '''        "paint_deficit_palette_count": int(used_palette_count),
        "paint_deficit_support_rect_count": int(support_rect_count),
        "paint_deficit_support_serialized_bytes": int(support_serialized_bytes),''',
        "support report metric",
    )
    path.write_text(text, encoding="utf-8")


def patch_painter() -> None:
    path = Path("engine/app/alpha_candidate_painter.py")
    text = path.read_text(encoding="utf-8")

    marker = "\ndef _painter_primary_error(\n"
    helper = '''
def _paint_deficit_palette_retry_eligible(
    attempts: list[dict[str, Any]],
) -> bool:
    """Retry support palette only after the complete legacy ladder is byte-bound.

    A quality, geometry, journal, or mixed rejection is terminal for this new
    family.  Requiring the exact existing ladder also guarantees that no
    currently accepted paint-deficit route is displaced.
    """
    expected_labels = [
        "paint-deficit-q24",
        "paint-deficit-cumulative",
        *[
            f"paint-deficit-cumulative-q{levels}"
            for levels in _PAINT_DEFICIT_CUMULATIVE_LEVELS
        ],
    ]
    return (
        [str(entry.get("encoding_label", "")) for entry in attempts]
        == expected_labels
        and all(
            entry.get("exact_or_quantized") == "paint_deficit"
            and entry.get("status") == "byte_rejected"
            for entry in attempts
        )
    )
'''
    if marker not in text:
        raise SystemExit("primary error marker missing")
    text = text.replace(marker, "\n" + helper + marker, 1)

    text = replace_once(
        text,
        '''        _validated("exact")
        or _validated("quantized")
        or _validated("paint_deficit")
        or _smallest_byte_rejected("exact")
        or _smallest_byte_rejected("quantized")
        or _smallest_byte_rejected("paint_deficit")''',
        '''        _validated("exact")
        or _validated("quantized")
        or _validated("paint_deficit")
        or _validated("paint_deficit_palette")
        or _smallest_byte_rejected("exact")
        or _smallest_byte_rejected("quantized")
        or _smallest_byte_rejected("paint_deficit")
        or _smallest_byte_rejected("paint_deficit_palette")''',
        "primary error family",
    )
    text = replace_once(
        text,
        '''        def _try(
            mask_encoding: str,
            label: str,
            levels: int = _ALPHA_LEVELS,
        ) -> list[Any] | None:''',
        '''        def _try(
            mask_encoding: str,
            label: str,
            levels: int = _ALPHA_LEVELS,
            *,
            palette_limit: int | None = None,
        ) -> list[Any] | None:''',
        "try signature",
    )
    text = replace_once(
        text,
        '''                "encoding_label": label,
                "encoding_family": "paint_deficit",
                "exact_or_quantized": "paint_deficit",''',
        '''                "encoding_label": label,
                "encoding_family": (
                    "paint_deficit"
                    if palette_limit is None
                    else "paint_deficit_palette"
                ),
                "exact_or_quantized": (
                    "paint_deficit"
                    if palette_limit is None
                    else "paint_deficit_palette"
                ),''',
        "ledger family",
    )
    text = replace_once(
        text,
        '''                "journal_reason_codes": [],
            }
            try:
                probe_root, probe_geometry = (
                    build_paint_deficit_reconstruction_tree(
                        original_root,
                        canvas,
                        grid_rgba,
                        txn,
                        mask_encoding=mask_encoding,
                        levels=levels,
                    )
                )''',
        '''                "journal_reason_codes": [],
            }
            if palette_limit is not None:
                entry["paint_deficit_palette_limit"] = int(palette_limit)
            try:
                if palette_limit is None:
                    probe_root, probe_geometry = (
                        build_paint_deficit_reconstruction_tree(
                            original_root,
                            canvas,
                            grid_rgba,
                            txn,
                            mask_encoding=mask_encoding,
                            levels=levels,
                        )
                    )
                else:
                    probe_root, probe_geometry = (
                        build_paint_deficit_reconstruction_tree(
                            original_root,
                            canvas,
                            grid_rgba,
                            txn,
                            mask_encoding=mask_encoding,
                            levels=levels,
                            palette_limit=int(palette_limit),
                        )
                    )''',
        "builder call",
    )
    text = replace_once(
        text,
        '''            for stat_key in (
                "paint_deficit_pixel_count",
                "source_component_count",
                "anchored_source_component_count",
                "detached_source_component_count",
            ):
                if stat_key in probe_geometry:
                    entry[stat_key] = int(probe_geometry[stat_key])
            if probe_size > byte_limit:''',
        '''            for stat_key in (
                "paint_deficit_pixel_count",
                "source_component_count",
                "anchored_source_component_count",
                "detached_source_component_count",
            ):
                if stat_key in probe_geometry:
                    entry[stat_key] = int(probe_geometry[stat_key])
            if palette_limit is not None:
                for stat_key in (
                    "paint_deficit_palette_count",
                    "paint_deficit_support_rect_count",
                    "paint_deficit_support_serialized_bytes",
                ):
                    if stat_key in probe_geometry:
                        entry[stat_key] = int(probe_geometry[stat_key])
            if probe_size > byte_limit:''',
        "fallback diagnostics",
    )
    old_phase = '''        winner = _try("polygon", "paint-deficit-q24")
        if winner is None:
            winner = _try("cumulative", "paint-deficit-cumulative")
        if winner is None:
            # Dikişi kapatan TEK kodlama kümülatif eşiktir; ayrık seviye
            # poligonları komşu seviye sınırında tasarımı gereği seam bırakır
            # (bkz. _cumulative_threshold_children). Kümülatif kodlamada seviye
            # başına bir <path> yazıldığından bayt maliyeti kafes çözünürlüğüyle
            # ölçeklenir: yoğun alfa rampalı kaynaklarda sabit q24 kafesi bayt
            # bütçesini aşıp journal kapısına HİÇ ulaşamıyordu, dolayısıyla seam
            # reddi onarılamaz kalıyordu. contour ailesinin q128/q64/q32
            # taramasının kümülatif karşılığı: kafesi kademeli seyreltip bütçeye
            # sığan ilk dikiş-kapatan adayı üret.
            #
            # Bu adım kesinlikle EKlemedir: yukarıdaki iki deneme aynen korunur,
            # yalnız ikisi de başarısızken devreye girer. Seyreltme kaliteyi
            # düşürürse aday, DEĞİŞMEMİŞ alfa IoU/MAE ve TransformJournal
            # kapılarında reddedilir — kalite düşürerek geçme yolu yoktur.
            for coarse_levels in _PAINT_DEFICIT_CUMULATIVE_LEVELS:
                winner = _try(
                    "cumulative",
                    f"paint-deficit-cumulative-q{coarse_levels}",
                    levels=coarse_levels,
                )
                if winner is not None:
                    break
        return winner
'''
    new_phase = '''        paint_deficit_attempt_start = len(attempts)
        winner = _try("polygon", "paint-deficit-q24")
        if winner is None:
            winner = _try("cumulative", "paint-deficit-cumulative")
        if winner is None:
            # Dikişi kapatan TEK kodlama kümülatif eşiktir; ayrık seviye
            # poligonları komşu seviye sınırında tasarımı gereği seam bırakır
            # (bkz. _cumulative_threshold_children). Kümülatif kodlamada seviye
            # başına bir <path> yazıldığından bayt maliyeti kafes çözünürlüğüyle
            # ölçeklenir: yoğun alfa rampalı kaynaklarda sabit q24 kafesi bayt
            # bütçesini aşıp journal kapısına HİÇ ulaşamıyordu, dolayısıyla seam
            # reddi onarılamaz kalıyordu. contour ailesinin q128/q64/q32
            # taramasının kümülatif karşılığı: kafesi kademeli seyreltip bütçeye
            # sığan ilk dikiş-kapatan adayı üret.
            #
            # Bu adım kesinlikle EKlemedir: yukarıdaki iki deneme aynen korunur,
            # yalnız ikisi de başarısızken devreye girer. Seyreltme kaliteyi
            # düşürürse aday, DEĞİŞMEMİŞ alfa IoU/MAE ve TransformJournal
            # kapılarında reddedilir — kalite düşürerek geçme yolu yoktur.
            for coarse_levels in _PAINT_DEFICIT_CUMULATIVE_LEVELS:
                winner = _try(
                    "cumulative",
                    f"paint-deficit-cumulative-q{coarse_levels}",
                    levels=coarse_levels,
                )
                if winner is not None:
                    break

        legacy_paint_attempts = attempts[paint_deficit_attempt_start:]
        if winner is None and _paint_deficit_palette_retry_eligible(
            legacy_paint_attempts
        ):
            # Task C: the measured public-15 ladder has an alpha-independent
            # serialization floor. Keep the smallest existing cumulative lattice
            # fixed and compact only the palette-labelled support geometry. This
            # family is byte-only: once a candidate fits the byte budget but fails
            # any unchanged evaluator/Journal gate, a coarser palette MUST NOT be
            # used to route around that quality rejection.
            fallback_levels = min(_PAINT_DEFICIT_CUMULATIVE_LEVELS)
            for palette_limit in (4, 2, 1):
                attempt_start = len(attempts)
                winner = _try(
                    "cumulative",
                    (
                        f"paint-deficit-cumulative-q{fallback_levels}"
                        f"-p{palette_limit}"
                    ),
                    levels=fallback_levels,
                    palette_limit=palette_limit,
                )
                if winner is not None:
                    break
                fallback_attempts = attempts[attempt_start:]
                if (
                    len(fallback_attempts) != 1
                    or fallback_attempts[0].get("status") != "byte_rejected"
                ):
                    break
        return winner
'''
    text = replace_once(text, old_phase, new_phase, "paint deficit phase")
    path.write_text(text, encoding="utf-8")


def patch_tests() -> None:
    path = Path("engine/test_alpha_painter_paint_deficit.py")
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''from app.alpha_candidate_paint_deficit import (
    _anchored_source_component_mask,
    _fixed_alpha_levels,
    _paint_deficit_labels,
    build_paint_deficit_reconstruction_tree,
)
''',
        '''from app.alpha_candidate_paint_deficit import (
    _anchored_source_component_mask,
    _fixed_alpha_levels,
    _paint_deficit_labels,
    build_paint_deficit_reconstruction_tree,
)
from app.alpha_candidate_painter import _paint_deficit_palette_retry_eligible
from app.alpha_svg_mask import _merged_rectangles_by_level
''',
        "test imports",
    )
    marker = '''    def test_builder_is_vector_only_deterministic_and_repairs_paint(self):
'''
    addition = '''    def test_palette_compaction_preserves_deficit_and_reduces_rectangles(self):
        source = np.zeros((8, 8, 4), dtype=np.uint8)
        source[:, :, 3] = 255
        artwork = np.full_like(source, 255)
        for y in range(8):
            for x in range(8):
                source[y, x, :3] = (
                    (32, 32, 32)
                    if (x + y) % 2 == 0
                    else (224, 32, 32)
                )

        labels8, palette8, stats8 = _paint_deficit_labels(source, artwork)
        labels1, palette1, stats1 = _paint_deficit_labels(
            source,
            artwork,
            palette_limit=1,
        )
        labels1_again, palette1_again, stats1_again = _paint_deficit_labels(
            source,
            artwork,
            palette_limit=1,
        )

        np.testing.assert_array_equal(labels8 > 0, labels1 > 0)
        self.assertEqual(stats8, stats1)
        self.assertEqual(stats1, stats1_again)
        self.assertGreaterEqual(len(palette8), 2)
        self.assertEqual(len(palette1), 1)
        np.testing.assert_array_equal(labels1, labels1_again)
        np.testing.assert_array_equal(palette1, palette1_again)

        rects8 = _merged_rectangles_by_level(labels8)
        rects1 = _merged_rectangles_by_level(labels1)
        count8 = sum(
            len(rectangles)
            for label, rectangles in rects8.items()
            if int(label) > 0
        )
        count1 = sum(
            len(rectangles)
            for label, rectangles in rects1.items()
            if int(label) > 0
        )
        self.assertGreater(count8, count1)
        self.assertEqual(count1, 1)

    def test_explicit_default_palette_limit_is_byte_identical(self):
        root, canvas = self._root()
        source = self._source()
        default_tree, default_report = build_paint_deficit_reconstruction_tree(
            root,
            canvas,
            source,
            "txn-default-palette",
            mask_encoding="cumulative",
            levels=8,
        )
        root2 = copy.deepcopy(root)
        canvas2 = list(root2)[0]
        explicit_tree, explicit_report = build_paint_deficit_reconstruction_tree(
            root2,
            canvas2,
            source,
            "txn-default-palette",
            mask_encoding="cumulative",
            levels=8,
            palette_limit=8,
        )
        self.assertEqual(ET.tostring(default_tree), ET.tostring(explicit_tree))
        self.assertEqual(default_report, explicit_report)

    def test_palette_retry_requires_complete_byte_rejected_legacy_ladder(self):
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
        self.assertTrue(_paint_deficit_palette_retry_eligible(attempts))
        self.assertFalse(_paint_deficit_palette_retry_eligible(attempts[:-1]))

        quality_rejected = [dict(entry) for entry in attempts]
        quality_rejected[-1]["status"] = "evaluator_rejected"
        self.assertFalse(_paint_deficit_palette_retry_eligible(quality_rejected))

        mixed_family = [dict(entry) for entry in attempts]
        mixed_family[-1]["exact_or_quantized"] = "quantized"
        self.assertFalse(_paint_deficit_palette_retry_eligible(mixed_family))

'''
    if marker not in text:
        raise SystemExit("test insertion marker missing")
    text = text.replace(marker, addition + marker, 1)
    path.write_text(text, encoding="utf-8")


patch_deficit()
patch_painter()
patch_tests()
