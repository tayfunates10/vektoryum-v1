from __future__ import annotations

import importlib.util
import re
from pathlib import Path

BASE_SCRIPT = Path("engine/regression/faz3d4_integrate_paint_deficit.py")


def _load_base():
    spec = importlib.util.spec_from_file_location("faz3d4_integrator_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("paint-deficit base integrator could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _materialize_robust(base, workflow: str) -> None:
    module_path = Path("engine/app/alpha_candidate_paint_deficit.py")
    test_path = Path("engine/test_alpha_painter_paint_deficit.py")
    module_path.write_text(
        base._extract_cat(workflow, str(module_path)), encoding="utf-8"
    )

    test_text = base._extract_cat(workflow, str(test_path))
    pattern = re.compile(
        r'(?P<indent>^[ \t]*)tree2, report2 = '
        r'build_paint_deficit_reconstruction_tree\(\n'
        r'(?P=indent)[ \t]+copy\.deepcopy\(root\), '
        r'list\(copy\.deepcopy\(root\)\)\[0\], source, "txn-fixed"\n'
        r'(?P=indent)\)',
        re.MULTILINE,
    )

    def replacement(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}root2 = copy.deepcopy(root)\n"
            f"{indent}canvas2 = list(root2)[0]\n"
            f"{indent}tree2, report2 = build_paint_deficit_reconstruction_tree(\n"
            f"{indent}    root2, canvas2, source, \"txn-fixed\"\n"
            f"{indent})"
        )

    test_text, count = pattern.subn(replacement, test_text, count=1)
    if count != 1:
        raise RuntimeError(f"determinism test repair count invalid: {count}")
    test_path.write_text(test_text, encoding="utf-8")


def _anchor_candidate_to_existing_artwork() -> None:
    module_path = Path("engine/app/alpha_candidate_paint_deficit.py")
    text = module_path.read_text(encoding="utf-8")
    old = "    deficit = source_foreground & artwork_missing\n"
    new = (
        "    artwork_occupied = artwork[:, :, 3] > 0\n"
        "    deficit = source_foreground & artwork_missing & artwork_occupied\n"
    )
    if old not in text:
        raise RuntimeError("paint-deficit occupancy anchor missing")
    module_path.write_text(text.replace(old, new, 1), encoding="utf-8")

    test_path = Path("engine/test_alpha_painter_paint_deficit.py")
    test_text = test_path.read_text(encoding="utf-8")
    anchor = "                  return root, canvas\n"
    addition = '''                  ET.SubElement(
                      root,
                      qname("rect"),
                      {"x": "2", "y": "0", "width": "1", "height": "4", "fill": "white"},
                  )
                  return root, canvas
'''
    if anchor not in test_text:
        raise RuntimeError("paint-deficit test artwork anchor missing")
    test_path.write_text(test_text.replace(anchor, addition, 1), encoding="utf-8")


def main() -> None:
    base = _load_base()
    workflow = base._source_workflow()
    _materialize_robust(base, workflow)
    _anchor_candidate_to_existing_artwork()
    base._integrate_painter()


if __name__ == "__main__":
    main()
