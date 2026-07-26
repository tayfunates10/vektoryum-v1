from __future__ import annotations

from pathlib import Path

REVIEWED_CASES_SHA256 = "5f151a6cb1a433b0cb0989a67bd7cc7940162f4b36d67903d6ccdd173f9e7d89"


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def update_rfv3b() -> None:
    path = Path(".github/workflows/real-world-fidelity-rfv3b-live.yml")
    text = path.read_text()
    old = '''              echo "RFV-3B source mode: live acquisition"
            else
              echo "RFV-3B live acquisition unavailable; restoring SHA-pinned immutable corpus"
'''
    new = f'''              if RFV_OUT="${{RFV_ROOT}}/out" \\
                REVIEWED_CASES_SHA256="{REVIEWED_CASES_SHA256}" \\
                python - <<'PYIDENTITY'
          import hashlib
          import json
          import os
          from pathlib import Path

          root = Path(os.environ["RFV_OUT"])
          expected_cases = os.environ["REVIEWED_CASES_SHA256"]
          bundle = root / "rfv2-public-corpus.tar.gz"
          checksums = json.loads((root / "bundle-checksums.json").read_text())
          assert checksums["schema"] == "vektoryum-rfv2-live-bundle-checksums-v1"
          assert checksums["qualified_case_count"] == 24
          assert bundle.stat().st_size == checksums["bundle_bytes"]
          assert hashlib.sha256(bundle.read_bytes()).hexdigest() == checksums["bundle_sha256"]
          actual_cases = checksums["cases_sha256"]
          print(json.dumps({{
              "actual_cases_sha256": actual_cases,
              "expected_reviewed_cases_sha256": expected_cases,
              "status": "reviewed_identity_match" if actual_cases == expected_cases else "reviewed_identity_mismatch",
          }}, sort_keys=True))
          raise SystemExit(0 if actual_cases == expected_cases else 42)
          PYIDENTITY
              then
                echo "RFV-3B source mode: live acquisition with reviewed identity"
              else
                echo "RFV-3B live acquisition identity mismatch; restoring SHA-pinned immutable corpus"
                acquired=0
                rm -f "${{RFV_ROOT}}/out/"{{rfv2-public-corpus.tar.gz,qualification-manifest.json,qualification-audit.json,bundle-checksums.json}}
              fi
            fi

            if [ "${{acquired}}" -eq 0 ]; then
              echo "RFV-3B live acquisition unavailable or identity-mismatched; restoring SHA-pinned immutable corpus"
'''
    path.write_text(replace_once(text, old, new, label="RFV-3B control flow"))


def update_rfv3e_scope() -> None:
    path = Path(".github/workflows/real-world-fidelity-rfv3e-viewbox-fix.yml")
    text = path.read_text()

    trigger_line = '      - ".github/workflows/real-world-fidelity-rfv3b-live.yml"\n'
    if trigger_line not in text:
        anchor = '      - ".github/workflows/real-world-fidelity-rfv3e-viewbox-journal-diagnostics.yml"\n'
        text = replace_once(text, anchor, anchor + trigger_line, label="RFV-3E trigger anchor")

    scope_line = '              .github/workflows/real-world-fidelity-rfv3b-live.yml|\\\n'
    if scope_line not in text:
        anchor = '              .github/workflows/real-world-fidelity-rfv3e-viewbox-journal-diagnostics.yml|\\\n'
        text = replace_once(text, anchor, anchor + scope_line, label="RFV-3E scope anchor")

    path.write_text(text)


if __name__ == "__main__":
    update_rfv3b()
    update_rfv3e_scope()
