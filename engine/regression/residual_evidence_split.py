"""Split AI-4 native and verified multi-scale evidence into explicit artifacts."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def _dump(path: Path, payload): path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")

def split(output: Path) -> None:
    report=json.loads((output/"output-quality-report.json").read_text(encoding="utf-8")); cases=report.get("cases") or []
    native={"schema":"vektoryum-ai4-native-residual-evidence-v1","engine_version":report.get("engine_version"),"repeat_count":report.get("repeat_count"),"fixture_provenance":report.get("fixture_provenance"),"cases":[]}
    scale={"schema":"vektoryum-ai4-multiscale-topology-evidence-v1","engine_version":report.get("engine_version"),"repeat_count":report.get("repeat_count"),"source_contract_policy":"normalized_analytic_geometry_v1","release_evidence":report.get("multi_scale_release_evidence","UNVERIFIED"),"cases":[]}
    for c in cases:
        native["cases"].append({"case_id":c.get("case_id"),"fixture_provenance":c.get("fixture_provenance"),"source_sha256":c.get("source_sha256"),"source_rgba_sha256":c.get("source_rgba_sha256"),"deterministic":c.get("deterministic"),"residual":c.get("residual"),"near_zero":c.get("near_zero")})
        scale["cases"].append({"case_id":c.get("case_id"),"fixture_provenance":c.get("fixture_provenance"),"multi_scale":c.get("multi_scale"),"near_zero_blockers":(c.get("near_zero") or {}).get("blocker_codes")})
    _dump(output/"native-residual-evidence.json",native); _dump(output/"multiscale-topology-evidence.json",scale)
    native_lines=["# Native residual evidence","",f"- Cases: **{len(native['cases'])}**",f"- Repeats: **{native.get('repeat_count')}**","", "| Case | Provenance | Deterministic | Near-zero |", "|---|---|---:|---:|"]
    for c in native["cases"]: native_lines.append(f"| `{c['case_id']}` | {c['fixture_provenance']} | {c['deterministic']} | {(c.get('near_zero') or {}).get('ready')} |")
    (output/"native-residual-evidence.md").write_text("\n".join(native_lines)+"\n",encoding="utf-8")
    scale_lines=["# Multi-scale + topology evidence","",f"- Source policy: `{scale['source_contract_policy']}`",f"- Release evidence: **{scale['release_evidence']}**","", "| Case | 64/128/256/512/1024 complete | Source contract |", "|---|---:|---:|"]
    for c in scale["cases"]:
        m=c.get("multi_scale") or {}; scale_lines.append(f"| `{c['case_id']}` | {m.get('all_scales_measured')} | {m.get('source_contract_verified')} |")
    (output/"multiscale-topology-evidence.md").write_text("\n".join(scale_lines)+"\n",encoding="utf-8")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); args=p.parse_args(); split(args.output)
if __name__=="__main__": main()
