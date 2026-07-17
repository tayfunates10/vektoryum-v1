"""RFV-3E remediation plan doğrulayıcı.

Plan, en güncel kanonik RFV-3 generation'ına SHA ile bağlıdır ve yalnız gerçek
ölçüm ihlallerini kümeler. Doğrulayıcı fail-closed çalışır: eksik dosya, digest
uyuşmazlığı, eksik/yinelenen vaka, geçen vakanın remediation listesine girmesi,
eşik kayması veya NO-GO kararının kaybı hata üretir. Hiçbir metrik değeri
üretilmez veya değiştirilmez.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = ROOT / "docs" / "real_world_fidelity" / "evidence" / "rfv3e_remediation_plan.json"
DEFAULT_RESULTS = ROOT / "docs" / "real_world_fidelity" / "evidence" / "rfv3_pipeline_results.json"
DEFAULT_DECISION = ROOT / "docs" / "real_world_fidelity" / "evidence" / "rfv3_quality_decision.json"

PLAN_SCHEMA = "vektoryum-rfv3e-remediation-plan-v1"
REQUIRED_THRESHOLDS = {"fidelity": 99.0, "ssim": 0.98, "edge_f1": 0.98, "alpha_iou": 0.98}
ALLOWED_CONFIDENCE = ("proven", "strong", "tentative")
REQUIRED_CLUSTER_FIELDS = (
    "cluster_id",
    "root_cause_class",
    "evidence_case_ids",
    "failed_metrics",
    "suspected_modules",
    "confidence",
    "allowed_change_scope",
    "forbidden_changes",
    "verification_cases",
    "regression_cases",
)


class RemediationPlanError(RuntimeError):
    pass


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RemediationPlanError(f"invalid evidence file: {path}") from exc


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RemediationPlanError(f"unreadable evidence file: {path}") from exc


def case_violations(metrics: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    """Null gerekli metrik fail-closed ihlaldir; hiçbir değer doldurulmaz."""
    failed = []
    for name, threshold in thresholds.items():
        value = metrics.get(name)
        if value is None or float(value) < float(threshold):
            failed.append(name)
    return sorted(failed)


def verify_plan(
    plan_path: Path = DEFAULT_PLAN,
    results_path: Path = DEFAULT_RESULTS,
    decision_path: Path = DEFAULT_DECISION,
) -> dict[str, Any]:
    plan = _load(plan_path)
    results = _load(results_path)
    decision = _load(decision_path)

    if plan.get("schema") != PLAN_SCHEMA:
        raise RemediationPlanError(f"unexpected plan schema: {plan.get('schema')}")

    # Kaynak evidence bağları: plan yalnız bu generation için geçerlidir.
    if plan.get("source_results_sha256") != _sha256(results_path):
        raise RemediationPlanError("plan is not bound to the committed pipeline results")
    if plan.get("source_decision_file_sha256") != _sha256(decision_path):
        raise RemediationPlanError("plan is not bound to the committed quality decision file")
    if plan.get("source_decision_sha256") != decision.get("decision_sha256"):
        raise RemediationPlanError("plan decision digest does not match the canonical decision digest")
    if plan.get("source_cases_sha256") != decision.get("cases_sha256"):
        raise RemediationPlanError("plan corpus identity does not match the canonical case-set SHA")
    if plan.get("source_measurement_head_sha") != decision.get("measurement_head_sha"):
        raise RemediationPlanError("plan measurement head SHA does not match the decision")

    # Karar korunur: NO-GO ve RFV-4 bloğu zayıflatılamaz.
    if plan.get("release_decision") != "no_go" or decision.get("release_decision") != "no_go":
        raise RemediationPlanError("RFV-3E plan requires the canonical no_go decision")
    if plan.get("rfv4_allowed") is not False or decision.get("rfv4_allowed") is not False:
        raise RemediationPlanError("rfv4_allowed must remain false")

    # Eşikler değişmemiş olmalı (hem plana hem karara karşı).
    thresholds = plan.get("thresholds")
    if thresholds != REQUIRED_THRESHOLDS:
        raise RemediationPlanError(f"threshold drift in plan: {thresholds}")
    gate_metrics = (decision.get("quality_gate") or {}).get("metrics") or {}
    for name, expected in REQUIRED_THRESHOLDS.items():
        actual = (gate_metrics.get(name) or {}).get("threshold")
        if actual != expected:
            raise RemediationPlanError(f"threshold drift in decision for {name}: {actual}")

    rows = {row.get("case_id"): row for row in results.get("results") or []}
    if len(rows) != 24 or plan.get("case_count") != 24 or decision.get("case_count") != 24:
        raise RemediationPlanError("corpus must contain exactly the 24 qualified cases")

    clusters = plan.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        raise RemediationPlanError("plan must define clusters")

    seen: dict[str, str] = {}
    for cluster in clusters:
        missing_fields = [name for name in REQUIRED_CLUSTER_FIELDS if name not in cluster]
        if missing_fields:
            raise RemediationPlanError(f"cluster {cluster.get('cluster_id')} missing fields: {missing_fields}")
        if cluster["confidence"] not in ALLOWED_CONFIDENCE:
            raise RemediationPlanError(f"invalid confidence for {cluster['cluster_id']}")

        evidence_ids = cluster["evidence_case_ids"]
        if not evidence_ids:
            raise RemediationPlanError(f"cluster {cluster['cluster_id']} has no evidence cases")
        union: set[str] = set()
        for case_id in evidence_ids:
            if case_id not in rows:
                raise RemediationPlanError(f"unknown case in cluster {cluster['cluster_id']}: {case_id}")
            if case_id in seen:
                raise RemediationPlanError(f"case {case_id} appears in both {seen[case_id]} and {cluster['cluster_id']}")
            seen[case_id] = cluster["cluster_id"]
            violated = case_violations(rows[case_id]["metrics"], REQUIRED_THRESHOLDS)
            if not violated:
                raise RemediationPlanError(f"passing case wrongly listed for remediation: {case_id}")
            union.update(violated)
        if sorted(union) != sorted(cluster["failed_metrics"]):
            raise RemediationPlanError(
                f"failed_metrics for {cluster['cluster_id']} do not match measured violations: "
                f"{sorted(cluster['failed_metrics'])} != {sorted(union)}"
            )

        if not set(cluster["verification_cases"]) <= set(evidence_ids):
            raise RemediationPlanError(f"verification cases outside evidence for {cluster['cluster_id']}")
        regression = set(cluster["regression_cases"])
        if not regression:
            raise RemediationPlanError(f"cluster {cluster['cluster_id']} needs regression sentinels")
        if regression & set(evidence_ids):
            raise RemediationPlanError(f"regression cases overlap evidence for {cluster['cluster_id']}")
        if not regression <= set(rows):
            raise RemediationPlanError(f"unknown regression case for {cluster['cluster_id']}")

        # Tentative kök neden için production fix kapsamı yasaktır: önce tanı.
        if cluster["confidence"] == "tentative":
            scope_text = " ".join(cluster["allowed_change_scope"]).lower()
            if "diagnostics" not in scope_text:
                raise RemediationPlanError(
                    f"tentative cluster {cluster['cluster_id']} must be diagnostics-first"
                )

    failing = sorted(case_id for case_id, row in rows.items() if case_violations(row["metrics"], REQUIRED_THRESHOLDS))
    if sorted(seen) != failing:
        unclustered = sorted(set(failing) - set(seen))
        extra = sorted(set(seen) - set(failing))
        raise RemediationPlanError(f"cluster partition mismatch: unclustered={unclustered} extra={extra}")
    if plan.get("failing_case_count") != len(failing) or plan.get("passing_case_count") != 24 - len(failing):
        raise RemediationPlanError("plan case counts do not match measured violations")

    return {
        "schema": "vektoryum-rfv3e-remediation-plan-verification-v1",
        "plan_sha256": _sha256(plan_path),
        "case_count": 24,
        "failing_case_count": len(failing),
        "passing_case_count": 24 - len(failing),
        "cluster_count": len(clusters),
        "release_decision": "no_go",
        "rfv4_allowed": False,
        "status": "verified",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    args = parser.parse_args()
    report = verify_plan(args.plan, args.results, args.decision)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
