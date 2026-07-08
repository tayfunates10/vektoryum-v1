"""Local worker skeleton for processing queued /v1/vectorize jobs."""
from __future__ import annotations

import json
import os
from pathlib import Path

from services.vectorizer_worker.vectorizer import PerfectVectorizer

DATA_ROOT = Path(os.getenv("VEKTORYUM_V2_DATA_ROOT", "/tmp/vektoryum_v2"))
QUEUE_DIR = DATA_ROOT / "queue"
RESULTS_DIR = DATA_ROOT / "results"


def process_one(queue_file: Path) -> Path:
    payload = json.loads(queue_file.read_text(encoding="utf-8"))
    job_id = payload["job_id"]
    input_path = Path(payload["input_path"])
    output_dir = RESULTS_DIR / job_id
    artifacts = PerfectVectorizer().vectorize(input_path, output_dir)
    result = {
        "job_id": job_id,
        "status": "completed",
        "outputs": {"svg": str(artifacts.svg_path)},
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    queue_file.unlink(missing_ok=True)
    return result_path


def main() -> None:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    for queue_file in sorted(QUEUE_DIR.glob("*.json")):
        process_one(queue_file)


if __name__ == "__main__":
    main()
