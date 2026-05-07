import json
from pathlib import Path
from dataclasses import asdict

HISTORY_FILE = Path("history/eval_history.json")
HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)


def append_run(snapshot, detector_result):
    record = {
        "provider": snapshot.provider,
        "model": snapshot.model,
        "run_id": snapshot.run_id,
        "timestamp": snapshot.timestamp,
        "prompt_hash": snapshot.prompt_hash,
        "git_commit": snapshot.git_commit,
        "decision": detector_result.decision,
        "pass_rate": snapshot.pass_rate,
        "mean_accuracy": snapshot.mean_accuracy,
        "task_scores": snapshot.task_scores,
        "metadata": snapshot.metadata,
    }

    history = []

    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            history = json.load(f)

    history.append(record)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
