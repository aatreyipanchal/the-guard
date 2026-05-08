import json
from pathlib import Path

HISTORY_FILE = Path("history/eval_history.json")
HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)


def _avg(values):
    return sum(values) / len(values) if values else 0.0


def _build_task_accuracy_avg(snapshot) -> dict:
    return {
        task_type: _avg(scores)
        for task_type, scores in snapshot.task_scores.items()
    }


def _build_language_accuracy_avg(snapshot) -> dict:
    language_scores = snapshot.metadata.get("language_scores", {})
    return {
        language: _avg(scores)
        for language, scores in language_scores.items()
    }


def _build_task_language_accuracy_avg(snapshot) -> dict:
    task_language_scores = snapshot.metadata.get("task_language_scores", {})
    return {
        task_type: {
            language: _avg(scores)
            for language, scores in languages.items()
        }
        for task_type, languages in task_language_scores.items()
    }


def append_run(snapshot, detector_result):
    prompt_metadata = snapshot.metadata.get("prompt_metadata", {})
    record = {
        "provider": snapshot.provider,
        "model": snapshot.model,
        "run_id": snapshot.run_id,
        "timestamp": snapshot.timestamp,
        "prompt_hash": snapshot.prompt_hash,
        "prompt_version": snapshot.prompt_version or snapshot.prompt_hash,
        "task_prompt_hashes": snapshot.task_prompt_hashes,
        "git_commit": snapshot.git_commit,
        "git_branch": snapshot.git_branch,
        "decision": detector_result.decision,
        "pass_rate": snapshot.pass_rate,
        "mean_accuracy": snapshot.mean_accuracy,
        "task_scores": snapshot.task_scores,
        "task_accuracy_avg": _build_task_accuracy_avg(snapshot),
        "language_accuracy_avg": _build_language_accuracy_avg(snapshot),
        "task_language_accuracy_avg": _build_task_language_accuracy_avg(snapshot),
        "metadata": snapshot.metadata,
        "prompt_metadata": {
            "prompt_hash": prompt_metadata.get("prompt_hash", snapshot.prompt_hash),
            "prompt_version": prompt_metadata.get("prompt_hash", snapshot.prompt_version or snapshot.prompt_hash),
            "task_prompt_hashes": prompt_metadata.get("task_prompt_hashes", snapshot.task_prompt_hashes),
            "git_commit": prompt_metadata.get("git_commit", snapshot.git_commit),
            "git_branch": prompt_metadata.get("git_branch", snapshot.git_branch),
        },
    }

    history = []

    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, encoding="utf-8") as f:
            history = json.load(f)

    history.append(record)

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
