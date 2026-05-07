import json
from pathlib import Path

HISTORY_FILE = Path("history/eval_history.json")


def query_language_regression(language: str):
    if not HISTORY_FILE.exists():
        return []

    with open(HISTORY_FILE) as f:
        history = json.load(f)

    results = []

    for run in history:
        scores = run.get("metadata", {}).get("language_scores", {})

        if language in scores:
            results.append(
                {
                    "timestamp": run["timestamp"],
                    "commit": run["git_commit"],
                    "score": sum(scores[language]) / len(scores[language]),
                }
            )

    return results
