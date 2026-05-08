import argparse
import json
from datetime import datetime
from pathlib import Path

HISTORY_FILE = Path("history/eval_history.json")


def _avg(values):
    return sum(values) / len(values) if values else 0.0


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []

    with open(HISTORY_FILE, encoding="utf-8") as f:
        history = json.load(f)

    normalized = []
    for run in history:
        task_scores = run.get("task_scores", {})
        metadata = run.get("metadata", {})
        language_scores = metadata.get("language_scores", {})
        task_language_scores = metadata.get("task_language_scores", {})
        normalized.append(
            {
                **run,
                "prompt_version": run.get("prompt_version") or run.get("prompt_hash", ""),
                "task_accuracy_avg": run.get("task_accuracy_avg", {
                    task_type: _avg(scores)
                    for task_type, scores in task_scores.items()
                }),
                "language_accuracy_avg": run.get("language_accuracy_avg", {
                    language: _avg(scores)
                    for language, scores in language_scores.items()
                }),
                "task_language_accuracy_avg": run.get("task_language_accuracy_avg", {
                    task_type: {
                        language: _avg(scores)
                        for language, scores in languages.items()
                    }
                    for task_type, languages in task_language_scores.items()
                }),
            }
        )

    return normalized


def _matches(run: dict, provider: str | None, model: str | None, prompt_version: str | None) -> bool:
    if provider and run.get("provider") != provider:
        return False
    if model and run.get("model") != model:
        return False
    if prompt_version and run.get("prompt_version") != prompt_version:
        return False
    return True


def _parse_ts(timestamp: str):
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def historical_accuracy(
    task_type: str | None = None,
    language: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
) -> list[dict]:
    rows = []
    for run in load_history():
        if not _matches(run, provider=provider, model=model, prompt_version=prompt_version):
            continue

        if task_type and language:
            score = (
                run.get("task_language_accuracy_avg", {})
                .get(task_type, {})
                .get(language)
            )
        elif task_type:
            score = run.get("task_accuracy_avg", {}).get(task_type)
        elif language:
            score = run.get("language_accuracy_avg", {}).get(language)
        else:
            score = run.get("mean_accuracy")

        if score is None:
            continue

        rows.append(
            {
                "timestamp": run["timestamp"],
                "provider": run.get("provider"),
                "model": run.get("model"),
                "task_type": task_type or "overall",
                "language": language or "all",
                "score": score,
                "decision": run.get("decision"),
                "git_commit": run.get("git_commit"),
                "git_branch": run.get("git_branch"),
                "prompt_version": run.get("prompt_version"),
            }
        )

    return sorted(rows, key=lambda row: _parse_ts(row["timestamp"]))


def detect_quality_drops(
    language: str,
    task_type: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    min_drop: float = 0.03,
) -> list[dict]:
    series = historical_accuracy(
        task_type=task_type,
        language=language,
        provider=provider,
        model=model,
    )
    events = []
    previous = None

    for current in series:
        if previous is not None:
            delta = current["score"] - previous["score"]
            if delta <= -abs(min_drop):
                events.append(
                    {
                        "language": language,
                        "task_type": current["task_type"],
                        "provider": current["provider"],
                        "model": current["model"],
                        "previous_timestamp": previous["timestamp"],
                        "timestamp": current["timestamp"],
                        "previous_score": previous["score"],
                        "score": current["score"],
                        "delta": delta,
                        "git_commit": current["git_commit"],
                        "git_branch": current["git_branch"],
                        "prompt_version": current["prompt_version"],
                    }
                )
        previous = current

    return events


def answer_quality_drop_question(
    language: str = "telugu",
    task_type: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    min_drop: float = 0.03,
) -> dict:
    events = detect_quality_drops(
        language=language,
        task_type=task_type,
        provider=provider,
        model=model,
        min_drop=min_drop,
    )

    if not events:
        return {
            "language": language,
            "task_type": task_type or "all",
            "provider": provider or "all",
            "model": model or "all",
            "drop_found": False,
            "message": f"No {language} quality drop found for the selected filters.",
        }

    first_drop = events[0]
    return {
        "language": language,
        "task_type": first_drop["task_type"],
        "provider": first_drop["provider"],
        "model": first_drop["model"],
        "drop_found": True,
        "dropped_at": first_drop["timestamp"],
        "previous_run_at": first_drop["previous_timestamp"],
        "previous_score": first_drop["previous_score"],
        "new_score": first_drop["score"],
        "delta": first_drop["delta"],
        "git_commit": first_drop["git_commit"],
        "git_branch": first_drop["git_branch"],
        "prompt_version": first_drop["prompt_version"],
        "message": (
            f"{language} quality dropped on {first_drop['timestamp']} "
            f"from {first_drop['previous_score']:.4f} to {first_drop['score']:.4f} "
            f"(delta {first_drop['delta']:.4f}) on commit {first_drop['git_commit']}."
        ),
    }


def query_language_regression(language: str):
    return historical_accuracy(language=language)


def _print_json(payload):
    print(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Historical eval dashboard queries")
    subparsers = parser.add_subparsers(dest="command", required=True)

    history_parser = subparsers.add_parser("history", help="Show historical accuracy slices")
    history_parser.add_argument("--task-type")
    history_parser.add_argument("--language")
    history_parser.add_argument("--provider")
    history_parser.add_argument("--model")
    history_parser.add_argument("--prompt-version")

    drop_parser = subparsers.add_parser("drop", help="Detect when quality dropped")
    drop_parser.add_argument("--language", default="telugu")
    drop_parser.add_argument("--task-type")
    drop_parser.add_argument("--provider")
    drop_parser.add_argument("--model")
    drop_parser.add_argument("--min-drop", type=float, default=0.03)

    args = parser.parse_args()

    if args.command == "history":
        _print_json(
            historical_accuracy(
                task_type=args.task_type,
                language=args.language,
                provider=args.provider,
                model=args.model,
                prompt_version=args.prompt_version,
            )
        )
        return

    if args.command == "drop":
        _print_json(
            answer_quality_drop_question(
                language=args.language,
                task_type=args.task_type,
                provider=args.provider,
                model=args.model,
                min_drop=args.min_drop,
            )
        )


if __name__ == "__main__":
    main()
