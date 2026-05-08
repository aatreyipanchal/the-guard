from datetime import datetime
from pathlib import Path
import sys

import streamlit as st

try:
    from dashboard.query import (
        answer_quality_drop_question,
        historical_accuracy,
        load_history,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dashboard.query import (
        answer_quality_drop_question,
        historical_accuracy,
        load_history,
    )


def _collect_filter_options(history: list[dict]) -> dict[str, list[str]]:
    providers = sorted({run.get("provider", "") for run in history if run.get("provider")})
    models = sorted({run.get("model", "") for run in history if run.get("model")})
    prompt_versions = sorted({run.get("prompt_version", "") for run in history if run.get("prompt_version")})

    task_types = set()
    languages = set()
    for run in history:
        task_types.update(run.get("task_accuracy_avg", {}).keys())
        languages.update(run.get("language_accuracy_avg", {}).keys())
        for task_type, lang_scores in run.get("task_language_accuracy_avg", {}).items():
            task_types.add(task_type)
            languages.update(lang_scores.keys())

    return {
        "providers": providers,
        "models": models,
        "prompt_versions": prompt_versions,
        "task_types": sorted(task_types),
        "languages": sorted(languages),
    }


def _render_history(rows: list[dict]) -> None:
    if not rows:
        st.info("No history rows match the current filters.")
        return

    scores = [row["score"] for row in rows]
    st.line_chart({"score": scores})
    st.dataframe(rows, use_container_width=True)


def _render_drop_summary(result: dict) -> None:
    st.subheader("Drop Detector")
    if not result.get("drop_found"):
        st.info(result["message"])
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Previous score", f"{result['previous_score']:.4f}")
    col2.metric("New score", f"{result['new_score']:.4f}")
    col3.metric("Delta", f"{result['delta']:.4f}")

    st.success(result["message"])
    st.json(
        {
            "dropped_at": result["dropped_at"],
            "previous_run_at": result["previous_run_at"],
            "git_commit": result["git_commit"],
            "git_branch": result["git_branch"],
            "prompt_version": result["prompt_version"],
            "task_type": result["task_type"],
            "provider": result["provider"],
            "model": result["model"],
        }
    )


def main() -> None:
    st.set_page_config(page_title="The Guard Dashboard", layout="wide")
    st.title("The Guard Dashboard")

    history = load_history()
    if not history:
        st.warning("No eval history found. Run the eval pipeline first.")
        return

    options = _collect_filter_options(history)

    with st.sidebar:
        st.header("Filters")
        provider = st.selectbox("Provider", ["All"] + options["providers"])
        model = st.selectbox("Model", ["All"] + options["models"])
        task_type = st.selectbox("Output type", ["All"] + options["task_types"])
        language = st.selectbox("Language", ["All"] + options["languages"])
        prompt_version = st.selectbox("Prompt version", ["All"] + options["prompt_versions"])
        min_drop = st.slider("Minimum drop threshold", min_value=0.01, max_value=0.25, value=0.03, step=0.01)

    provider_filter = None if provider == "All" else provider
    model_filter = None if model == "All" else model
    task_filter = None if task_type == "All" else task_type
    language_filter = None if language == "All" else language
    prompt_filter = None if prompt_version == "All" else prompt_version

    rows = historical_accuracy(
        task_type=task_filter,
        language=language_filter,
        provider=provider_filter,
        model=model_filter,
        prompt_version=prompt_filter,
    )

    latest_timestamp = max(datetime.fromisoformat(run["timestamp"].replace("Z", "+00:00")) for run in history)
    col1, col2, col3 = st.columns(3)
    col1.metric("Runs loaded", str(len(history)))
    col2.metric("Filtered rows", str(len(rows)))
    col3.metric("Latest run", latest_timestamp.strftime("%Y-%m-%d %H:%M UTC"))

    st.subheader("Historical Accuracy")
    _render_history(rows)

    if language_filter:
        drop_result = answer_quality_drop_question(
            language=language_filter,
            task_type=task_filter,
            provider=provider_filter,
            model=model_filter,
            min_drop=min_drop,
        )
        _render_drop_summary(drop_result)
    else:
        st.subheader("Drop Detector")
        st.info("Select a language to detect historical quality drops.")


if __name__ == "__main__":
    main()
