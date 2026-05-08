import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from difflib import unified_diff
from pathlib import Path
from typing import Any

PROMPT_HISTORY_DIR = Path("history/prompts")
PROMPT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
REPO_ROOT = Path(__file__).resolve().parent.parent

_PROMPT_SOURCE_PATHS = [
    "prompts/",
    "tests.json",
    "tests/golden_suite.py",
]
_WINDOWS_GIT_CANDIDATES = [
    r"C:\Program Files\Git\cmd\git.exe",
    r"C:\Program Files\Git\bin\git.exe",
    r"C:\Program Files (x86)\Git\cmd\git.exe",
    r"C:\Program Files (x86)\Git\bin\git.exe",
]


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def normalize_prompt_bundle(prompt: str | dict[str, Any]) -> dict[str, dict[str, str]]:
    if isinstance(prompt, str):
        return {"adhoc": {"prompt": prompt}}

    normalized: dict[str, dict[str, str]] = {}
    for task_type, prompts in prompt.items():
        if isinstance(prompts, str):
            normalized[task_type] = {"prompt": prompts}
            continue

        if not isinstance(prompts, dict):
            raise TypeError(
                f"Prompt bundle for task '{task_type}' must be a string or dict, got {type(prompts).__name__}"
            )

        normalized[task_type] = {
            str(case_id): str(case_prompt)
            for case_id, case_prompt in sorted(prompts.items())
        }

    return dict(sorted(normalized.items()))


def _render_task_prompt_text(case_prompts: dict[str, str]) -> str:
    blocks = []
    for case_id, prompt in sorted(case_prompts.items()):
        blocks.append(f"## {case_id}\n{prompt.rstrip()}")
    return "\n\n".join(blocks).strip()


def render_prompt_bundle(bundle: dict[str, dict[str, str]]) -> str:
    sections = []
    for task_type, case_prompts in sorted(bundle.items()):
        sections.append(f"# TASK: {task_type}\n{_render_task_prompt_text(case_prompts)}")
    return "\n\n".join(sections).strip()


def build_prompt_bundle(test_cases: list[Any]) -> dict[str, dict[str, str]]:
    bundle: dict[str, dict[str, str]] = {}
    for tc in test_cases:
        bundle.setdefault(tc.task_type, {})[tc.id] = tc.prompt
    return normalize_prompt_bundle(bundle)


def compute_task_prompt_hashes(bundle: dict[str, dict[str, str]]) -> dict[str, str]:
    return {
        task_type: hash_prompt(_render_task_prompt_text(case_prompts))
        for task_type, case_prompts in sorted(bundle.items())
    }


def hash_prompt_bundle(bundle: dict[str, dict[str, str]]) -> str:
    return hash_prompt(render_prompt_bundle(bundle))


def _git_executable() -> str | None:
    discovered = shutil.which("git")
    if discovered:
        return discovered

    for candidate in _WINDOWS_GIT_CANDIDATES:
        if Path(candidate).exists():
            return candidate

    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        user_candidate = Path(local_app_data) / "Programs" / "Git" / "cmd" / "git.exe"
        if user_candidate.exists():
            return str(user_candidate)

    return None


def _git_dir() -> Path | None:
    git_entry = REPO_ROOT / ".git"
    if git_entry.is_dir():
        return git_entry

    if git_entry.is_file():
        try:
            content = git_entry.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if content.startswith("gitdir:"):
            git_dir = content.split(":", 1)[1].strip()
            return (REPO_ROOT / git_dir).resolve()

    return None


def _head_ref() -> tuple[str, str]:
    git_dir = _git_dir()
    if not git_dir:
        return "unknown", "unknown"

    head_path = git_dir / "HEAD"
    try:
        head_value = head_path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown", "unknown"

    if head_value.startswith("ref:"):
        ref_name = head_value.split(":", 1)[1].strip()
        ref_path = git_dir / Path(ref_name)
        if ref_path.exists():
            try:
                commit = ref_path.read_text(encoding="utf-8").strip()
            except OSError:
                commit = "unknown"
        else:
            packed_refs = git_dir / "packed-refs"
            commit = "unknown"
            if packed_refs.exists():
                try:
                    for line in packed_refs.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line or line.startswith("#") or line.startswith("^"):
                            continue
                        sha, name = line.split(" ", 1)
                        if name == ref_name:
                            commit = sha
                            break
                except OSError:
                    commit = "unknown"
        return commit or "unknown", Path(ref_name).name or "unknown"

    return head_value or "unknown", "DETACHED"


def _run_git_command(args: list[str]) -> str:
    git_path = _git_executable()
    if not git_path:
        return ""

    try:
        return (
            subprocess.check_output([git_path, *args], stderr=subprocess.DEVNULL, cwd=REPO_ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return ""


def get_git_commit() -> str:
    return _run_git_command(["rev-parse", "HEAD"]) or _head_ref()[0]


def get_git_branch() -> str:
    return _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"]) or _head_ref()[1]


def get_git_metadata() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()

        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
        ).strip()

        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
            ).strip()
        )

        return {
            "git_commit": commit,
            "git_branch": branch,
            "git_dirty": dirty,
        }

    except Exception:
        return {
            "git_commit": "unknown",
            "git_branch": "unknown",
            "git_dirty": False,
        }


def get_prompt_diff() -> str:
    diff = _run_git_command(["diff", "HEAD~1", "--", *_PROMPT_SOURCE_PATHS])
    return diff or ""


def diff_prompt_bundles(
    current_bundle: dict[str, dict[str, str]],
    baseline_bundle: dict[str, dict[str, str]],
    task_types: list[str] | None = None,
) -> dict[str, str]:
    selected_tasks = sorted(
        task_types
        if task_types is not None
        else set(current_bundle) | set(baseline_bundle)
    )
    diffs: dict[str, str] = {}

    for task_type in selected_tasks:
        current_text = _render_task_prompt_text(current_bundle.get(task_type, {}))
        baseline_text = _render_task_prompt_text(baseline_bundle.get(task_type, {}))

        if current_text == baseline_text:
            continue

        diff_lines = unified_diff(
            baseline_text.splitlines(),
            current_text.splitlines(),
            fromfile=f"baseline/{task_type}",
            tofile=f"current/{task_type}",
            lineterm="",
        )
        diffs[task_type] = "\n".join(diff_lines).strip()

    return diffs


def summarize_prompt_diffs(prompt_diffs_by_task: dict[str, str], max_chars_per_task: int = 1200) -> str:
    if not prompt_diffs_by_task:
        return ""

    sections = []
    for task_type, diff_text in sorted(prompt_diffs_by_task.items()):
        clipped = diff_text[:max_chars_per_task].rstrip()
        sections.append(f"## {task_type}\n{clipped}")
    return "\n\n".join(sections).strip()


def create_prompt_metadata(prompt: str | dict[str, Any], provider: str) -> dict[str, Any]:
    prompt_bundle = normalize_prompt_bundle(prompt)
    task_prompt_hashes = compute_task_prompt_hashes(prompt_bundle)
    metadata = {
        "prompt_hash": hash_prompt_bundle(prompt_bundle),
        "provider": provider,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "git_branch": get_git_branch(),
        "prompt_diff": get_prompt_diff(),
        "task_prompt_hashes": task_prompt_hashes,
        "prompt_bundle": prompt_bundle,
    }

    provider_dir = PROMPT_HISTORY_DIR / provider
    provider_dir.mkdir(parents=True, exist_ok=True)
    path = provider_dir / f"{metadata['prompt_hash']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    metadata["prompt_history_path"] = str(path)
    return metadata
