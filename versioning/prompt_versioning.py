import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROMPT_HISTORY_DIR = Path("history/prompts")
PROMPT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def get_git_branch() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def get_prompt_diff() -> str:
    try:
        return (
            subprocess.check_output(["git", "diff", "HEAD~1", "--", "prompts/"])
            .decode()
            .strip()
        )
    except Exception:
        return ""


def create_prompt_metadata(prompt: str, provider: str) -> dict:
    metadata = {
        "prompt_hash": hash_prompt(prompt),
        "provider": provider,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "git_branch": get_git_branch(),
        "prompt_diff": get_prompt_diff(),
    }

    path = PROMPT_HISTORY_DIR / f"{metadata['prompt_hash']}.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata
