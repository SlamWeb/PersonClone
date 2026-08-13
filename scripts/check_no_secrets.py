"""Fail CI when tracked files look like local secrets or private runtime data."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


FORBIDDEN_PATH_PARTS = {
    ".env",
    "zhihu_storage_state.json",
    "cookies.json",
}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "OpenAI-compatible key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "Tavily key": re.compile(r"\btvly-[A-Za-z0-9_-]{16,}\b"),
}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=root
    ).decode("utf-8").split("\0")
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root
    ).decode("utf-8").split("\0")
    candidates = sorted(set(filter(None, [*tracked, *untracked])))
    findings: list[str] = []
    for relative in candidates:
        path = Path(relative)
        normalized = relative.replace("\\", "/")
        if normalized.startswith("data/"):
            findings.append(f"private data path is tracked: {relative}")
            continue
        if path.name in FORBIDDEN_PATH_PARTS and path.name != ".env.example":
            findings.append(f"secret-bearing filename is tracked: {relative}")
            continue
        full_path = root / path
        try:
            text = full_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{label} pattern found in {relative}")

    if findings:
        print("Secret scan failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print(f"Secret scan passed for {len(candidates)} tracked or candidate files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
