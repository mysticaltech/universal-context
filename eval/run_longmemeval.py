#!/usr/bin/env python3
"""LongMemEval runner wrapper for UC.

This wrapper evaluates a configured chat model on a LongMemEval oracle subset.
It is designed to run with UC's configured LLM provider/model (defaulting to
OpenRouter + config model), and records a machine-readable run report.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from openai import OpenAI

from universal_context.config import UCConfig

DEFAULT_REPO = "https://github.com/xiaowu0162/LongMemEval.git"
DEFAULT_DATASET_REL = Path("data/longmemeval_oracle.json")
DEFAULT_DATASET_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
    "longmemeval_oracle.json"
)
DEFAULT_MODELS = {
    "openrouter": "x-ai/grok-4.1-fast",
    "openai": "gpt-4.1-mini",
}


def _run(
    cmd: list[str],
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=check,
    )


def _ensure_repo(repo_dir: Path, repo_url: str) -> None:
    if repo_dir.exists():
        _run(["git", "pull", "--ff-only"], cwd=repo_dir, check=False)
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", repo_url, str(repo_dir)])


def _ensure_dataset(dataset_path: Path, dataset_url: str) -> None:
    if dataset_path.exists():
        return
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(dataset_url, timeout=60) as resp:
        raw = resp.read()
    dataset_path.write_bytes(raw)


def _load_entries(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported dataset structure in {path}")


def _resolve_provider(config: UCConfig, provider: str) -> str:
    normalized = (provider or "auto").strip().lower()
    if normalized != "auto":
        return normalized
    for candidate in ("openrouter", "openai"):
        if config.get_api_key(candidate):
            return candidate
    return "openrouter"


def _resolve_model(provider: str, cli_model: str, config_model: str) -> str:
    raw = (cli_model or "").strip() or (config_model or "").strip() or DEFAULT_MODELS[provider]
    if provider == "openrouter" and raw.startswith("openrouter/"):
        return raw.removeprefix("openrouter/")
    if provider == "openai" and raw.startswith("openai/"):
        return raw.removeprefix("openai/")
    return raw


def _resolve_base_url(provider: str, cli_base_url: str) -> str | None:
    if cli_base_url.strip():
        return cli_base_url.strip()
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1"
    return None


def _normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    return " ".join(re.findall(r"[a-z0-9]+", lowered))


def _approx_match(answer: str, hypothesis: str) -> bool:
    a = _normalize_text(answer)
    h = _normalize_text(hypothesis)
    if not a or not h:
        return False
    return a in h or h in a


def _build_prompt(entry: dict[str, Any], max_sessions: int) -> str:
    sessions = list(
        zip(
            entry.get("haystack_dates", []),
            entry.get("haystack_sessions", []),
        )
    )[:max_sessions]

    history_blocks: list[str] = []
    for idx, (date, sess) in enumerate(sessions, start=1):
        history_blocks.append(
            f"### Session {idx}\n"
            f"Date: {date}\n"
            f"Content:\n{json.dumps(sess, ensure_ascii=False)}"
        )

    history = "\n\n".join(history_blocks)
    question_date = entry.get("question_date", "unknown")
    question = entry.get("question", "")

    return (
        "You are answering a long-term memory QA question based only on the "
        "provided timestamped chat history.\n\n"
        "Return only the direct answer. If the history is insufficient, say \"I don't know\".\n\n"
        f"Question Date: {question_date}\n"
        f"Question: {question}\n\n"
        f"History:\n{history}\n\n"
        "Answer:"
    )


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run LongMemEval oracle subset with UC-configured LLM."
    )
    parser.add_argument("--repo-dir", type=Path, default=Path(".cache/LongMemEval"))
    parser.add_argument("--repo-url", default=DEFAULT_REPO)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--chat-file", default=str(DEFAULT_DATASET_REL))
    parser.add_argument("--dataset-url", default=DEFAULT_DATASET_URL)
    parser.add_argument("--provider", choices=["auto", "openrouter", "openai"], default="auto")
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--task", default="qa")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-questions", type=int, default=10)
    parser.add_argument("--max-sessions", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--install-requirements", action="store_true")
    parser.add_argument("--allow-skip", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--hyp-out",
        type=Path,
        default=Path("eval/longmemeval_hypotheses_latest.jsonl"),
    )
    parser.add_argument("--json-out", type=Path, default=Path("eval/longmemeval_latest.json"))
    args = parser.parse_args()

    config = UCConfig.load()
    provider = _resolve_provider(config, args.provider)
    model = _resolve_model(provider, args.model, config.llm_model)
    base_url = _resolve_base_url(provider, args.base_url)
    api_key = config.get_api_key(provider)

    timestamp = datetime.now(UTC).isoformat()
    dataset_path = args.repo_dir / args.chat_file
    payload: dict[str, Any] = {
        "timestamp": timestamp,
        "provider": provider,
        "model": model,
        "task": args.task,
        "split": args.split,
        "chat_file": args.chat_file,
        "dataset_path": str(dataset_path),
        "base_url": base_url,
        "max_questions": args.max_questions,
        "max_sessions": args.max_sessions,
        "hypotheses_path": str(args.hyp_out),
    }

    if provider not in {"openrouter", "openai"}:
        payload["status"] = "failed"
        payload["reason"] = f"Unsupported provider for this wrapper: {provider}"
        _write_payload(args.json_out, payload)
        print(json.dumps(payload, indent=2))
        return 2

    if not api_key:
        payload["status"] = "skipped"
        payload["reason"] = f"API key is not set for provider '{provider}'"
        _write_payload(args.json_out, payload)
        print(json.dumps(payload, indent=2))
        return 0 if args.allow_skip else 2

    if args.dry_run:
        payload["status"] = "dry_run"
        _write_payload(args.json_out, payload)
        print(json.dumps(payload, indent=2))
        return 0

    _ensure_repo(args.repo_dir, args.repo_url)

    if args.install_requirements:
        _run(
            [args.python, "-m", "pip", "install", "-r", "requirements-lite.txt"],
            cwd=args.repo_dir,
        )

    _ensure_dataset(dataset_path, args.dataset_url)
    entries = _load_entries(dataset_path)
    selected = entries[: max(0, args.max_questions)]

    if not selected:
        payload["status"] = "failed"
        payload["reason"] = "No questions selected for execution"
        _write_payload(args.json_out, payload)
        print(json.dumps(payload, indent=2))
        return 2

    client = OpenAI(api_key=api_key, base_url=base_url)

    args.hyp_out.parent.mkdir(parents=True, exist_ok=True)
    answered = 0
    approx_matches = 0
    errors: list[dict[str, str]] = []

    with args.hyp_out.open("w", encoding="utf-8") as out_f:
        for entry in selected:
            qid = str(entry.get("question_id", ""))
            try:
                prompt = _build_prompt(entry, max_sessions=max(1, args.max_sessions))
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    n=1,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                hypothesis = (completion.choices[0].message.content or "").strip()
                out_f.write(
                    json.dumps(
                        {"question_id": qid, "hypothesis": hypothesis},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                answered += 1
                if _approx_match(str(entry.get("answer", "")), hypothesis):
                    approx_matches += 1
            except Exception as exc:
                errors.append({"question_id": qid, "error": f"{type(exc).__name__}: {exc}"})

    payload["requested_questions"] = len(selected)
    payload["answered_questions"] = answered
    payload["failed_questions"] = len(errors)
    payload["approx_accuracy"] = round(approx_matches / answered, 4) if answered else None

    if errors:
        payload["errors"] = errors[:10]

    if answered == 0:
        payload["status"] = "failed"
        payload["reason"] = "No questions were answered successfully"
        rc = 2
    elif errors:
        payload["status"] = "partial"
        rc = 0
    else:
        payload["status"] = "ok"
        rc = 0

    _write_payload(args.json_out, payload)
    print(json.dumps(payload, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
