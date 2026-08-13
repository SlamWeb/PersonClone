"""Run one queued retrieval-evaluation job without resetting other workers."""

from __future__ import annotations

import argparse
from pathlib import Path

from personaforge.web.retrieval_eval_jobs import RetrievalEvalJobConfig, RetrievalEvalJobManager


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-name", default="BAAI/bge-m3")
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    manager = RetrievalEvalJobManager(
        RetrievalEvalJobConfig(
            data_dir=args.data_dir,
            model_name=args.model_name,
            embedding_device=args.embedding_device,
            use_fp16=not args.no_fp16,
            working_dir=Path.cwd(),
        )
    )
    job = manager.get(args.job_id)
    if job["status"] in {"failed", "interrupted", "paused_budget"}:
        manager.resume(args.job_id)
    if not manager.run_once():
        raise RuntimeError(f"Job is not queued: {args.job_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
