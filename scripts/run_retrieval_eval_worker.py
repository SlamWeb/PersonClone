"""Keep the restart-safe retrieval evaluation worker alive outside the Web process."""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

from dotenv import load_dotenv

from personaforge.web.retrieval_eval_jobs import RetrievalEvalJobConfig, RetrievalEvalJobManager


def _active_jobs(database_path: Path) -> list[tuple[str, str, str]]:
    with sqlite3.connect(database_path) as connection:
        return list(
            connection.execute(
                """
                SELECT id, author, status
                FROM retrieval_eval_jobs
                WHERE status IN ('queued', 'running')
                ORDER BY created_at
                """
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-name", default="BAAI/bge-m3")
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    config = RetrievalEvalJobConfig(
        data_dir=args.data_dir.expanduser().resolve(),
        model_name=args.model_name,
        embedding_device=args.embedding_device,
        use_fp16=True,
        working_dir=Path.cwd().resolve(),
    )
    manager = RetrievalEvalJobManager(config)
    database_path = config.data_dir / "system" / "personaforge.sqlite3"
    manager.start()
    try:
        while True:
            active = _active_jobs(database_path)
            if not active:
                return 0
            time.sleep(max(args.poll_seconds, 1.0))
    finally:
        manager.stop()


if __name__ == "__main__":
    raise SystemExit(main())
