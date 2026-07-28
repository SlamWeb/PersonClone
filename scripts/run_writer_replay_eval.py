"""CLI wrapper for a frozen-context writer replay experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from personaforge.eval.replay import WriterReplayConfig, run_writer_replay
from personaforge.llm import DeepSeekJsonClient


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay only Writer using frozen query background and parent contexts."
    )
    parser.add_argument("--source-runs", type=Path, required=True)
    parser.add_argument("--parent-store", type=Path, required=True)
    parser.add_argument("--persona-pack", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    llm = DeepSeekJsonClient.from_env(args.env_file)
    llm.model = args.model
    result = run_writer_replay(
        WriterReplayConfig(
            source_runs_path=args.source_runs,
            parent_store_path=args.parent_store,
            persona_pack_path=args.persona_pack,
            run_name=args.run_name,
            out_dir=args.out_dir,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            limit=args.limit,
        ),
        llm=llm,
    )
    print(result.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
