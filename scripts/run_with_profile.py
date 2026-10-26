"""Run a command with an experimental model profile applied as environment overrides.

Profiles live in a versioned JSON file (``common`` overrides, named ``models``
override groups, and ``profiles`` that combine groups). Before running, every
AVA and VidOR checkpoint named by the profile must validate against its
metadata, model version, and checksum; otherwise nothing runs. The production
``.env`` and code defaults are never modified.

    python -m scripts.run_with_profile --profile det-ava8-vidor4 \
        --set DATABASE_PATH=data/app/benchmark/x/x.sqlite3 \
        -- python -m scripts.index_videos_batch ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_PROFILES = Path("evaluation/configs/model_profiles_2026-10-26.json")


def resolve_profile(payload: dict[str, Any], name: str) -> dict[str, str]:
    if name not in payload.get("profiles", {}):
        raise SystemExit(f"unknown profile {name!r}; known: {sorted(payload.get('profiles', {}))}")
    overrides = {key: str(value) for key, value in payload.get("common", {}).items()}
    for group in payload["profiles"][name]:
        if group not in payload.get("models", {}):
            raise SystemExit(f"profile {name!r} names unknown model group {group!r}")
        overrides.update({key: str(value) for key, value in payload["models"][group].items()})
    return overrides


def validate_checkpoints(overrides: dict[str, str]) -> dict[str, str]:
    """Status of each configured checkpoint; raises if any is not usable."""

    from vidquery.ava_gnn80 import ava80_gnn_status
    from vidquery.vidor_improved import configured_relation_checkpoint_status

    statuses = {}
    if "GNN_CHECKPOINT" in overrides:
        statuses["ava_action_gnn"] = ava80_gnn_status(
            Path(overrides["GNN_CHECKPOINT"]),
            Path(overrides["GNN_METADATA"]),
            overrides.get("GNN_MODEL_VERSION"),
        )
    if "RELATION_GNN_CHECKPOINT" in overrides:
        statuses["vidor_relation_gnn"] = configured_relation_checkpoint_status(
            Path(overrides["RELATION_GNN_CHECKPOINT"]),
            Path(overrides["RELATION_GNN_METADATA"]),
            overrides.get("RELATION_GNN_MODEL_VERSION"),
        )
    failed = {name: status for name, status in statuses.items() if status != "available_validated"}
    if failed:
        raise SystemExit(f"profile checkpoints are not usable: {failed}")
    return statuses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--set", action="append", default=[], help="extra KEY=VALUE override")
    parser.add_argument("--record", type=Path, default=None, help="write the effective overrides")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("give the command to run after --")
    overrides = resolve_profile(json.loads(args.profiles.read_text(encoding="utf-8")), args.profile)
    for item in args.set:
        key, _, value = item.partition("=")
        overrides[key] = value
    statuses = validate_checkpoints(overrides)
    record = {"profile": args.profile, "profiles_file": args.profiles.as_posix(),
              "overrides": overrides, "checkpoint_status": statuses}
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record), file=sys.stderr, flush=True)
    return subprocess.call(command, env={**os.environ, **overrides})


if __name__ == "__main__":
    raise SystemExit(main())
