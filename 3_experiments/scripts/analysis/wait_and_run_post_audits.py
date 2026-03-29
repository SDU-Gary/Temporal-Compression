#!/usr/bin/env python3
"""Wait for a training run to finish, then execute post-training audits.

This is mainly for attaching new diagnostics to an already-running training job
that started before the diagnostics were integrated into train.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Wait for run completion and execute audits")
    p.add_argument("--run-dir", required=True, help="Experiment output directory")
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--max-samples", type=int, default=8192)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--max-wait-seconds", type=int, default=0, help="0 means no timeout")
    p.add_argument("--skip-wait", action="store_true")
    p.add_argument(
        "--profiles-json",
        default="",
        help="JSON list or {'profiles': [...]} for routing profiles",
    )
    return p


def _default_profiles(top_k: int) -> List[Dict[str, Any]]:
    return [
        {
            "name": f"hard{int(top_k)}",
            "top_k": int(top_k),
            "training_soft_routing": False,
        },
        {
            "name": "soft8_t020",
            "top_k": int(top_k),
            "training_soft_routing": True,
            "routing_soft_topk": 8,
            "routing_temperature": 0.20,
        },
    ]


def _resolve_profiles(args: argparse.Namespace) -> Dict[str, Any]:
    raw = str(args.profiles_json).strip()
    if not raw:
        return {"profiles": _default_profiles(int(args.top_k))}
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        return {"profiles": parsed}
    if isinstance(parsed, dict):
        return parsed
    raise ValueError("profiles-json must decode to list or mapping")


def _read_heartbeat(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _wait_for_finish(args: argparse.Namespace, heartbeat_path: Path) -> Dict[str, Any]:
    if args.skip_wait:
        return _read_heartbeat(heartbeat_path)

    started = time.time()
    poll = max(1, int(args.poll_seconds))
    while True:
        hb = _read_heartbeat(heartbeat_path)
        state = str(hb.get("state", "")).strip().lower()
        epoch = hb.get("epoch")
        total = hb.get("num_epochs")
        now = datetime.now().strftime("%F %T")
        if state:
            print(f"[{now}] heartbeat: state={state} epoch={epoch}/{total}")
        else:
            print(f"[{now}] heartbeat: waiting for file {heartbeat_path}")

        if state == "finished":
            return hb
        if state in {"failed", "error", "aborted"}:
            return hb

        if int(args.max_wait_seconds) > 0 and (time.time() - started) > int(args.max_wait_seconds):
            raise TimeoutError(f"wait timeout exceeded: {args.max_wait_seconds}s")
        time.sleep(poll)


def _run_cmd(cmd: List[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    heartbeat_path = run_dir / "runtime" / "heartbeat.json"
    hb = _wait_for_finish(args, heartbeat_path)
    state = str(hb.get("state", "")).strip().lower()
    if state and state != "finished":
        print(f"Warning: run ended with non-finished state={state}, will still try audits")

    best_ckpt = run_dir / "best_model.pt"
    last_ckpt = run_dir / "last_model.pt"
    if not best_ckpt.exists() and not last_ckpt.exists():
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")

    ckpt_for_util = best_ckpt if best_ckpt.exists() else last_ckpt
    profiles_payload = _resolve_profiles(args)
    profiles_json = json.dumps(profiles_payload, ensure_ascii=False)

    script_dir = Path(__file__).resolve().parent
    expert_script = script_dir / "run_expert_utilization_audit.py"
    drift_script = script_dir / "run_semantic_drift_audit.py"

    diagnostics_dir = run_dir / "diagnostics"
    util_out = diagnostics_dir / "expert_utilization" / "posttrain_summary.json"
    drift_out = diagnostics_dir / "semantic_drift" / "posttrain_best_vs_last.json"
    util_out.parent.mkdir(parents=True, exist_ok=True)
    drift_out.parent.mkdir(parents=True, exist_ok=True)

    util_cmd = [
        sys.executable,
        str(expert_script),
        "--data-root",
        str(args.data_root),
        "--checkpoint",
        str(ckpt_for_util),
        "--output",
        str(util_out),
        "--split",
        str(args.split),
        "--device",
        str(args.device),
        "--batch-size",
        str(int(args.batch_size)),
        "--num-workers",
        str(int(args.num_workers)),
        "--seed",
        str(int(args.seed)),
        "--top-k",
        str(int(args.top_k)),
        "--max-samples",
        str(int(args.max_samples)),
        "--sample-seed",
        str(int(args.sample_seed)),
        "--profiles-json",
        profiles_json,
    ]
    _run_cmd(util_cmd)

    drift_ran = False
    if best_ckpt.exists() and last_ckpt.exists():
        drift_cmd = [
            sys.executable,
            str(drift_script),
            "--data-root",
            str(args.data_root),
            "--checkpoint-a",
            str(best_ckpt),
            "--checkpoint-b",
            str(last_ckpt),
            "--output",
            str(drift_out),
            "--name",
            "best_vs_last",
            "--split",
            str(args.split),
            "--device",
            str(args.device),
            "--batch-size",
            str(int(args.batch_size)),
            "--num-workers",
            str(int(args.num_workers)),
            "--seed",
            str(int(args.seed)),
            "--top-k",
            str(int(args.top_k)),
            "--max-samples",
            str(int(args.max_samples)),
            "--sample-seed",
            str(int(args.sample_seed)),
            "--profiles-json",
            profiles_json,
        ]
        _run_cmd(drift_cmd)
        drift_ran = True
    else:
        print("Warning: missing best_model.pt or last_model.pt, semantic drift audit skipped")

    summary = {
        "meta": {
            "run_dir": str(run_dir),
            "state_at_attach": state,
            "heartbeat_path": str(heartbeat_path),
        },
        "outputs": {
            "expert_utilization_summary": str(util_out),
            "semantic_drift_summary": str(drift_out) if drift_ran else None,
        },
        "checkpoint_used": {
            "expert_utilization": str(ckpt_for_util),
            "best_model": str(best_ckpt) if best_ckpt.exists() else None,
            "last_model": str(last_ckpt) if last_ckpt.exists() else None,
        },
    }
    summary_path = diagnostics_dir / "posttrain_audits_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved post-training audits summary: {summary_path}")


if __name__ == "__main__":
    main()

