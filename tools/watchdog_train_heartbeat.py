#!/usr/bin/env python3
"""Watchdog for long-running training jobs based on heartbeat status files.

Default behavior is conservative: warn and exit when heartbeat is stale.
This avoids auto-restart side effects while still allowing tmux/shell wrappers
to detect failures quickly.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _parse_iso_timestamp(text: str) -> datetime:
    value = str(text).strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def _load_heartbeat(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("heartbeat payload must be a JSON object")
    return payload


def _staleness_seconds(payload: Dict[str, Any], now: float) -> float:
    ts_raw = payload.get("timestamp")
    if ts_raw is None:
        return float("inf")
    try:
        ts = _parse_iso_timestamp(str(ts_raw))
        return max(0.0, now - ts.timestamp())
    except Exception:
        return float("inf")


def _status_line(payload: Dict[str, Any], stale_seconds: float) -> str:
    state = payload.get("state", "unknown")
    event = payload.get("last_event", "unknown")
    epoch = payload.get("epoch", "?")
    num_epochs = payload.get("num_epochs", "?")
    stage_name = payload.get("stage_name", "?")
    return (
        f"state={state} event={event} stage={stage_name} "
        f"epoch={epoch}/{num_epochs} stale={stale_seconds:.1f}s"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch heartbeat.json and detect stalled training.")
    parser.add_argument("--heartbeat", type=str, required=True, help="Path to runtime/heartbeat.json")
    parser.add_argument(
        "--stale-seconds",
        type=float,
        default=1800.0,
        help="Staleness threshold in seconds before reporting failure.",
    )
    parser.add_argument(
        "--check-interval-seconds",
        type=float,
        default=60.0,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--startup-grace-seconds",
        type=float,
        default=900.0,
        help="Allow heartbeat to appear within this grace window.",
    )
    parser.add_argument(
        "--action",
        choices=["warn_exit", "warn_continue"],
        default="warn_exit",
        help="Action when stale heartbeat is detected.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Check one time and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    heartbeat_path = Path(args.heartbeat)
    stale_seconds_limit = max(1.0, float(args.stale_seconds))
    check_interval = max(0.2, float(args.check_interval_seconds))
    startup_grace = max(0.0, float(args.startup_grace_seconds))
    action = str(args.action)

    start_time = time.time()

    while True:
        now = time.time()

        if not heartbeat_path.exists():
            waited = now - start_time
            if waited > startup_grace:
                print(
                    f"[watchdog] heartbeat missing for {waited:.1f}s path={heartbeat_path}",
                    flush=True,
                )
                return 1
            if args.once:
                print(f"[watchdog] heartbeat not found yet path={heartbeat_path}", flush=True)
                return 1
            time.sleep(check_interval)
            continue

        try:
            payload = _load_heartbeat(heartbeat_path)
        except Exception as exc:
            print(f"[watchdog] failed to read heartbeat: {exc}", flush=True)
            if args.once:
                return 1
            time.sleep(check_interval)
            continue

        stale_seconds = _staleness_seconds(payload, now)
        state = str(payload.get("state", "unknown")).lower()

        if state == "finished":
            print(f"[watchdog] training finished: {_status_line(payload, stale_seconds)}", flush=True)
            return 0
        if state == "error":
            print(f"[watchdog] training error state: {_status_line(payload, stale_seconds)}", flush=True)
            return 2

        if stale_seconds > stale_seconds_limit:
            message = (
                f"[watchdog] stale heartbeat detected (limit={stale_seconds_limit:.1f}s): "
                f"{_status_line(payload, stale_seconds)}"
            )
            print(message, flush=True)
            if action == "warn_exit":
                return 1

        if args.once:
            print(f"[watchdog] ok: {_status_line(payload, stale_seconds)}", flush=True)
            return 0

        time.sleep(check_interval)


if __name__ == "__main__":
    raise SystemExit(main())

