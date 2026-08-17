"""Watchdog: tee + 90s poll + kill after 180s with no new log bytes.

Usage:
    PYTHONUNBUFFERED=1 python -m gating_ladder.watchdog --name fase1_a0 -- \\
        python -m gating_ladder.fase1
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


STALL_SEC = 180
POLL_SEC = 90


def _now() -> str:
    return time.strftime("%H:%M:%S")


def run_watched(name: str, argv: list[str], logs_dir: Path) -> int:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}_{int(time.time())}.log"
    print(f"[watchdog {_now()}] name={name} log={log_path} cmd={argv}", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", buffering=1) as logf:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=0,
        )
        assert proc.stdout is not None
        last_byte_t = time.time()
        last_poll = time.time()
        last_size = 0
        fd = proc.stdout.fileno()
        os.set_blocking(fd, False)
        while True:
            try:
                chunk = os.read(fd, 65536)
            except BlockingIOError:
                chunk = b""
            if chunk:
                text = chunk.decode("utf-8", errors="replace")
                sys.stdout.write(text)
                sys.stdout.flush()
                logf.write(text)
                logf.flush()
                last_byte_t = time.time()
                last_size = log_path.stat().st_size
            rc = proc.poll()
            if rc is not None:
                # drain
                while True:
                    try:
                        rest = os.read(fd, 65536)
                    except BlockingIOError:
                        rest = b""
                    if not rest:
                        break
                    text = rest.decode("utf-8", errors="replace")
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    logf.write(text)
                print(f"[watchdog {_now()}] exit={rc} log={log_path}", flush=True)
                return int(rc)
            now = time.time()
            stall = now - last_byte_t
            if now - last_poll >= POLL_SEC:
                print(
                    f"[watchdog {_now()}] poll size={last_size} stall_s={stall:.0f} "
                    f"pid={proc.pid} state=running",
                    flush=True,
                )
                last_poll = now
            if stall >= STALL_SEC:
                print(
                    f"[watchdog {_now()}] KILL stall={stall:.0f}s no new log lines. "
                    f"log={log_path} pid={proc.pid} stage=FASE1 fold=unknown",
                    flush=True,
                )
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                print(
                    f"[watchdog {_now()}] killed. last_log_bytes={last_size}. "
                    f"Inspect log for last fold/stage.",
                    flush=True,
                )
                return 124
            time.sleep(0.25)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--logs-dir", default="logs")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    args = p.parse_args()
    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("watchdog: missing command", flush=True)
        return 2
    return run_watched(args.name, cmd, Path(args.logs_dir))


if __name__ == "__main__":
    raise SystemExit(main())
