"""Self-containment verification for dist/OverlayCat.app.

Usage: ../.venv/bin/python -B packaging/verify_bundle.py   (any cwd)

Copies the bundle to /tmp/OverlayCatTest.app, launches its binary from
cwd=/ with no arguments, waits 8 s, then asserts — loudly — that:
  1. the process is still alive;
  2. it owns a CGWindow at layer 25 (the overlay panel level);
  3. RSS is sane (< 400 MB);
  4. it holds no open files under ~/Documents (clips come from the bundle,
     state from ~/Library/Application Support/OverlayCat/).
The test instance is SIGKILLed afterwards so it never writes shared state.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGNullWindowID,
    kCGWindowListOptionAll,
)

sys.dont_write_bytecode = True

PANEL_LAYER = 25
RSS_LIMIT_MB = 400.0
WAIT_S = 8.0
TEST_APP = Path("/tmp/OverlayCatTest.app")


def main() -> None:
    src = Path(__file__).resolve().parent.parent / "dist" / "OverlayCat.app"
    if not src.is_dir():
        raise FileNotFoundError(f"bundle missing: {src}")
    if TEST_APP.exists():
        shutil.rmtree(TEST_APP)
    shutil.copytree(src, TEST_APP, symlinks=True)
    binary = TEST_APP / "Contents" / "MacOS" / "OverlayCat"
    proc = subprocess.Popen([str(binary)], cwd="/")
    time.sleep(WAIT_S)
    try:
        if proc.poll() is not None:
            raise RuntimeError(
                f"bundle process exited early: returncode={proc.returncode}")
        windows = CGWindowListCopyWindowInfo(
            kCGWindowListOptionAll, kCGNullWindowID)
        owned = [w for w in windows if w.get("kCGWindowOwnerPID") == proc.pid]
        layer25 = [w for w in owned
                   if w.get("kCGWindowLayer") == PANEL_LAYER]
        if not layer25:
            raise RuntimeError(
                f"pid {proc.pid} owns no CGWindow at layer {PANEL_LAYER}; "
                f"owned layers: {[w.get('kCGWindowLayer') for w in owned]}")
        rss_out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(proc.pid)],
            capture_output=True, text=True, check=True).stdout.strip()
        rss_mb = int(rss_out) / 1024.0
        if rss_mb >= RSS_LIMIT_MB:
            raise RuntimeError(f"RSS {rss_mb:.1f} MB >= {RSS_LIMIT_MB} MB")
        lsof = subprocess.run(
            ["lsof", "-p", str(proc.pid)], capture_output=True, text=True)
        if not lsof.stdout.strip():
            raise RuntimeError(f"lsof produced no output: {lsof.stderr}")
        docs = str(Path.home() / "Documents")
        docs_open = [line for line in lsof.stdout.splitlines() if docs in line]
        if docs_open:
            raise RuntimeError(
                "bundle process touches ~/Documents:\n" + "\n".join(docs_open))
    finally:
        proc.kill()
        proc.wait()
    bounds = layer25[0].get("kCGWindowBounds")
    print(f"self-containment ok: pid alive after {WAIT_S:.0f}s, "
          f"{len(layer25)} CGWindow(s) at layer {PANEL_LAYER} "
          f"(bounds={dict(bounds) if bounds else None}), "
          f"RSS {rss_mb:.1f} MB, no ~/Documents file handles")


if __name__ == "__main__":
    main()
