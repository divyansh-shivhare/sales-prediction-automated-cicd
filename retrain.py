#!/usr/bin/env python3
"""
Automated retrain script (improved)

Usage:
  # one-shot run
  python retrain.py

  # run as a watcher (poll every 300 seconds)
  python retrain.py --watch --interval 300
"""
from __future__ import annotations
import os
import sys
import time
import shutil
import subprocess
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# ----- CONFIG -----
DATA_PATH = Path("data/add.csv")
META_PATH = Path("data/last_retrain.txt")
MODELS_DIR = Path("models")
MODEL_SRC = Path("model.pkl")    # training script should write this
TRAIN_SCRIPT = Path("train_model.py")
POLL_INTERVAL = 300              # seconds, used when --watch
TRAIN_TIMEOUT = 60 * 60          # seconds, timeout for training process
LOCKFILE = Path(".retrain.lock")
# -------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("retrain")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)


def file_md5(path: Path) -> str:
    import hashlib
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def read_meta() -> Optional[str]:
    if not META_PATH.exists():
        return None
    return META_PATH.read_text().strip()


def write_meta_atomic(md5: str) -> None:
    # atomic write via temporary file + replace
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(META_PATH.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(md5)
        os.replace(tmp, META_PATH)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def acquire_lock() -> bool:
    """
    Try to create a lock file atomically. Return True if lock acquired.
    Simple mechanism to prevent concurrent runs.
    """
    try:
        # O_EXCL ensures failure if file exists.
        fd = os.open(str(LOCKFILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"{os.getpid()}\n")
        return True
    except FileExistsError:
        return False
    except Exception as e:
        logger.warning("Could not acquire lock due to unexpected error: %s", e)
        return False


def release_lock() -> None:
    try:
        if LOCKFILE.exists():
            LOCKFILE.unlink()
    except Exception as e:
        logger.debug("Failed to remove lockfile: %s", e)


def run_training_script(timeout: int = TRAIN_TIMEOUT) -> subprocess.CompletedProcess:
    """
    Run training script in a subprocess. Enforce UTF-8 for stdout/stderr to avoid encoding issues.
    """
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Training script not found: {TRAIN_SCRIPT}")

    env = os.environ.copy()
    # Ensure python uses utf-8 for IO to prevent UnicodeEncodeError on Windows consoles.
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [sys.executable, str(TRAIN_SCRIPT)]

    logger.info("Launching training: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=timeout,
    )
    return proc


def save_model_version(dst_dir: Path, src: Path) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"Expected model file not found after training: {src}")

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dst = dst_dir / f"model_{ts}.pkl"
    shutil.copy2(src, dst)

    # update latest link/copy
    latest = dst_dir / "latest_model.pkl"
    try:
        # remove if exists
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        # attempt to create relative symlink if possible
        try:
            os.symlink(dst.name, latest)
        except (AttributeError, NotImplementedError, OSError):
            # fallback to copy for Windows / non-symlink-friendly FS
            shutil.copy2(dst, latest)
    except Exception as e:
        logger.warning("Failed to update latest model pointer: %s", e)

    return dst


def retrain_once() -> bool:
    """
    Perform a single check-and-retrain cycle.
    Returns True if retraining happened, False otherwise.
    """
    if not DATA_PATH.exists():
        logger.warning("No data file found at %s", DATA_PATH)
        return False

    try:
        current_md5 = file_md5(DATA_PATH)
    except Exception as e:
        logger.error("Failed to compute checksum of %s: %s", DATA_PATH, e)
        return False

    last_md5 = read_meta()
    if current_md5 == last_md5:
        logger.info("No data change detected. Retrain not required.")
        return False

    logger.info("Data change detected. Starting retrain...")

    try:
        proc = run_training_script()
    except subprocess.TimeoutExpired:
        logger.error("Training timed out after %s seconds", TRAIN_TIMEOUT)
        return False
    except Exception as e:
        logger.exception("Failed to launch training: %s", e)
        return False

    # log outputs
    if proc.stdout:
        logger.info("Training stdout:\n%s", proc.stdout.strip())
    if proc.stderr:
        logger.error("Training stderr:\n%s", proc.stderr.strip())

    if proc.returncode != 0:
        logger.error("Training failed with exit code %d", proc.returncode)
        return False

    # ensure model exists and save version
    try:
        dst = save_model_version(MODELS_DIR, MODEL_SRC)
    except Exception as e:
        logger.exception("Failed to save model: %s", e)
        return False

    # update metadata
    try:
        write_meta_atomic(current_md5)
    except Exception as e:
        logger.exception("Failed to write metadata: %s", e)
        # still return True because model saved successfully
    logger.info("Retrain complete. Model saved to %s", dst)
    return True


def watch_loop(interval: int = POLL_INTERVAL) -> None:
    logger.info("Starting watch mode (poll every %d seconds). Ctrl-C to stop.", interval)
    try:
        while True:
            if acquire_lock():
                try:
                    retrain_once()
                finally:
                    release_lock()
            else:
                logger.info("Another retrain process is active; skipping this cycle.")
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Watch mode stopped by user.")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Automated retrain script")
    parser.add_argument("--watch", action="store_true", help="Run in watch/polling mode")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL, help="Polling interval seconds (when --watch)")
    parser.add_argument("--once", action="store_true", help="Force a retrain attempt regardless of checksum (useful for testing)")
    args = parser.parse_args(argv)

    if args.watch:
        watch_loop(interval=args.interval)
        return

    # single run
    if not acquire_lock():
        logger.error("Another retrain is already running (lockfile present). Exiting.")
        return

    try:
        if args.once:
            logger.info("Forced run (--once): will retrain regardless of checksum.")
            # bypass checksum: temporarily write a different last_retrain to force retrain
            write_meta_atomic("__FORCE__")
            retrain_once()
        else:
            retrain_once()
    finally:
        release_lock()


if __name__ == "__main__":
    main()
