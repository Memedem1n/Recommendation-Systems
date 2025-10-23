import os
import sys
import time
import subprocess
from pathlib import Path

SCRIPT_NAMES = [
    "Serums_&_Essences.py",
]
PYTHON_INTERPRETER = sys.executable or "python"
LOG_DIR = Path("parallel_logs")
PROGRESS_INTERVAL = 300  # seconds


def launch_process(script_path: Path) -> subprocess.Popen:
    log_file = LOG_DIR / f"{script_path.stem}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[SPAWN] {script_path.name} -> {log_file}")
    fh = open(log_file, "w", encoding="utf-8", errors="ignore")
    proc = subprocess.Popen(
        [PYTHON_INTERPRETER, str(script_path)],
        stdout=fh,
        stderr=subprocess.STDOUT,
    )
    return proc


def main():
    here = Path(__file__).parent
    scripts = [here / name for name in SCRIPT_NAMES]
    missing = [s for s in scripts if not s.exists()]
    if missing:
        print("[ERROR] Missing scripts:")
        for m in missing:
            print(f" - {m}")
        sys.exit(1)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    procs = {}
    for script in scripts:
        procs[script] = launch_process(script)

    try:
        while procs:
            time.sleep(PROGRESS_INTERVAL)
            for script, proc in list(procs.items()):
                ret = proc.poll()
                if ret is None:
                    print(f"[ALIVE] {script.name} still running...")
                else:
                    print(f"[DONE] {script.name} exited with code {ret}")
                    procs.pop(script)
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received. Terminating children...")
        for proc in procs.values():
            proc.terminate()
    finally:
        for proc in procs.values():
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    main()
