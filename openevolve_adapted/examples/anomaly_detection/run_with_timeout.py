"""Run a command until it exits or its wall-clock deadline is reached."""

import argparse
import os
import signal
import subprocess
import sys
from typing import List


SHUTDOWN_GRACE_SECONDS = 20


def run_with_timeout(command: List[str], timeout_seconds: int) -> int:
    """Request graceful shutdown at the deadline, then kill a stuck process group."""
    process = subprocess.Popen(command, start_new_session=(os.name == "posix"))
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        print(
            f"Maximum runtime of {timeout_seconds}s reached; requesting graceful shutdown.",
            flush=True,
        )
        process.terminate()
        try:
            return process.wait(timeout=SHUTDOWN_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            print("Grace period expired; terminating the run process group.", flush=True)
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
            return 124


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timeout_seconds", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.timeout_seconds < 1 or not command:
        parser.error("timeout_seconds must be positive and a command is required after --")
    return run_with_timeout(command, args.timeout_seconds)


if __name__ == "__main__":
    sys.exit(main())