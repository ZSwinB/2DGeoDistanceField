# run_all.py

import subprocess
import sys
import os


def run_precompute():
    print("\n===== Stage 1-4: Precompute =====\n")

    cmd = [sys.executable, "run_intermdata.py"]

    subprocess.run(cmd, check=True)


def run_simulation():
    print("\n===== Stage 5: Simulation =====\n")

    script_path = os.path.join(
        "pipeline",
        "stage5_sim",
        "batch_run.py"
    )

    cmd = [sys.executable, script_path]

    subprocess.run(cmd, check=True)


def main():

    run_precompute()
    run_simulation()


if __name__ == "__main__":
    main()