# run_all.py

import subprocess
import sys
import os


def run_precompute():
    """
    Execute Stage 1–4: precomputation pipeline.

    This function launches the intermediate data generation script.

    Behavior:
        - Uses the current Python interpreter
        - Runs `run_intermdata.py` as a subprocess
        - Raises an exception if the process fails

    Raises:
        subprocess.CalledProcessError: if the subprocess exits with non-zero status
    """
    print("\n===== Stage 1-4: Precompute =====\n")

    cmd = [sys.executable, "run_intermdata.py"]

    subprocess.run(cmd, check=True)


def run_simulation():
    """
    Execute Stage 5: simulation pipeline.

    This function runs the batch simulation script.

    Behavior:
        - Resolves the script path under pipeline/stage5_sim/
        - Uses the current Python interpreter
        - Executes batch_run.py as a subprocess
        - Raises an exception if the process fails

    Raises:
        subprocess.CalledProcessError: if the subprocess exits with non-zero status
    """
    print("\n===== Stage 5: Simulation =====\n")

    script_path = os.path.join(
        "pipeline",
        "stage5_sim",
        "batch_run.py"
    )

    cmd = [sys.executable, script_path]

    subprocess.run(cmd, check=True)


def main():
    """
    Entry point for the full pipeline.

    Pipeline stages:
        1. Precompute (Stage 1–4)
        2. Simulation (Stage 5)

    Notes:
    you can use # to control which part you want
    set configures in config.py
    """
    run_precompute()
    run_simulation()


if __name__ == "__main__":
    main()