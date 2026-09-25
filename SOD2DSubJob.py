import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple


def submit_job(script_path: Path, ntasks: int) -> Optional[str]:
    script_path = Path(script_path).resolve()
    script_dir = script_path.parent

    print(f"[submit_job] Submitting from directory: {script_dir}")
    print(f"[submit_job] Script path: {script_path}")
    print(f"[submit_job] ntasks={ntasks}")

    cmd = [
        "sbatch",
        "--nodes=1",
        f"--ntasks={ntasks}",
        f"--ntasks-per-node={ntasks}",
        script_path.name,
    ]

    result = subprocess.run(
        cmd,
        cwd=str(script_dir),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("[submit_job] Submission FAILED:")
        print(result.stderr.strip())
        return None

    print("[submit_job] Submitted successfully:")
    print(result.stdout.strip())

    # Usually: "Submitted batch job 41056854"
    try:
        return result.stdout.strip().split()[-1]
    except Exception:
        return None


def get_job_state(job_id: str) -> Optional[str]:
    """
    Return Slurm state from squeue.
    Returns None if the job is no longer visible in squeue.
    """
    try:
        result = subprocess.run(
            ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        state = result.stdout.strip()
        if not state:
            return None

        # In rare cases squeue can return multiple lines
        return state.splitlines()[0].strip()

    except Exception as e:
        print(f"[get_job_state] Could not read state for job {job_id}: {e}")
        return None


def wait_until_job_starts(
    job_id: str,
    poll_interval: float = 5.0,
    timeout: float = 7200.0,
) -> str:
    """
    Wait until the job reaches RUNNING.

    Important:
    - A valid job_id means sbatch succeeded.
    - If the job disappears from squeue quickly, return UNKNOWN.
      Do not treat this as submission failure.
    """
    print(f"[wait_until_job_starts] Waiting for job {job_id} to start...")
    start_time = time.time()

    while True:
        state = get_job_state(job_id)

        if state == "RUNNING":
            print(f"[wait_until_job_starts] Job {job_id} is RUNNING.")
            return "R"

        if state == "PENDING":
            print(f"[wait_until_job_starts] Job {job_id} is PENDING.")

        elif state is None:
            print(
                f"[wait_until_job_starts] Job {job_id} is not in squeue. "
                "It may have finished quickly or failed after starting."
            )
            return "UNKNOWN"

        elif state in (
            "COMPLETED",
            "COMPLETING",
        ):
            print(f"[wait_until_job_starts] Job {job_id} already {state}.")
            return "CD"

        elif state in (
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
        ):
            print(f"[wait_until_job_starts] Job {job_id} failed with state={state}.")
            return state

        elif state == "PREEMPTED":
            print(f"[wait_until_job_starts] Job {job_id} PREEMPTED.")
            return "PR"

        elif state == "SUSPENDED":
            print(f"[wait_until_job_starts] Job {job_id} SUSPENDED.")
            return "S"

        else:
            print(f"[wait_until_job_starts] Job {job_id} unexpected state={state}.")
            return state

        if time.time() - start_time > timeout:
            raise TimeoutError(
                f"Timeout: Job {job_id} did not start within {timeout} seconds."
            )

        time.sleep(poll_interval)


def check_job_status(job_id: str) -> bool:
    """
    True if job is still visible in squeue.
    False if not visible.
    """
    result = subprocess.run(
        ["squeue", "--jobs", str(job_id)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and str(job_id) in result.stdout


def wait_for_job_completion(
    job_id: str,
    poll_interval: float = 30.0,
) -> None:
    while True:
        if not check_job_status(job_id):
            print(f"[wait_for_job_completion] Job {job_id} has completed or left squeue.")
            break

        print(f"[wait_for_job_completion] Job {job_id} still active...")
        time.sleep(poll_interval)


def kill_job(job_id: str) -> None:
    if not job_id:
        return

    try:
        subprocess.run(["scancel", str(job_id)], check=True)
        print(f"[kill_job] Job {job_id} has been canceled.")
    except subprocess.CalledProcessError as e:
        print(f"[kill_job] Failed to cancel job {job_id}: {e}")


def main_sod2d(script_name: str, ntasks: int) -> Tuple[Optional[str], Optional[str]]:
    """
    Submit SOD2D job using the sbatch script located next to this Python file.
    """
    project_root = Path(__file__).resolve().parent
    script_path = project_root / script_name

    if not script_path.exists():
        print(f"[main_sod2d] sbatch script not found: {script_path}")
        return None, None

    job_id = submit_job(script_path, ntasks)

    if not job_id:
        print("[main_sod2d] Failed to submit job.")
        return None, None

    job_state = wait_until_job_starts(job_id)

    return job_id, job_state
