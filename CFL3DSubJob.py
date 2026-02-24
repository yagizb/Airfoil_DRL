import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple

def submit_job(script_path: Path, ntasks: int) -> Optional[str]:
    """
    Submit sbatch with dynamic resources, overriding #SBATCH lines in the script.
    """
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
        text=True
    )

    if result.returncode == 0:
        print("[submit_job] Submitted successfully:")
        print(result.stdout.strip())
        job_id = result.stdout.strip().split()[-1]
        return job_id

    print("[submit_job] Submission FAILED:")
    print(result.stderr.strip())
    return None

def check_job_status(job_id):
    # Check if the job with the given JOBID is still running
    check_command = ["squeue", "--jobs", job_id]
    result = subprocess.run(check_command, capture_output=True, text=True)
    if result.returncode == 0 and job_id in result.stdout:
        return True
    return False

def wait_for_job_completion(job_id):
    # Wait for job to finish
    while True:
        if not check_job_status(job_id):
            print(f"Job ID '{job_id}' has completed.")
            break
        print(f"Job ID '{job_id}' is still running, waiting...")
        time.sleep(30)  # Check status every 60 seconds

def get_job_state(job_id):
    try:
        result = subprocess.run(
            ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        state = result.stdout.strip()
        if not state:
            # job not in queue anymore
            return None
        return state
    except subprocess.CalledProcessError:
        return None
    
def wait_until_job_running(job_id, poll_interval=5, timeout=1800):
    print(f" Waiting for job {job_id} to start...")
    start_time = time.time()
    
    while True:
        state = get_job_state(job_id)

        if state == "RUNNING":
            print(f" Job {job_id} is now RUNNING.")
            time.sleep(25)
            control_state = get_job_state(job_id)
            if control_state == "RUNNING":
                return "R"
            else :
                return "Unknown"
        elif state == "COMPLETED":
            print(f" Job {job_id} already COMPLETED.")
            return "CD"
        elif state == "FAILED":
            raise RuntimeError(f"Job {job_id} FAILED.")
        elif state == "PREEMPTED":
            print(f" Job {job_id} already COMPLETED.")
            return "PR"
        elif state == "SUSPENDED":
            print(f" Job {job_id} already SUSPENDED.")
            return "S" 
        elif state is None:
            print(f" Job {job_id} not found in queue. It may have finished quickly or failed.")
            return "Unknown"
        elif state == "PENDING":
            print(f" Job {job_id} is PENDING.")
        else:
            print(f"Job {job_id} in unexpected state: {state}")
            return "unexpected state"
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout: Job {job_id} did not start within {timeout} seconds.")
        
        time.sleep(poll_interval)
    
def kill_job(job_id):
    try:
        subprocess.run(["scancel", str(job_id)], check=True)
        print(f"Job {job_id} has been canceled.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to cancel job {job_id}. Error: {e}")

### CFL3D PART ##############################


def main_cfl3d(script_name: str, ntasks: int) -> Tuple[Optional[str], Optional[str]]:
    """
    Submit the CFL3D job using the sbatch script located next to this python file,
    overriding ntasks dynamically.
    """
    project_root = Path(__file__).resolve().parent
    script_path = project_root / script_name

    if not script_path.exists():
        print(f"[main_cfl3d] sbatch script not found: {script_path}")
        return None, None

    job_id = submit_job(script_path, ntasks)
    if not job_id:
        print("[main_cfl3d] Failed to submit job.")
        return None, None

    job_state = wait_until_job_running(job_id)
    return job_id, job_state