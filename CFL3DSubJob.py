import subprocess
import time
from pathlib import Path

def submit_job(script_name):
    # Absolute path to script
    script_path = Path(script_name).resolve()
    script_dir = script_path.parent

    print(f"Submitting job from directory: {script_dir}")
    print(f"Script path: {script_path}")

    # Always run sbatch inside the folder where the script lives
    result = subprocess.run(
        ["sbatch", script_path.name],
        cwd=str(script_dir),
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f"Job {script_path.name} submitted successfully:")
        print(result.stdout)
        job_id = result.stdout.strip().split()[-1]
        return job_id
    else:
        print(f"Error submitting job {script_path.name}:")
        print(result.stderr)
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

def main_cfl3d(script_name):
    """
    Submit the CFL3D job using the main-folder sbatch script.

    We compute the project root as the directory of THIS Python file
    (where this code lives), and then join script_name to it.
    """
    project_root = Path(__file__).resolve().parent
    script_path = project_root / script_name

    if not script_path.exists():
        print(f"sbatch script not found: {script_path}")
        return None, None

    job_id = submit_job(script_path)

    if not job_id:
        print("Failed to submit job.")
        return None, None

    job_state = wait_until_job_running(job_id)
    return job_id, job_state