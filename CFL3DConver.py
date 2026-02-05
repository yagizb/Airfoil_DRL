import time
from CAL_CLCD import calclcd  
from CFL3DSubJob import kill_job
from typing import Tuple, Optional
from pathlib import Path

def _count_ready(shared_root: Path, n_envs: int) -> int:
    c = 0
    for i in range(n_envs):
        if (shared_root / f"env_{i}" / "cfl3d_ready.flag").exists():
            c += 1
    return c

def wait_ready_stable(shared_root: Path, n_envs: int, stable_s: float = 5.0,
                       poll_s: float = 2.0, max_wait_s: float = 50.0, vprint=None):
    """
    Wait until number of READY envs stops changing for stable_s seconds,
    or until max_wait_s is reached.
    """
    last = -1
    stable = 0.0
    t0 = time.time()

    while True:
        cur = _count_ready(shared_root, n_envs)
        if cur == last:
            stable += poll_s
        else:
            stable = 0.0
            last = cur

        if vprint:
            vprint(f"[Leader] READY envs so far: {cur} (stable {stable:.1f}/{stable_s}s)")

        if stable >= stable_s:
            return cur

        if time.time() - t0 >= max_wait_s:
            return cur

        time.sleep(poll_s)

def checkconv(env_id, output_geo, angle_of_attack):
    """
    Monitors the convergence of the Cl (Lift Coefficient) for a given environment.
    
    Args:
        env_id (int): Environment ID.
        output_geo (str): Path or reference to geometry output.
        angle_of_attack (float): Angle of attack in degrees.

    Returns:
        dict: Contains the final Cl, Cd, and convergence status.
    """
    # Initial computation of Cl and Cd
    time.sleep(60)
    Cl, Cd = calclcd(output_geo, angle_of_attack)
    print(f"Env_ID: {env_id}, Initial Cl={Cl:.6f}, Initial Cd={Cd:.6f}")

    # Variables for convergence tracking
    Cl_old = Cl
    convergence_threshold = 0.01  # 1% threshold for convergence
    converged = False

    while not converged:
        time.sleep(60)  # Wait for 60 seconds before checking again
        
        # Recalculate Cl and Cd
        Cl, Cd = calclcd(output_geo, angle_of_attack)
        print(f"Env_ID: {env_id}, Updated Cl={Cl:.6f}, Updated Cd={Cd:.6f}")

        # Calculate relative change in Cl
        relative_change = abs((Cl - Cl_old) / Cl_old)
        print(f"Relative Change in Cl: {relative_change:.4%}")

        if relative_change < convergence_threshold:
            converged = True
            print(f"Run has converged for Env_ID: {env_id}")
        else:
            Cl_old = Cl  # Update for next iteration

    return Cl,Cd

BEGIN_INIT = "***** BEGINNING INITIALIZATION *****"
END_INIT   = "***** ENDING INITIALIZATION *****"

def cfl3d_out_init_crashed(
    work_dir: Path,
    stall_timeout: float = 10.0,
    poll: float = 2.0,
):
    """
    Returns (failed_init: bool, reason: str)

    FAIL if:
      - cfl3d.out contains BEGIN_INIT
      - cfl3d.out does NOT contain END_INIT
      - and cfl3d.out stops growing for stall_timeout seconds

    OK if:
      - END_INIT appears (init finished)
      - or BEGIN_INIT hasn't appeared yet (still starting)
      - or file is still growing (still doing something)
    """
    out_path = work_dir / "cfl3d.out"
    if not out_path.exists():
        return False, "NO_OUT"

    last_size = out_path.stat().st_size
    last_change = time.time()

    while True:
        # read safely
        txt = out_path.read_text(errors="ignore")

        # init done => OK
        if END_INIT in txt:
            return False, "INIT_OK"

        begin_seen = (BEGIN_INIT in txt)

        # growth check
        sz = out_path.stat().st_size
        if sz != last_size:
            last_size = sz
            last_change = time.time()
        else:
            # stalled
            if begin_seen and (time.time() - last_change) >= stall_timeout:
                return True, f"INIT_STALLED_NO_END (>{stall_timeout}s)"

        time.sleep(poll)


def check_cfl3d_error(work_dir: Path) -> Tuple[bool, Optional[int]]:
    """
    CFL3D failure condition:
      - FAIL  : cfl3d.error exists AND code == -1
      - OK    : file does not exist OR code != -1
    """
    err_file = work_dir / "cfl3d.error"

    # No error file → CFL3D OK
    if not err_file.exists():
        return False, None

    try:
        code = int(err_file.read_text().strip())

        # Only -1 means failure
        if code == -1:
            return True, code

        # Any other integer → OK
        return False, code

    except Exception:
        # File exists but unreadable → treat as failure (safe side)
        return False, None
        
def follow_cfl3d_output(
    file_path,
    sleep_time,
    threshold,
    max_iter,
    conv_flag_path=None,
):
    """
    Monitor a CFL3D output file until convergence / failure.

    - Writes a small flag file if conv_flag_path is given:
        'CONVERGED', 'MAX_ITER', or 'INVALID_MESH'.
    """
    print(f"Monitoring '{file_path}' every {sleep_time}s for convergence...\n")

    while True:
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()

            # Find the last data line (must start with digit)
            for line in reversed(lines):
                tokens = line.strip().split()
                if len(tokens) >= 8 and tokens[0].replace(".", "", 1).isdigit():
                    iteration = int(tokens[2])
                    total_res = float(tokens[4].replace("E", "e"))
                    lift = float(tokens[5].replace("E", "e"))
                    drag = float(tokens[6].replace("E", "e"))


                    print(
                        f"[Iter {iteration}] Total Res = {total_res:.2e}, "
                        f"Lift = {lift:.5f}, Drag = {drag:.5f}"
                    )

                    # --- convergence ---
                    if total_res < threshold:
                        print("Convergence achieved!")
                        if conv_flag_path is not None:
                            Path(conv_flag_path).write_text("CONVERGED\n")
                        return lift, drag

                    # --- max iteration ---
                    if iteration > max_iter:
                        print(f" Max iteration limit ({max_iter}) exceeded.")
                        if conv_flag_path is not None:
                            Path(conv_flag_path).write_text("MAX_ITER\n")
                        return lift, drag       ##???????????

                    # --- mesh invalid sentinel ---
                    if total_res == 1.00e0 and lift == 1.0:
                        print(f"Mesh is invalid, Res: {total_res}, Lift: {lift}")
                        if conv_flag_path is not None:
                            Path(conv_flag_path).write_text("INVALID_MESH\n")
                        return 0.0, 1.0e-8  # fallback values

                    break  # only need the latest valid line

        except FileNotFoundError:
            print(" Waiting for cfl3d.out to be created...")
        except Exception as e:
            print(f" Error reading output: {e}")

        time.sleep(sleep_time)