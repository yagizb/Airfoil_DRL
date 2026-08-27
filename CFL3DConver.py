import time
from typing import Tuple, Optional
from pathlib import Path

def count_cfl3d_ready_failed(shared_root: Path, n_envs: int) -> Tuple[int, int]:
    """
    Count how many env_i have:
      - cfl3d_ready.flag
      - cfl3d_failed.flag

    Returns
    -------
    (n_ready, n_failed)
    """
    n_ready = 0
    n_failed = 0

    for i in range(n_envs):
        env_dir = shared_root / f"env_{i}"
        if not env_dir.exists():
            continue

        ready_flag = env_dir / "cfl3d_ready.flag"
        failed_flag = env_dir / "cfl3d_failed.flag"

        # IMPORTANT: failed overrides ready
        if failed_flag.exists():
            n_failed += 1
        elif ready_flag.exists():
            n_ready += 1

    return n_ready, n_failed

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
        cur = count_cfl3d_ready_failed(shared_root, n_envs)
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
        

def plot3d_files_ready(work_dir):
    work_dir = Path(work_dir)

    plot3dg = work_dir / "plot3dg.bin"
    plot3dq = work_dir / "plot3dq.bin"

    return (
        plot3dg.exists()
        and plot3dq.exists()
        and plot3dg.stat().st_size > 0
        and plot3dq.stat().st_size > 0
    )


def follow_cfl3d_output(
    file_path,
    sleep_time,
    threshold,
    max_iter,
    fidelity,
    conv_flag_path=None,
):
    """
    Monitor a CFL3D output file.

    FIDELITY = 1:
        Check residual convergence.

    FIDELITY = 2:
        Ignore residual convergence.
        Wait until max_iter is reached and plot3dg.bin / plot3dq.bin exist.

    Flag values:
        CONVERGED
        RANS_INIT_DONE
        MAX_ITER
        INVALID_MESH
    """

    file_path = Path(file_path)
    work_dir = file_path.parent

    print(f"Monitoring '{file_path}' every {sleep_time}s...\n")

    while True:
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()

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

                    # -------------------------
                    # Invalid mesh sentinel
                    # -------------------------
                    if total_res == 1.00e0 and lift == 1.0:
                        print(f"Mesh is invalid, Res={total_res}, Lift={lift}")

                        if conv_flag_path is not None:
                            Path(conv_flag_path).write_text("INVALID_MESH\n")

                        return 0.0, 1.0e-8

                    # -------------------------
                    # FIDELITY = 1
                    # Normal RANS convergence
                    # -------------------------
                    if fidelity == 1:
                        if total_res < threshold:
                            print("CFL3D convergence achieved!")

                            if conv_flag_path is not None:
                                Path(conv_flag_path).write_text("CONVERGED\n")

                            return lift, drag

                        if iteration >= max_iter:
                            print(f"Max iteration limit ({max_iter}) reached.")

                            if conv_flag_path is not None:
                                Path(conv_flag_path).write_text("MAX_ITER\n")

                            return lift, drag

                        # --- mesh invalid sentinel ---
                    if total_res == 1.00e0 and lift == 1.0:
                        print(f"Mesh is invalid, Res: {total_res}, Lift: {lift}")
                        if conv_flag_path is not None:
                            Path(conv_flag_path).write_text("INVALID_MESH\n")
                        return 0.0, 1.0e-8  # fallback values
                    # -------------------------
                    # FIDELITY = 2
                    # RANS only for LES initialization
                    # -------------------------
                    elif fidelity == 2:
                        if iteration >= max_iter:
                            if plot3d_files_ready(work_dir):
                                print(
                                    "RANS initialization finished. "
                                    "plot3dg.bin and plot3dq.bin are ready."
                                )

                                if conv_flag_path is not None:
                                    Path(conv_flag_path).write_text("RANS_INIT_DONE\n")

                                return lift, drag

                            else:
                                print(
                                    f"Reached iteration {iteration}, "
                                    "waiting for plot3dg.bin / plot3dq.bin..."
                                )

                    else:
                        raise ValueError(f"Unsupported fidelity value: {fidelity}")

                    break

        except FileNotFoundError:
            print("Waiting for cfl3d.out to be created...")

        except Exception as e:
            print(f"Error reading output: {e}")

        time.sleep(sleep_time)