from CFL3DMesh import cfl3d_mesh
from CFL3DSubJob import main_cfl3d, kill_job
from CFL3DConver import follow_cfl3d_output, check_cfl3d_error, cfl3d_out_init_crashed,count_ready_failed
from CFL3DPrep import update_cfl3d_inp, copy_main_inputs, clean_env

import time
import os
import subprocess
from pathlib import Path
import config
from typing import Optional

def _safe_unlink(p: Path, vprint=None):
    try:
        p.unlink()
    except FileNotFoundError:
        return
    except Exception as e:
        if vprint:
            vprint(f"Could not remove {p}: {e}")

def _read_leader_env_id(leader_flag: Path) -> Optional[int]:
    try:
        return int(leader_flag.read_text().strip())
    except Exception:
        return None

def _try_become_leader(leader_flag: Path, env_id: int, vprint=None) -> bool:
    # only succeeds if leader_flag does not exist
    try:
        fd = os.open(leader_flag, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(env_id).encode())
        os.close(fd)
        if vprint:
            vprint(f"[Env {env_id}] Took leadership.")
        return True
    except FileExistsError:
        return False

def _force_replace_leader(leader_flag: Path, env_id: int, vprint=None):
    # best-effort replace (race-safe enough for this use)
    try:
        leader_flag.write_text(str(env_id) + "\n")
        if vprint:
            vprint(f"[Env {env_id}] Replaced leader (forced).")
    except Exception as e:
        if vprint:
            vprint(f"[Env {env_id}] Failed to replace leader: {e}")

def _leader_is_failed(shared_root: Path, leader_id: Optional[int]) -> bool:
    if leader_id is None:
        return True
    env_leader = shared_root / f"env_{leader_id}"
    # if leader env has failed flag, treat as failed
    return (env_leader / "cfl3d_failed.flag").exists()

def _pick_new_leader_from_ready(shared_root: Path, n_envs: int) -> Optional[int]:
    # choose smallest env_id with READY and not FAILED
    for i in range(n_envs):
        env_i = shared_root / f"env_{i}"
        if (env_i / "cfl3d_ready.flag").exists() and not (env_i / "cfl3d_failed.flag").exists():
            return i
    return None

def _wait_done_flag(done_flag: Path, timeout: float = 3600.0, poll: float = 2.0, vprint=None):
    """Wait until leader writes DONE/TIMEOUT/FAILED into the shared done_flag."""
    t0 = time.time()
    while True:
        if done_flag.exists():
            try:
                st = done_flag.read_text().strip()
            except OSError:
                st = ""
            if st.startswith(("DONE", "TIMEOUT", "FAILED")):
                return st

        if time.time() - t0 > timeout:
            if vprint:
                vprint(f"[wait_done] Timeout waiting {done_flag} ({timeout}s).")
            return "WAIT_DONE_TIMEOUT"

        time.sleep(poll)


def cfl3d_airfoil(
    env_id,
    n_envs,
    airfoil_file,
    angle_of_attack,
    Re_number,
    work_dir,
    startup_wait: float = 10.0,
    eps: float = 1e-8,
    verbose: int = 1,
    done_wait_timeout: float = 3600.0,   # how long non-leaders wait before giving up cleanup
):
    """
    Run a CFL3D simulation and return (CL, CD).
    IMPORTANT CHANGE:
      - Non-leaders DO NOT clean env directory until leader writes cfl3d_done.flag.
      - Leader can optionally clean at the very end after scancel + DONE.
    """
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    work_dir = Path(work_dir)
    ## shared_root = work_dir.parent   it was done for OPTUNA
    shared_root = work_dir.parent

    # flags
    leader_flag = shared_root / "cfl3d_leader.flag"
    done_flag   = shared_root / "cfl3d_done.flag"
    cleanup_lock = shared_root / "cfl3d_cleanup.lock"

    conv_flag   = work_dir / "cfl3d_conv.flag"
    status_flag = work_dir / "status.flag"
    ready_flag  = work_dir / "cfl3d_ready.flag"
    failed_flag = work_dir / "cfl3d_failed.flag"
    cfl3d_error = work_dir / "cfl3d.error"

    job_id = None
    is_leader = False
    cwd = Path.cwd()

    # -------------------------
    # Per-env cleanup at entry
    # (SAFE: we only remove our own flags from previous attempt)
    # -------------------------
    for flag in (status_flag, failed_flag, conv_flag, ready_flag, cfl3d_error):
        if flag.exists():
            _safe_unlink(flag, vprint=vprint)

    # -------------------------
    # One-time shared cleanup at entry
    # -------------------------
    try:
        fd = os.open(cleanup_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        for f in (leader_flag, done_flag):
            if f.exists():
                _safe_unlink(f, vprint=vprint)
    except FileExistsError:
        pass

    try:
        os.chdir(work_dir)

        vprint(f"[Env {env_id}] Starting in {work_dir} AoA={angle_of_attack}, Re={Re_number}")

        # Step 0: copy inputs
        copy_main_inputs(shared_root, work_dir)

        # Step 1: update inp
        update_cfl3d_inp(
            folder=work_dir,
            file_name="cfl3d.inp_1",
            new_alpha=angle_of_attack,
            new_reynolds=Re_number,
        )

        # Step 2: mesh
        try:
            cfl3d_mesh(airfoil_file, Re_number, work_dir)
        except Exception as e:
            vprint(f"[Env {env_id}] Mesh FAILED: {e}")
            failed_flag.write_text("FAILED_MESH\n")
            status_flag.write_text("FAILED_MESH\n")
            return eps, eps

        # Step 3: splitter
        try:
            split_inp_path = work_dir / "split.inp"
            if not split_inp_path.exists():
                vprint(f"[Env {env_id}] split.inp missing")
                failed_flag.write_text("FAILED_NO_SPLIT_INP\n")
                status_flag.write_text("FAILED_NO_SPLIT_INP\n")
                return eps, eps

            with open(split_inp_path, "r") as input_file:
                subprocess.run(["splitter"], stdin=input_file, cwd=work_dir, check=False)
        except Exception as e:
            vprint(f"[Env {env_id}] Splitter FAILED: {e}")
            failed_flag.write_text("FAILED_SPLITTER\n")
            status_flag.write_text("FAILED_SPLITTER\n")
            return eps, eps

        # Mark ready
        ready_flag.write_text("READY\n")
        status_flag.write_text("READY\n")

        # Step 3b: elect leader
        if not leader_flag.exists():
            try:
                fd = os.open(leader_flag, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(env_id).encode())
                os.close(fd)
                is_leader = True
                vprint(f"[Env {env_id}] Became leader.")
            except FileExistsError:
                pass

        # Leader submits job, others wait for done_flag STARTED/DONE
        if is_leader:
            vprint(f"[Env {env_id}] Submitting CFL3D job...")

            # --- Gather window: wait for READY count to stabilize ---
            gather_timeout = 20.0   # seconds (tune 10–30)
            stable_for     = 4.0    # seconds stable before we submit
            poll           = 0.5

            t0 = time.time()
            last_ready = -1
            last_change = time.time()

            while True:
                n_ready, n_failed = count_ready_failed(shared_root, n_envs)

                if n_ready != last_ready:
                    last_ready = n_ready
                    last_change = time.time()
                    vprint(f"[Leader {env_id}] READY now: {n_ready} / {n_envs} (failed={n_failed})")

                # stop early if everyone is ready
                if n_ready >= n_envs:
                    break

                # submit once READY count hasn't changed for stable_for seconds
                if time.time() - last_change >= stable_for and n_ready > 0:
                    break

                # hard timeout
                if time.time() - t0 >= gather_timeout:
                    break

                time.sleep(poll)

            if n_ready == 0:
                vprint(f"[Env {env_id}] No READY envs -> not submitting.")
                failed_flag.write_text("FAILED_NO_READY\n")
                status_flag.write_text("FAILED_NO_READY\n")
                done_flag.write_text("FAILED_NO_READY\n")
                _safe_unlink(leader_flag, vprint=vprint)
                return eps, eps

            ntasks = 4 * n_ready
            vprint(f"[Leader {env_id}] Submitting batch for n_ready={n_ready} -> ntasks={ntasks}")

            job_id, job_state = main_cfl3d(script_name=config.CFL3D_SCRIPT, ntasks=ntasks)
            
            if not job_id or job_state != "R":
                vprint(f"[Env {env_id}] Submit FAILED (job_id={job_id}, state={job_state})")
                failed_flag.write_text("FAILED_CFL3D_SUBMIT\n")
                status_flag.write_text("FAILED_CFL3D_SUBMIT\n")
                done_flag.write_text("FAILED_SUBMIT\n")   # shared -> wake others
                return eps, eps

            if startup_wait > 0:
                time.sleep(startup_wait)

            status_flag.write_text("STARTED\n")
            done_flag.write_text("STARTED\n")  # shared start signal
            
        else:
            vprint(f"[Env {env_id}] Waiting for STARTED via {done_flag.name}")
            t0 = time.time()
            while True:
                if done_flag.exists():
                    st = done_flag.read_text().strip()
                    if st.startswith(("STARTED", "DONE", "TIMEOUT")):
                        break
                    if st.startswith("FAILED"):
                        failed_flag.write_text("FAILED_CFL3D_START\n")
                        status_flag.write_text("FAILED_CFL3D_START\n")
                        return eps, eps
                if time.time() - t0 > 7200.0:
                    failed_flag.write_text("FAILED_WAIT_START\n")
                    status_flag.write_text("FAILED_WAIT_START\n")
                    return eps, eps
                time.sleep(2.0)

        # Step 4a: wait for cfl3d.out to appear
        cfl3d_out = work_dir / "cfl3d.out"
        t0 = time.time()
        start_timeout = startup_wait + 60.0
        while True:
            failed, err_code = check_cfl3d_error(work_dir)
            if failed:
                vprint(f"[Env {env_id}] FAILED early (cfl3d.error={err_code})")
                failed_flag.write_text(f"CFL3D_ERROR {err_code}\n")
                status_flag.write_text("FAILED_CFL3D_ERROR\n")
                return eps, eps

            if cfl3d_out.exists() and cfl3d_out.stat().st_size > 0:
                break

            if time.time() - t0 > start_timeout:
                vprint(f"[Env {env_id}] No output within {start_timeout}s")
                failed_flag.write_text("FAILED_NO_CFL3D_OUTPUT\n")
                status_flag.write_text("FAILED_NO_CFL3D_OUTPUT\n")
                return eps, eps

            time.sleep(2.0)

        # Step 4b: init crash detector
        failed_init, reason = cfl3d_out_init_crashed(work_dir, stall_timeout=60.0)
        if failed_init:
            failed, err_code = check_cfl3d_error(work_dir)
            msg = reason if not failed else f"{reason} + cfl3d.error={err_code}"
            vprint(f"[Env {env_id}] INIT FAILED: {msg}")
            failed_flag.write_text("FAILED_CFL3D_INIT\n")
            status_flag.write_text("FAILED_CFL3D_INIT\n")
            return eps, eps

        # Step 4c: monitor to get CL/CD and write conv_flag
        Cl_new, Cd_new = follow_cfl3d_output(
            "cfl3d.out",
            sleep_time=5,
            threshold=0.5e-9,
            max_iter=90000,
            conv_flag_path=conv_flag,
        )
        if Cl_new is None or Cd_new is None:
            vprint(f"[Env {env_id}] Monitor returned None")
            failed_flag.write_text("FAILED_MONITOR\n")
            status_flag.write_text("FAILED_MONITOR\n")
            return eps, eps

        Cl, Cd = Cl_new, Cd_new
        status_flag.write_text("OK\n")

        # --- Leader failover before Step 5 ---
        # If current leader failed, elect a new leader among READY envs.
        leader_id = _read_leader_env_id(leader_flag) if leader_flag.exists() else None
        if _leader_is_failed(shared_root, leader_id):
            new_leader = _pick_new_leader_from_ready(shared_root, n_envs)
            if new_leader is not None:
                if env_id == new_leader:
                    # force takeover (overwrite leader_flag)
                    _force_replace_leader(leader_flag, env_id, vprint=vprint)
                    is_leader = True
                    vprint(f"[Env {env_id}] Failover: became new leader before Step 5.")
                else:
                    vprint(f"[Env {env_id}] Failover: leader failed; candidate leader is env_{new_leader}.")
        # Step 5: leader waits for all READY envs to finish (conv or failed), then scancel and DONE
        if is_leader:
            timeout = 7200.0
            poll = 2.0
            t0 = time.time()

            # Track only envs that were READY (avoid waiting on never-ready envs)
            target_envs = []
            for i in range(n_envs):
                env_i = shared_root / f"env_{i}"
                if (env_i / "cfl3d_ready.flag").exists():
                    target_envs.append(i)

            vprint(f"[Leader {env_id}] Tracking READY envs: {target_envs} (job_id={job_id})")

            while True:
                pending = []
                for i in target_envs:
                    env_i = shared_root / f"env_{i}"
                    if (env_i / "cfl3d_conv.flag").exists():
                        continue
                    if (env_i / "cfl3d_failed.flag").exists():
                        continue
                    pending.append(i)

                if not pending:
                    vprint(f"[Leader {env_id}] All READY envs have terminal flags.")
                    # If we know the SLURM job id, cancel it (optional; safe)
                    if job_id is not None:
                        try:
                            vprint(f"[Leader {env_id}] scancel {job_id}")
                            kill_job(job_id)
                        except Exception as e:
                            vprint(f"[Leader {env_id}] kill_job failed: {e}")
                    # CRITICAL: ALWAYS wake everyone up
                    done_flag.write_text("DONE\n")
                    break

                if time.time() - t0 > timeout:
                    vprint(f"[Leader {env_id}] TIMEOUT pending={pending}")
                    if job_id is not None:
                        try:
                            kill_job(job_id)
                        except Exception as e:
                            vprint(f"[Leader {env_id}] kill_job failed: {e}")
                    done_flag.write_text("TIMEOUT\n")
                    break

                time.sleep(poll)

            _safe_unlink(cleanup_lock, vprint=vprint)

        # -------------------------
        # Cleanup only after DONE/TIMEOUT/FAILED is visible.
        # Non-leaders must not delete flags before leader sees them.
        # -------------------------
        if not is_leader:
            _wait_done_flag(done_flag, timeout=done_wait_timeout, poll=2.0, vprint=vprint)

        # Now cleanup this env safely
        clean_env(work_dir)

        return Cl, Cd

    finally:
        os.chdir(cwd)