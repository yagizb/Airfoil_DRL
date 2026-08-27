from networkx import config

from CFL3DMesh import cfl3d_mesh
from CFL3DSubJob import main_cfl3d, kill_job
from CFL3DConver import (
    follow_cfl3d_output,
    check_cfl3d_error,
    cfl3d_out_init_crashed,
    count_ready_failed,
)
from CFL3DHelper import (
    _safe_unlink,
    _try_acquire_lock,
   _release_lock,
   _read_int_flag,
   _write_int_flag_atomic,
   _pick_new_leader_from_ready,
   _wait_done_flag,
   _fail_and_return,
   _leader_is_failed,
   )
from CFL3DPrep import update_cfl3d_inp, copy_main_inputs, clean_env

import time
import os
import subprocess
from pathlib import Path
import DRL_config
from typing import Optional

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
    done_wait_timeout: float = 3600.0,
):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    work_dir = Path(work_dir)
    shared_root = work_dir.parent

    leader_flag = shared_root / "cfl3d_leader.flag"
    done_flag = shared_root / "cfl3d_done.flag"
    cleanup_lock = shared_root / "cfl3d_cleanup.lock"
    takeover_lock = shared_root / "cfl3d_takeover.lock"

    conv_flag = work_dir / "cfl3d_conv.flag"
    status_flag = work_dir / "status.flag"
    ready_flag = work_dir / "cfl3d_ready.flag"
    failed_flag = work_dir / "cfl3d_failed.flag"
    cfl3d_error = work_dir / "cfl3d.error"

    job_id = None
    is_leader = False
    cwd = Path.cwd()

    for flag in (status_flag, failed_flag, conv_flag, ready_flag, cfl3d_error):
        if flag.exists():
            _safe_unlink(flag, vprint=vprint)

    try:
        fd = os.open(str(cleanup_lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        for f in (leader_flag, done_flag, takeover_lock):
            if f.exists():
                _safe_unlink(f, vprint=vprint)
    except FileExistsError:
        pass

    try:
        os.chdir(work_dir)
        vprint(f"[Env {env_id}] Starting in {work_dir} AoA={angle_of_attack}, Re={Re_number}")

        copy_main_inputs(shared_root, work_dir)

        update_cfl3d_inp(
            folder=work_dir,
            file_name="cfl3d.inp_1",
            new_alpha=angle_of_attack,
            new_reynolds=Re_number,
        )

        try:
            cfl3d_mesh(airfoil_file, Re_number, work_dir)
        except Exception as e:
            vprint(f"[Env {env_id}] Mesh FAILED: {e}")
            return _fail_and_return(
                env_id=env_id,
                reason="FAILED_MESH",
                failed_flag=failed_flag,
                status_flag=status_flag,
                done_flag=done_flag,
                leader_flag=leader_flag,
                cleanup_lock=cleanup_lock,
                takeover_lock=takeover_lock,
                is_leader=is_leader,
                eps=eps,
                vprint=vprint,
            )

        try:
            split_inp_path = work_dir / "split.inp"
            if not split_inp_path.exists():
                vprint(f"[Env {env_id}] split.inp missing")
                return _fail_and_return(
                    env_id=env_id,
                    reason="FAILED_NO_SPLIT_INP",
                    failed_flag=failed_flag,
                    status_flag=status_flag,
                    done_flag=done_flag,
                    leader_flag=leader_flag,
                    cleanup_lock=cleanup_lock,
                    takeover_lock=takeover_lock,
                    is_leader=is_leader,
                    eps=eps,
                    vprint=vprint,
                )

            with open(split_inp_path, "r") as input_file:
                subprocess.run(["splitter"], stdin=input_file, cwd=work_dir, check=False)
        except Exception as e:
            vprint(f"[Env {env_id}] Splitter FAILED: {e}")
            return _fail_and_return(
                env_id=env_id,
                reason="FAILED_SPLITTER",
                failed_flag=failed_flag,
                status_flag=status_flag,
                done_flag=done_flag,
                leader_flag=leader_flag,
                cleanup_lock=cleanup_lock,
                takeover_lock=takeover_lock,
                is_leader=is_leader,
                eps=eps,
                vprint=vprint,
            )

        ready_flag.write_text("READY\n")
        status_flag.write_text("READY\n")

        if not leader_flag.exists():
            try:
                fd = os.open(str(leader_flag), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(env_id).encode())
                os.close(fd)
                is_leader = True
                vprint(f"[Env {env_id}] Became leader.")
            except FileExistsError:
                pass

        if is_leader:
            vprint(f"[Env {env_id}] Submitting CFL3D job...")

            gather_timeout = 20.0
            stable_for = 4.0
            poll = 0.5

            t0 = time.time()
            last_ready = -1
            last_change = time.time()

            while True:
                n_ready, n_failed = count_ready_failed(shared_root, n_envs)

                if n_ready != last_ready:
                    last_ready = n_ready
                    last_change = time.time()
                    vprint(f"[Leader {env_id}] READY now: {n_ready} / {n_envs} (failed={n_failed})")

                if n_ready >= n_envs:
                    break

                if time.time() - last_change >= stable_for and n_ready > 0:
                    break

                if time.time() - t0 >= gather_timeout:
                    break

                time.sleep(poll)

            if n_ready == 0:
                return _fail_and_return(
                    env_id=env_id,
                    reason="FAILED_NO_READY",
                    failed_flag=failed_flag,
                    status_flag=status_flag,
                    done_flag=done_flag,
                    leader_flag=leader_flag,
                    cleanup_lock=cleanup_lock,
                    takeover_lock=takeover_lock,
                    is_leader=is_leader,
                    eps=eps,
                    vprint=vprint,
                )

            ntasks = 4 * n_ready
            vprint(f"[Leader {env_id}] Submitting batch for n_ready={n_ready} -> ntasks={ntasks}")

            job_id, job_state = main_cfl3d(script_name=config.CFL3D_SCRIPT, ntasks=ntasks)

            if not job_id or job_state != "R":
                vprint(f"[Env {env_id}] Submit FAILED (job_id={job_id}, state={job_state})")
                return _fail_and_return(
                    env_id=env_id,
                    reason="FAILED_CFL3D_SUBMIT",
                    failed_flag=failed_flag,
                    status_flag=status_flag,
                    done_flag=done_flag,
                    leader_flag=leader_flag,
                    cleanup_lock=cleanup_lock,
                    takeover_lock=takeover_lock,
                    is_leader=is_leader,
                    eps=eps,
                    vprint=vprint,
                )

            if startup_wait > 0:
                time.sleep(startup_wait)

            status_flag.write_text("STARTED\n")
            done_flag.write_text("STARTED\n")

        else:
            vprint(f"[Env {env_id}] Waiting for STARTED via {done_flag.name}")
            t0 = time.time()
            while True:
                if done_flag.exists():
                    st = done_flag.read_text().strip()
                    if st.startswith(("STARTED", "DONE", "TIMEOUT")):
                        break
                    if st.startswith("FAILED"):
                        return _fail_and_return(
                            env_id=env_id,
                            reason="FAILED_CFL3D_START",
                            failed_flag=failed_flag,
                            status_flag=status_flag,
                            done_flag=done_flag,
                            leader_flag=leader_flag,
                            cleanup_lock=cleanup_lock,
                            takeover_lock=takeover_lock,
                            is_leader=is_leader,
                            eps=eps,
                            vprint=vprint,
                        )
                if time.time() - t0 > 7200.0:
                    return _fail_and_return(
                        env_id=env_id,
                        reason="FAILED_WAIT_START",
                        failed_flag=failed_flag,
                        status_flag=status_flag,
                        done_flag=done_flag,
                        leader_flag=leader_flag,
                        cleanup_lock=cleanup_lock,
                        takeover_lock=takeover_lock,
                        is_leader=is_leader,
                        eps=eps,
                        vprint=vprint,
                    )
                time.sleep(2.0)

        cfl3d_out = work_dir / "cfl3d.out"
        t0 = time.time()
        start_timeout = startup_wait + 60.0

        while True:
            failed, err_code = check_cfl3d_error(work_dir)
            if failed:
                vprint(f"[Env {env_id}] FAILED early (cfl3d.error={err_code})")
                return _fail_and_return(
                    env_id=env_id,
                    reason="FAILED_CFL3D_ERROR",
                    failed_flag=failed_flag,
                    status_flag=status_flag,
                    done_flag=done_flag,
                    leader_flag=leader_flag,
                    cleanup_lock=cleanup_lock,
                    takeover_lock=takeover_lock,
                    is_leader=is_leader,
                    eps=eps,
                    vprint=vprint,
                )

            if cfl3d_out.exists() and cfl3d_out.stat().st_size > 0:
                break

            if time.time() - t0 > start_timeout:
                vprint(f"[Env {env_id}] No output within {start_timeout}s")
                return _fail_and_return(
                    env_id=env_id,
                    reason="FAILED_NO_CFL3D_OUTPUT",
                    failed_flag=failed_flag,
                    status_flag=status_flag,
                    done_flag=done_flag,
                    leader_flag=leader_flag,
                    cleanup_lock=cleanup_lock,
                    takeover_lock=takeover_lock,
                    is_leader=is_leader,
                    eps=eps,
                    vprint=vprint,
                )

            time.sleep(2.0)

        failed_init, reason = cfl3d_out_init_crashed(work_dir, stall_timeout=60.0)
        if failed_init:
            failed, err_code = check_cfl3d_error(work_dir)
            msg = reason if not failed else f"{reason} + cfl3d.error={err_code}"
            vprint(f"[Env {env_id}] INIT FAILED: {msg}")
            return _fail_and_return(
                env_id=env_id,
                reason="FAILED_CFL3D_INIT",
                failed_flag=failed_flag,
                status_flag=status_flag,
                done_flag=done_flag,
                leader_flag=leader_flag,
                cleanup_lock=cleanup_lock,
                takeover_lock=takeover_lock,
                is_leader=is_leader,
                eps=eps,
                vprint=vprint,
            )

        Cl_new, Cd_new = follow_cfl3d_output(
            "cfl3d.out",
            sleep_time=5,
            threshold=0.5e-9,
            max_iter=90000,
            conv_flag_path=conv_flag,
        )
        if Cl_new is None or Cd_new is None:
            vprint(f"[Env {env_id}] Monitor returned None")
            return _fail_and_return(
                env_id=env_id,
                reason="FAILED_MONITOR",
                failed_flag=failed_flag,
                status_flag=status_flag,
                done_flag=done_flag,
                leader_flag=leader_flag,
                cleanup_lock=cleanup_lock,
                takeover_lock=takeover_lock,
                is_leader=is_leader,
                eps=eps,
                vprint=vprint,
            )

        Cl, Cd = Cl_new, Cd_new
        status_flag.write_text("OK\n")

        leader_id = _read_int_flag(leader_flag) if leader_flag.exists() else None
        if _leader_is_failed(shared_root, leader_id):
            if _try_acquire_lock(takeover_lock):
                try:
                    leader_id2 = _read_int_flag(leader_flag) if leader_flag.exists() else None
                    if _leader_is_failed(shared_root, leader_id2):
                        new_leader = _pick_new_leader_from_ready(shared_root, n_envs)
                        if new_leader is not None:
                            _write_int_flag_atomic(leader_flag, new_leader)
                            vprint(f"[Env {env_id}] Failover elected env_{new_leader} as leader.")
                            if env_id == new_leader:
                                is_leader = True
                                vprint(f"[Env {env_id}] I took over leadership.")
                        else:
                            vprint(f"[Env {env_id}] Failover: no READY env available.")
                finally:
                    _release_lock(takeover_lock)

        if is_leader:
            timeout = 7200.0
            poll = 2.0
            t0 = time.time()

            target_envs = []
            for i in range(n_envs):
                env_i = shared_root / f"env_{i}"
                ready = (env_i / "cfl3d_ready.flag").exists()
                failed = (env_i / "cfl3d_failed.flag").exists()
                conv = (env_i / "cfl3d_conv.flag").exists()
                if ready and not failed and not conv:
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
                    if job_id is not None:
                        try:
                            vprint(f"[Leader {env_id}] scancel {job_id}")
                            kill_job(job_id)
                        except Exception as e:
                            vprint(f"[Leader {env_id}] kill_job failed: {e}")
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
            _safe_unlink(leader_flag, vprint=vprint)
            _safe_unlink(takeover_lock, vprint=vprint)

        else:
            _wait_done_flag(done_flag, timeout=done_wait_timeout, poll=2.0, vprint=vprint)

        clean_env(work_dir)
        return Cl, Cd

    finally:
        os.chdir(cwd)