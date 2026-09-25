from SOD2DMesh import (sod2d_mesh,write_ext_nmf,write_geo_file,write_partition_input_json)
from SOD2DPrep import copy_sod2d_main_inputs
from p3d2gmsh import p3d2gmsh
from SOD2DSolver import write_BluffBodySolverIncomp_json
from SOD2DConver import wait_until_sod2d_converged
from SOD2DHelper import (count_sod2d_ready_failed)
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

from SOD2DSubJob import (main_sod2d, kill_job)

from CFL3DPrep import (clean_env)

import json
import os
import subprocess
import time
from pathlib import Path
import DRL_config

def valid_file(path, min_size=1024):
    return path.exists() and path.is_file() and path.stat().st_size > min_size

def read_sod2d_config(config_file="flow_config.json"):
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / config_file

    with open(config_path, "r") as f:
        sod2d_config = json.load(f)

    return sod2d_config


def sod2d_airfoil(
    env_id,
    n_envs,
    airfoil_file,
    angle_of_attack,
    Re_number,
    fidelity,
    work_dir,
    startup_wait: float = 10.0,
    eps: float = 1e-8,
    verbose: int = 1,
    done_wait_timeout: float = 3600.0,
    max_sod2d_restarts: int = 5,
):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    sod2d_config = read_sod2d_config()

    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    shared_root = work_dir.parent.resolve()

    # Python file location
    SCRIPT_DIR = Path(__file__).resolve().parent
    MAIN_SOD2D = SCRIPT_DIR / "main_sod2d"

    vprint(f"[Env {env_id}] SCRIPT_DIR  = {SCRIPT_DIR}")
    vprint(f"[Env {env_id}] MAIN_SOD2D  = {MAIN_SOD2D}")
    vprint(f"[Env {env_id}] work_dir    = {work_dir}")

    leader_flag = shared_root / "sod2d_leader.flag"
    done_flag = shared_root / "sod2d_done.flag"
    cleanup_lock = shared_root / "sod2d_cleanup.lock"
    takeover_lock = shared_root / "sod2d_takeover.lock"

    conv_flag = work_dir / "sod2d_conv.flag"
    status_flag = work_dir / "status.flag"
    ready_flag = work_dir / "sod2d_ready.flag"
    failed_flag = work_dir / "sod2d_failed.flag"

    job_id = None
    is_leader = False
    cwd = Path.cwd()
    
    for flag in (status_flag, failed_flag, conv_flag, ready_flag):
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
            
    #base_name = config['airfoil_dat_file'].replace(".dat", "")
    try:
        os.chdir(work_dir)  # change to the working directory for this environment
        vprint(f"[Env {env_id}] Starting in {work_dir} AoA={angle_of_attack}, Re={Re_number}")
        
        # Step 0: copy inputs
        copy_sod2d_main_inputs(work_dir,shared_root)
        # Step 2: mesh
        ## generate ext_p3d
        try:
            sod2d_mesh(airfoil_file,Re_number,work_dir,sod2d_config['jmax'],sod2d_config['radi'],
                sod2d_config['z_spanwise_len'],sod2d_config['z_spanwise_planes'],
                sod2d_config['LES_ypls'],sod2d_config['mesh_type'])
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
        ## prepare ext_nmf file
        write_ext_nmf(airfoil_file, work_dir,sod2d_config['idim'], sod2d_config['jmax'], sod2d_config['z_spanwise_planes'])
       ## convert p3d to GMSH format and run the simulation
        try:
            p3d2gmsh(
                p3d_file=f"{airfoil_file}_ext.p3d",
                idim=sod2d_config["idim"],
                angle_of_attack=sod2d_config["angle_of_attack"],
                width_outlet=sod2d_config["width_outlet"],
                map_file=f"{airfoil_file}_ext.nmf",
                output_file=f"{airfoil_file}.msh"
                )
        except Exception as e:
            vprint(f"[Env {env_id}] p3d to GMSH FAILED: {e}")
            return _fail_and_return(
                env_id=env_id,
                reason="FAILED_p3d2GMESH",
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
       ## gmsh airfoil_per.geo -0
        write_geo_file(airfoil_file, work_dir, sod2d_config['z_spanwise_len'],sod2d_config['porder'],sod2d_config['per_surface_ID'])
        subprocess.run(
            """
            module purge
            module load intel/2023.2.0 mkl/2023.2.0 impi/2021.10.0 \
            hdf5/1.14.1-2-gcc python/3.12.1 gmsh

            gmsh airfoil_per.geo -0
            """,
            shell=True,
            executable="/bin/bash",
            check=True,
        )
        ## convert gmsh .msh to SOD2D .h5

        GMSH2SOD2D = MAIN_SOD2D / "gmsh2sod2d.py"

        cmd_gmsh2sod2d = f"""
        module purge
        module load intel/2023.2.0 mkl/2023.2.0 impi/2021.10.0 \
                    hdf5/1.14.1-2-gcc python/3.12.1

        python3 "{GMSH2SOD2D}" \
            {airfoil_file}_per \
            -p {sod2d_config["per_surface_ID"]} \
            -r {sod2d_config["porder"]}
        """

        subprocess.run(
            cmd_gmsh2sod2d,
            cwd=work_dir,
            shell=True,
            executable="/bin/bash",
            check=True,
)
       
        ## write partition input json
        write_partition_input_json( airfoil_file, work_dir,sod2d_config["num_partitions"])

        TOOL_MESH = MAIN_SOD2D / "tool_meshConversorPar"

        cmd_toolmesh = f"""
        module purge
        module load openmpi/4.1.5-gcc ucx/1.16.0-gcc hdf5/1.14.1-2-gcc-openmpi cmake

        export OMPI_MCA_io=romio321
        export OMP_NUM_THREADS=1
        export SLURM_CPU_BIND=none

        echo "PWD=$(pwd)"
        echo "TOOL_MESH={TOOL_MESH}"
        echo "input.json:"
        ls -lh input.json

        mpirun --bind-to none -np {sod2d_config["num_partitions"]} \
            "{TOOL_MESH}" input.json
        """

        subprocess.run(
            cmd_toolmesh,
            cwd=work_dir,
            shell=True,
            executable="/bin/bash",
            check=True,
    )
        
        # İf interploate solution is available:
        grid_file = work_dir / "plot3dg.bin"
        q_file    = work_dir / "plot3dq.bin"

        loadRestartFile = False

        grid_ok = valid_file(grid_file, min_size=1024)
        q_ok    = valid_file(q_file, min_size=1024)

        if grid_ok and q_ok:
            try:
                CFL3D_TO_SOD2D = MAIN_SOD2D / "cfl3d_to_sod2d_source.py"

                cmd_cfl3d_to_sod2d = f"""
                module purge
                module load intel/2023.2.0 impi/2021.10.0 hdf5/1.14.1-2 mkl/2023.2.0 python/3.12.1

                python3 "{CFL3D_TO_SOD2D}" \
                    --grid plot3dg.bin \
                    --q plot3dq.bin \
                    --z-planes {sod2d_config["z_spanwise_planes"]} \
                    --z-span {sod2d_config["z_spanwise_len"]} \
                    --porder {sod2d_config["porder"]} \
                    --mach {sod2d_config["mach"]} \
                    --real float32 \
                    --out-src-mesh cfl3d.h5 \
                    --out-src-restart restart_cfl3d.h5 \
                    --rho-constant-one
                """

                subprocess.run(
                    cmd_cfl3d_to_sod2d,
                    cwd=work_dir,
                    shell=True,
                    executable="/bin/bash",
                    check=True,
                )

                INTERPOLATE = MAIN_SOD2D / "interpolate.py"

                cmd_interpolate = f"""
                module purge
                module load intel/2023.2.0 impi/2021.10.0 hdf5/1.14.1-2 mkl/2023.2.0 python/3.12.1

                mpirun --bind-to none -np 1 python3 "{INTERPOLATE}" \
                    cfl3d.h5 \
                    {airfoil_file}_per-{sod2d_config["num_partitions"]}.hdf \
                    restart_cfl3d.h5 \
                    restart_{airfoil_file}_per-{sod2d_config["num_partitions"]}_1.h5
                """

                subprocess.run(
                    cmd_interpolate,
                    cwd=work_dir,
                    shell=True,
                    executable="/bin/bash",
                    check=True,
                )

                loadRestartFile = True
                print(f"[env {env_id}] CFL3D restart interpolation succeeded.")

            except subprocess.CalledProcessError as e:
                print(f"[env {env_id}] CFL3D restart interpolation failed: {e}")
                print(f"[env {env_id}] Continuing SOD2D without restart.")

                loadRestartFile = False

        else:
            print(
                f"[env {env_id}] CFL3D restart files missing or empty. "
                f"plot3dg.bin exists={grid_file.exists()}, size={grid_file.stat().st_size if grid_file.exists() else 0}; "
                f"plot3dq.bin exists={q_file.exists()}, size={q_file.stat().st_size if q_file.exists() else 0}. "
                "Skipping CFL3D restart interpolation."
            )
            loadRestartFile = False
        
         
        write_BluffBodySolverIncomp_json(airfoil_file,loadRestartFile, work_dir, sod2d_config) 
        
        ready_flag.write_text("READY\n")
        status_flag.write_text("READY\n")
        
        if n_envs == 1:
            _write_int_flag_atomic(leader_flag, env_id)
            leader_id = env_id
            is_leader = True

            vprint(f"[Env {env_id}] Single environment: became leader.")

        else:
            if not leader_flag.exists():
                try:
                    fd = os.open(
                        str(leader_flag),
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    )
                    os.write(fd, str(env_id).encode())
                    os.close(fd)

                except FileExistsError:
                    pass

            leader_id = (
                _read_int_flag(leader_flag)
                if leader_flag.exists()
                else None
            )
            is_leader = leader_id == env_id

            vprint(
                f"[Env {env_id}] leader_id={leader_id}, "
                f"is_leader={is_leader}"
            )

        restart_id = 0
        CL_sod2d = None
        CD_sod2d = None

        while restart_id <= max_sod2d_restarts:

            # =================================================
            # LEADER: submit SOD2D batch job
            # =================================================
            if is_leader:
                vprint(
                    f"[Env {env_id}] Submitting SOD2D job "
                    f"(restart {restart_id}/{max_sod2d_restarts})..."
                )

                gather_timeout = 20.0
                stable_for = 4.0
                poll = 0.5

                t0 = time.time()
                last_ready = -1
                last_change = time.time()

                while True:
                    n_ready, n_failed = count_sod2d_ready_failed(shared_root, n_envs)

                    if n_ready != last_ready:
                        last_ready = n_ready
                        last_change = time.time()
                        vprint(
                            f"[Leader {env_id}] READY now: "
                            f"{n_ready} / {n_envs} failed={n_failed}"
                        )

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

                vprint(
                    f"[Leader {env_id}] Submitting batch for "
                    f"n_ready={n_ready} -> ntasks={ntasks}"
                )
                
                job_id, job_state = main_sod2d(
                    script_name=DRL_config.SOD2D_SCRIPT,
                    ntasks=ntasks,
                )

                if not job_id or job_state != "R":
                    vprint(
                        f"[Env {env_id}] Submit FAILED "
                        f"job_id={job_id}, state={job_state}"
                    )

                    return _fail_and_return(
                        env_id=env_id,
                        reason="FAILED_sod2d_SUBMIT",
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
                done_flag.write_text(f"STARTED_RESTART_{restart_id}\n")

            # =================================================
            # WORKER: wait until leader starts SOD2D
            # =================================================
            else:
                vprint(f"[Env {env_id}] Waiting for SOD2D STARTED...")

                t0 = time.time()

                while True:
                    if done_flag.exists():
                        st = done_flag.read_text().strip()

                        if st.startswith("STARTED"):
                            break

                        if st.startswith("FAILED"):
                            return _fail_and_return(
                                env_id=env_id,
                                reason="FAILED_SOD2D_START",
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

            # =================================================
            # MONITOR THIS ENVIRONMENT
            # =================================================
            status, CL_tmp, CD_tmp = wait_until_sod2d_converged(
                output_geo=airfoil_file,
                angle_of_attack=angle_of_attack,
                chord=sod2d_config["airfoil_chord"],
                Lz=sod2d_config["z_spanwise_len"],
                rho=sod2d_config["rho"],
                Uinf=sod2d_config["Uinf"],
                max_wait_time=7200,
            )

            # =================================================
            # SUCCESS
            # =================================================
            if status == "CONVERGED":
                CL_sod2d = CL_tmp
                CD_sod2d = CD_tmp
                last_CL = CL_tmp
                last_CD = CD_tmp
                break

            elif status == "TIMEOUT":
                if CL_tmp is not None and CD_tmp is not None:
                    last_CL = CL_tmp
                    last_CD = CD_tmp

            elif status == "DIVERGED":
                CL_sod2d = None
                CD_sod2d = None
                break

            vprint(
                f"[Env {env_id}] SOD2D not converged after 7200 s. "
                f"restart_id={restart_id}"
            )

            # =================================================
            # NO MORE RESTARTS
            # =================================================
            if restart_id >= max_sod2d_restarts:
                if last_CL is not None and last_CD is not None:
                    CL_sod2d = last_CL
                    CD_sod2d = last_CD
                    break
                else:
                    break

            # =================================================
            # LEADER: kill and announce restart
            # =================================================
            if is_leader:
                if job_id is not None:
                    try:
                        vprint(f"[Leader {env_id}] Killing SOD2D job_id={job_id}")
                        kill_job(job_id)
                    except Exception as e:
                        vprint(f"[Leader {env_id}] kill_job failed: {e}")

                done_flag.write_text(f"RESTART_{restart_id + 1}\n")

            # =================================================
            # WORKER: wait until leader announces restart
            # =================================================
            else:
                vprint(f"[Env {env_id}] Waiting for leader restart...")

                while True:
                    if done_flag.exists():
                        st = done_flag.read_text().strip()

                        if st.startswith("RESTART"):
                            break

                        if st.startswith("FAILED"):
                            break

                    time.sleep(2.0)

            restart_id += 1
            time.sleep(5.0)

        # =====================================================
        # FINAL FAILURE AFTER ALL RESTARTS
        # =====================================================
        if CL_sod2d is None or CD_sod2d is None:
            vprint(f"[Env {env_id}] Monitor returned None. SOD2D failed.")

            failed_flag.write_text("FAILED_MONITOR\n")
            status_flag.write_text("FAILED_MONITOR\n")

            if job_id is not None:
                try:
                    vprint(f"[Env {env_id}] Killing SOD2D job_id={job_id}")
                    kill_job(job_id)
                except Exception as e:
                    vprint(f"[Env {env_id}] kill_job failed: {e}")

            done_flag.write_text("FAILED_MONITOR\n")

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

        vprint(f"[Env {env_id}] Convergence achieved: CL={CL_sod2d}, CD={CD_sod2d}")
        Path(conv_flag).write_text("CONVERGED\n") 
        
        CL, CD = CL_sod2d, CD_sod2d
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
                ready = (env_i / "sod2d_ready.flag").exists()
                failed = (env_i / "sod2d_failed.flag").exists()
                conv = (env_i / "sod2d_conv.flag").exists()
                if ready and not failed and not conv:
                        target_envs.append(i)

            vprint(f"[Leader {env_id}] Tracking READY envs: {target_envs} (job_id={job_id})")

            while True:
                pending = []
                for i in target_envs:
                    env_i = shared_root / f"env_{i}"
                    if (env_i / "sod2d_conv.flag").exists():
                        continue
                    if (env_i / "sod2d_failed.flag").exists():
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

        #clean_env(work_dir)
        return CL, CD
    finally:
        os.chdir(cwd)