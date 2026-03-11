import time
import os
from pathlib import Path
from typing import Optional


def _safe_unlink(p: Path, vprint=None):
    try:
        p.unlink()
    except FileNotFoundError:
        return
    except Exception as e:
        if vprint:
            vprint(f"Could not remove {p}: {e}")


def _try_acquire_lock(lock_path: Path) -> bool:
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def _release_lock(lock_path: Path):
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _read_int_flag(p: Path) -> Optional[int]:
    try:
        return int(p.read_text().strip())
    except Exception:
        return None


def _write_int_flag_atomic(p: Path, val: int):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(f"{val}\n")
    os.replace(tmp, p)


def _leader_is_failed(shared_root: Path, leader_id: Optional[int]) -> bool:
    if leader_id is None:
        return True
    env_leader = shared_root / f"env_{leader_id}"
    return (env_leader / "cfl3d_failed.flag").exists()


def _pick_new_leader_from_ready(shared_root: Path, n_envs: int) -> Optional[int]:
    for i in range(n_envs):
        env_i = shared_root / f"env_{i}"
        ready = (env_i / "cfl3d_ready.flag").exists()
        failed = (env_i / "cfl3d_failed.flag").exists()
        if ready and not failed:
            return i
    return None


def _wait_done_flag(done_flag: Path, timeout: float = 3600.0, poll: float = 2.0, vprint=None):
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


def _fail_and_return(
    *,
    env_id: int,
    reason: str,
    failed_flag: Path,
    status_flag: Path,
    done_flag: Path,
    leader_flag: Path,
    cleanup_lock: Path,
    takeover_lock: Path,
    is_leader: bool,
    eps: float,
    vprint=None,
):
    if vprint:
        vprint(f"[Env {env_id}] FAIL -> {reason}")

    try:
        failed_flag.write_text(f"{reason}\n")
    except Exception:
        pass

    try:
        status_flag.write_text(f"{reason}\n")
    except Exception:
        pass

    # If leader fails, wake everybody up and remove shared leadership state
    if is_leader:
        try:
            done_flag.write_text(f"{reason}\n")
        except Exception:
            pass

        _safe_unlink(leader_flag, vprint=vprint)
        _safe_unlink(cleanup_lock, vprint=vprint)
        _safe_unlink(takeover_lock, vprint=vprint)

    return eps, eps