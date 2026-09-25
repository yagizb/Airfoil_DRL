from typing import Tuple
from pathlib import Path


def count_sod2d_ready_failed(shared_root: Path, n_envs: int) -> Tuple[int, int]:
    """
    Count how many env_i have:
      - sod2d_ready.flag
      - sod2d_failed.flag

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

        ready_flag = env_dir / "sod2d_ready.flag"
        failed_flag = env_dir / "sod2d_failed.flag"

        # IMPORTANT: failed overrides ready
        if failed_flag.exists():
            n_failed += 1
        elif ready_flag.exists():
            n_ready += 1

    return n_ready, n_failed