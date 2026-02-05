from pathlib import Path
import shutil
import shutil
from pathlib import Path

def clean_env(env_dir):
    """
    Delete files inside ONE env directory only.
    `keep` are filenames to preserve (optional).
    """
    env_dir = Path(env_dir)
    if not env_dir.exists():
        return

    for item in env_dir.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception as e:
            print(f"[clean_env] WARNING: Could not remove {item}: {e}")

def clean_all_envs(root="runs"):
    """
    Delete ALL files inside runs/env_i directories.
    Keeps the directories themselves.
    """
    root = Path(root)

    if not root.exists():
        print(f"[cleanup] Root folder {root} does not exist.")
        return

    # Find all env dirs
    env_dirs = sorted(root.glob("env_*"))

    for env_dir in env_dirs:
        if not env_dir.is_dir():
            continue

        # Iterate over contents
        for item in env_dir.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                print(f"  [WARNING] Could not remove {item}: {e}")

def copy_main_inputs(shared_root, dst_dir):
    """
    Copies cfl3d.inp_1 and split.inp from runs/main/
    into runs/env_{env_id}/
    """
    main_root = shared_root.parent
    src_dir = main_root / "main"

    # Make sure destination exists
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Files to copy
    files = ["cfl3d.inp_1", "split.inp"]

    for fname in files:
        src = src_dir / fname
        dst = dst_dir / fname

        if src.exists():
            shutil.copy(src, dst)
        else:
            print(f"Warning: Missing file in env_main: {src}")

def update_cfl3d_inp(
    folder,
    file_name,
    new_alpha=None,
    new_reynolds=None
):
    """
    Update ALPHA and REUE,MIL inside runs/env{i}/cfl3d.inp safely.

    folder: path to environment directory, e.g. 'runs/env3'
    """

    folder = Path(folder)
    file_path = folder / file_name

    if not file_path.exists():
        raise FileNotFoundError(f" File not found: {file_path}")

    # Read
    with open(file_path, "r") as f:
        lines = f.readlines()

    updated = False

    for i, line in enumerate(lines):
        # Find the line AFTER the header containing "XMACH ... REUE,MIL"
        if "XMACH" in line and "REUE,MIL" in line:
            target = i + 1
            values = lines[target].split()

            if len(values) < 4:
                raise ValueError(" Unexpected format in ALPHA/REUE line.")

            # Modify ALPHA
            if new_alpha is not None:
                values[1] = f"{new_alpha:.3f}"

            # Modify Reynolds (stored in millions)
            if new_reynolds is not None:
                values[3] = f"{new_reynolds / 1e6:.4f}"

            # Re-assemble line with nice spacing
            lines[target] = "   " + "    ".join(values) + "\n"

            updated = True
            break

    if not updated:
        raise RuntimeError("Could not find XMACH / REUE,MIL block in file.")

    # Write back
    with open(file_path, "w") as f:
        f.writelines(lines)

    # Print confirmation
    msg = []
    if new_alpha is not None:
        msg.append(f"ALPHA={new_alpha}")
    if new_reynolds is not None:
        msg.append(f"RE={new_reynolds:.3e}")

    print(f"Updated {' & '.join(msg)} in {file_path}")