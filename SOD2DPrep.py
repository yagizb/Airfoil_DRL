from pathlib import Path
import shutil

def copy_sod2d_main_inputs(dst_dir, shared_root):
    dst_dir = Path(dst_dir).resolve()
    shared_root = Path(shared_root).resolve()

    main_root = shared_root.parent
    src_dir = main_root / "main_sod2d"

    print("CWD     :", Path.cwd())
    print("src_dir :", src_dir)
    print("dst_dir :", dst_dir)

    files = [
        "tool_meshConversorPar",
        "gmsh2sod2d.py",
        "interpolate.py",
        "sod2d_p2",
        "p3d2gmsh.py",
        "mn5_bind.sh",
        "airfoil.sh",
    ]

    for fname in files:
        src = src_dir / fname
        dst = dst_dir / fname

        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied: {fname}")
        else:
            print(f"Warning: Missing file: {src}")