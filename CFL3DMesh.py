import numpy as np
import subprocess
from pathlib import Path


def construct2d(airfoil_file, Re_number, work_dir):
    """
    Run Construct2D inside work_dir, using airfoil_file.dat located there.

    Parameters
    ----------
    airfoil_file : str or Path
        Base name of airfoil file (without .dat) OR full path; the .dat
        is expected to be in work_dir.
    Re_number : float
        Chord Reynolds number.
    work_dir : str or Path
        Directory where Construct2D will run and write its output (.p3d).
    """

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    construct2d_path = "/gpfs/projects/bsc21/bsc545758/Construct2D_2.1.4/construct2d"

    airfoil_file = Path(airfoil_file)
    airfoil_basename = airfoil_file.stem  # strip .dat if present

    # Construct2D expects a file in the current working directory,
    # so we pass only the name and assume "<name>.dat" is in work_dir.
    airfoil_dat_name = f"{airfoil_basename}.dat"
    airfoil_dat_path = work_dir / airfoil_dat_name

    if not airfoil_dat_path.exists():
        raise FileNotFoundError(f"Airfoil file not found in {work_dir}: {airfoil_dat_name}")

    # Simulated interactive inputs to Construct2D
    input_lines = [
        airfoil_dat_name,                # airfoil geometry file
        "SOPT",
        "LESP",                          # Leading edge point spacing:
        "0.001",
        "RADI",                          # Farfield radius:
        "20",
        "QUIT",
        "VOPT",
        "JMAX",                          # Number of points in normal direction:
        "101",
        "TOPO",                          # Grid topology (O-GRID or C-GRID)
        "0GRD",
        "RECD",                          # Chord Reynolds number for y-plus:
        f"{Re_number}",
        "QUIT",
        "OOPT",
        "GDIM",                          # Output grid dimension:
        "2",
        "NPLN",                          # Number of planes for 3D output grid:
        "2",
        "DPLN",                          # Plane spacing for 3D output grid:
        "1.0",
        "QUIT",
        "GRID",
        "SMTH",
        "QUIT",
    ]
    input_str = "\n".join(input_lines) + "\n"

    # Run Construct2D in work_dir
    result = subprocess.run(
        [construct2d_path],
        input=input_str.encode("utf-8"),
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    print("=== Construct2D STDOUT ===")
    print(result.stdout.decode())

    print("=== Construct2D STDERR ===")
    print(result.stderr.decode())

    # Construct2D will write something like "<airfoil_basename>.p3d" in work_dir
    return work_dir / f"{airfoil_basename}.p3d"


def read_plot3d_2d(filename):
    filename = Path(filename)
    with open(filename, "r") as f:
        ni, nj = map(int, f.readline().split())

        # Read x values
        x_vals = []
        while len(x_vals) < ni * nj:
            x_vals += list(map(float, f.readline().split()))
        x2d = np.array(x_vals).reshape((ni, nj), order="F")  # (ni, nj)

        # Read y values
        y_vals = []
        while len(y_vals) < ni * nj:
            y_vals += list(map(float, f.readline().split()))
        y2d = np.array(y_vals).reshape((ni, nj), order="F")

    return x2d, y2d


def extrude_spanwise(x2d, y2d, z_vals):
    ni, nj = x2d.shape
    nk = len(z_vals)  # number of z (spanwise) layers

    x3d = np.zeros((ni, nj, nk))
    y3d = np.zeros_like(x3d)
    z3d = np.zeros_like(x3d)

    for k, zval in enumerate(z_vals):
        x3d[:, :, k] = x2d
        y3d[:, :, k] = y2d
        z3d[:, :, k] = zval

    return x3d, y3d, z3d  # shape: (ni, nj, nk)


def write_plot3d_ascii(filename, x, y, z):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    ni, nj, nk = x.shape
    with open(filename, "w") as f:
        f.write("1\n")  # One block
        f.write(f"{ni} {nj} {nk}\n")

        def write_array(arr):
            flat = arr.flatten(order="F")  # Fortran-style (i,j,k) → k fastest
            for i in range(0, len(flat), 5):
                f.write(" ".join(f"{v:.8e}" for v in flat[i : i + 5]) + "\n")

        write_array(x)
        write_array(y)
        write_array(z)


def cfl3d_mesh(filename, Re_number, work_dir):
    """
    Full mesh pipeline for a given airfoil, inside runs/env{i}.

    Parameters
    ----------
    filename : str or Path
        Base name of the airfoil (without .dat) or path to the .dat;
        the .dat must be located in work_dir.
    Re_number : float
        Reynolds number.
    work_dir : str or Path
        Directory for this environment, e.g. 'runs/env3'.
    """

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1) Run Construct2D to generate 2D Plot3D in work_dir
    output_file = construct2d(filename, Re_number, work_dir)

    # 2) Read 2D grid
    x2d, y2d = read_plot3d_2d(output_file)

    # 3) Define extrusion in z (spanwise direction)
    z_planes = np.linspace(0.0, 1.0, 2)  # can increase planes if needed

    x3d, y3d, z3d = extrude_spanwise(x2d, y2d, z_planes)

    # --- SWITCH i and k axes, then reverse all axes (as in your original) ---
    x3d = x3d.swapaxes(0, 2)[::-1, ::-1, ::-1]
    y3d = y3d.swapaxes(0, 2)[::-1, ::-1, ::-1]
    z3d = z3d.swapaxes(0, 2)[::-1, ::-1, ::-1]

    # 4) Write 3D extended Plot3D in work_dir with _ext suffix
    base = Path(filename).stem
    out_p3d = work_dir / f"{base}_ext.p3d"
    write_plot3d_ascii(out_p3d, x3d, y3d, z3d)

    print(f"3D mesh written to {out_p3d}")