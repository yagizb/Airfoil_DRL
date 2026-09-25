import numpy as np
import subprocess
import time
import json
from pathlib import Path

def construct2d(airfoil_file,Re_number,work_dir,JMAX,RADI,LES_YPLS,MESH_TYPE,max_tries=33, sleep_time=1.0):
    work_dir = Path(work_dir)
    construct2d_path = "/gpfs/projects/bsc21/bsc545758/Construct2D_2.1.4/construct2d"

    airfoil_file = Path(airfoil_file)
    airfoil_basename = airfoil_file.stem  # strip .dat if present

    # Construct2D expects a file in the current working directory,
    # so we pass only the name and assume "<name>.dat" is in work_dir.
    airfoil_dat_name = f"{airfoil_basename}.dat"
    airfoil_dat_path = work_dir / airfoil_dat_name

    if not airfoil_dat_path.exists():
        raise FileNotFoundError(f"Airfoil file not found in {work_dir}: {airfoil_dat_name}")
    
    input_lines = [
        f"{airfoil_file}.dat",
        "SOPT",
        "LESP",
        "0.001",
        "RADI",
        f"{RADI}",
        "NWKE",
        "100",
        "QUIT",
        "VOPT",
        "JMAX",
        f"{JMAX}",
        "TOPO",
        f"{MESH_TYPE}",
        "YPLS",
        f"{LES_YPLS}",
        "QUIT",
        "OOPT",
        "GDIM",
        "2",
        "NPLN",
        "2",
        "DPLN",
        "1.0",
        "QUIT",
        "GRID",
        "SMTH",
        "QUIT",
    ]

    input_str = "\n".join(input_lines) + "\n"

    expected_file = work_dir / f"{airfoil_file}.p3d"

    # Retry loop
    for attempt in range(1, max_tries + 1):
        print(f"\n=== Construct2D attempt {attempt}/{max_tries} ===")

        # 🧹 Clean previous outputs
        if expected_file.exists():
            expected_file.unlink()

        for f in Path(work_dir).glob("*.p3d"):
            f.unlink()

        # Run Construct2D
        result = subprocess.run(
            [construct2d_path],
            input=input_str,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )

        print("RETURN CODE:", result.returncode)

        # Print logs only on last attempt
        if attempt == max_tries:
            print("=== STDOUT ===")
            print(result.stdout)
            print("=== STDERR ===")
            print(result.stderr)

        time.sleep(sleep_time)

        # Check if file exists
        if expected_file.exists():
            print(f"SUCCESS: {expected_file.name} created on attempt {attempt}")
            return f"{airfoil_file}.p3d"

        print("No .p3d file found, retrying...")

    #  If all attempts fail
    raise RuntimeError(
        f"Construct2D failed after {max_tries} attempts to generate {airfoil_file}.p3d"
    )

def read_plot3d_2d(filename):
    with open(filename, 'r') as f:
        ni, nj = map(int, f.readline().split())

        # Read x values
        x_vals = []
        while len(x_vals) < ni * nj:
            x_vals += list(map(float, f.readline().split()))
        x2d = np.array(x_vals).reshape((ni, nj), order='F')  # shape: (ni, nj)

        # Read y values
        y_vals = []
        while len(y_vals) < ni * nj:
            y_vals += list(map(float, f.readline().split()))
        y2d = np.array(y_vals).reshape((ni, nj), order='F')

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
    ni, nj, nk = x.shape
    with open(filename, 'w') as f:
        f.write("1\n")  # One block
        f.write(f"{ni} {nj} {nk}\n")

        def write_array(arr):
            flat = arr.flatten(order='F')  # Fortran-style (i,j,k) → k fastest
            for i in range(0, len(flat), 5):
                f.write(" ".join(f"{v:.8e}" for v in flat[i:i+5]) + "\n")

        write_array(x)
        write_array(y)
        write_array(z)

def sod2d_mesh(filename,Re_number,work_dir,JMAX,RADI,z_len,z_plane,LES_YPLS,MESH_TYPE):
    
    work_dir = Path(work_dir)
    
    # 1) Run Construct2D to generate 2D Plot3D in work_dir
    output_file=construct2d(filename,Re_number,work_dir,JMAX,RADI,LES_YPLS,MESH_TYPE)
    # 2) Read 2D grid
    x2d, y2d = read_plot3d_2d(output_file)

    # 3) Define extrusion in z (spanwise direction)
    z_plane = int(z_plane)
    z_len = float(z_len)
    z_planes = np.linspace(0.0, z_len, z_plane)  # You can set more layers if needed

    x3d, y3d, z3d = extrude_spanwise(x2d, y2d, z_planes)

    # 4) Write 3D extended Plot3D in work_dir with _ext suffix
    base = Path(filename).stem
    out_p3d = work_dir / f"{base}_ext.p3d"
    write_plot3d_ascii(out_p3d, x3d, y3d, z3d)
    
def write_ext_nmf(airfoil_file, work_dir,idim, jmax, z_planes):
    work_dir = Path(work_dir)
    
    nmf_file = work_dir / f"{airfoil_file}_ext.nmf"
    idim=int(idim)
    jmax=int(jmax)
    z_planes=int(z_planes)

    with open(nmf_file, "w") as f:
        f.write("# ==================== Neutral Map File generated by Construct2D ====================\n")
        f.write("# ==================== ========================================= ====================\n")
        f.write("# Block#   IDIM   JDIM   KDIM\n")
        f.write("# -----------------------------------------------------------------------------------\n")
        f.write("       1\n\n")
        f.write(f"       1    {idim:3d}    {jmax:3d}      {z_planes:2d}\n\n")
        f.write("# ===================================================================================\n")
        f.write("# Type         B1  F1     S1   E1     S2   E2    B2  F2     S1   E1     S2   E2  Swap\n")
        f.write("#                                                              Compute forces (walls)\n")
        f.write("# -----------------------------------------------------------------------------------\n")

        f.write(f"SYMMETRY-Y    1    1      1    {idim}      1   {jmax}\n")
        f.write(f"SYMMETRY-Y    1    2      1    {idim}      1   {jmax}\n")
        f.write(
            f"ONE_TO_ONE    1    3      1    {jmax}      1     {z_planes}   "
            f"1   4      1    {jmax}     1  {z_planes} FALSE\n"
        )
        f.write(f"AIRFOIL       1    5      1      {z_planes}      1   {idim}                                TRUE\n")
        f.write(f"FARFIELD      1    6      1      {z_planes}      1   {idim}\n")

    print(f"Wrote {nmf_file}")

def write_geo_file(airfoil_file, work_dir, span_z, porder, per_surface_ID):

    work_dir = Path(work_dir)
    base_name = airfoil_file.replace(".dat", "")

    geo_text = f"""
    span_z = {span_z};

    //+ Set the output mesh file version
    Mesh.MshFileVersion = 2.2;

    Merge "{base_name}.msh";

    // Make all physical groups to Zero
    Delete Physicals;

    // Create physical groups
    Physical Surface("Periodic",{per_surface_ID}) = {{2,3}};
    Physical Surface("b5-FARFIELD") = {{5}};
    Physical Surface("bc_outlet") = {{6}};
    Physical Surface("b4-AIRFOIL") = {{4}};

    Physical Volume("mesh") = {{1}};

    //+ Options controlling mesh generation/
    Mesh.ElementOrder = {porder};

    Mesh 3;

    Recombine Volume {{1}};

    Periodic Surface {{3}} = {{2}} Translate {{0,0,span_z}};

    Save "{base_name}_per.msh";
    """

    with open(work_dir / "airfoil_per.geo", "w") as f:
        f.write(geo_text)

    print("Wrote airfoil_per.geo")
    
def write_partition_input_json(base_name, work_dir, n_partitions):

    input_data = {
        "gmsh_filePath": "",
        "gmsh_fileName": f"{base_name}_per",

        "mesh_h5_filePath": "",
        "mesh_h5_fileName": f"{base_name}_per",

        "num_partitions": int(n_partitions),
        "eval_mesh_quality": 0,
        "lineal_output": False,
        "uns_per_links": False
    }

    with open(work_dir / "input.json", "w") as f:
        json.dump(input_data, f, indent=4)

    print("Wrote input.json")