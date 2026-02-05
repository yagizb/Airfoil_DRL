# Import necessary modules
import os
import json
import uuid

def gen_geo(airfoil_file):

    # Step 1: Read `airfoil.dat`
    #with open(file_path_dat, 'r') as file:
    with open(airfoil_file, 'r') as file:
        airfoil_data = file.readlines()

    # Step 2: Reverse the data (excluding the first line if it's a header)
    header = airfoil_data[0] if not airfoil_data[0][0].isdigit() else None
    actual_data = airfoil_data[1:] if header else airfoil_data
    reversed_data_lines = actual_data[::-1]

    # Step 3: Add the first point as '1.000 0.000' to the reversed data
    modified_data_lines = ["1.000 0.000\n"] + reversed_data_lines
    if header:
        modified_data_lines.insert(0, header)

    # Step 4: Read `general.geo`
    file_path_geo = 'general.geo'

    with open(file_path_geo, 'r') as file:
        geo_data = file.readlines()

    # Step 5: Update `naca_hex_mono.geo` to replace all surface points with new data, only up to line 316
    geo_data_updated = []
    in_surface_points_section = False
    surface_points_start_index = None

    for idx, line in enumerate(geo_data):
        if idx > 316:
            # After line 316, do not make any changes
            geo_data_updated.append(line)
            continue
        
        if "//NACA0012 surface points" in line:
            geo_data_updated.append(line)
            in_surface_points_section = True
            surface_points_start_index = idx + 1
            # Add the new data points including the added first point
            for idx, data_line in enumerate(modified_data_lines):
                x_value, y_value = data_line.strip().split()
                point_definition = f"Point({idx + 1}) = {{ {x_value}, {y_value}, 0., ms1}};\n"
                geo_data_updated.append(point_definition)
        elif in_surface_points_section and line.startswith("Point"):
            # Skip the old Point definitions
            continue
        else:
            geo_data_updated.append(line)

    # Step 6: Write the updated `......geo` file
    output_geo_path_final = f"airfoil_{uuid.uuid4().hex}"

    with open(f"{output_geo_path_final}.geo", 'w') as file:
        file.writelines(geo_data_updated)

    print(f"Updated .geo file written to: {output_geo_path_final}")

    return str(output_geo_path_final)

def h5file(output):
    """
    Generates a batch script for SOD2D mesh preparation and writes it to a file.

    Parameters:
        output (str): The base name of the output geometry file (without extension).

    Returns:
        str: The unique identifier for the generated mesh file.
    """
    try:
        # Generate unique identifiers for file names
        h5file = f"h5_{uuid.uuid4().hex}"
        #mshgen = f"mshgen_{uuid.uuid4().hex}"
        logfile = f"logfile_{uuid.uuid4().hex}"
        errfile = f"errfile_{uuid.uuid4().hex}"

        # Define the script lines
        lines = [
            "#!/bin/bash",
            "#SBATCH --job-name=mshgen",
            f"#SBATCH --chdir=.",
            f"#SBATCH -o {logfile}.out",
            f"#SBATCH -e {errfile}.err",
            "#SBATCH --ntasks=1",
            "##SBATCH --cpus-per-task=28",
            "#SBATCH --time=0:29:59",
            "#SBATCH --qos=gp_debug",
            "#SBATCH --account=bsc21",
            "#SBATCH --exclusive",
            "module purge",
            "module load intel/2023.2.0 mkl/2023.2.0 impi/2021.10.0 hdf5/1.14.1-2-gcc python/3.12.1 GMSH",
            f"gmsh {output}.geo -o {output}.msh -0",
            f"srun -n 1 python3 gmsh2sod2d.py {output} -p 6 -r 4",
        ]

        # Write the script to a file
        script_filename = f"{h5file}.sh"
        with open(script_filename, "w") as file:
            file.write("\n".join(lines) + "\n")

        print(f"Batch script written to: {script_filename}")
        return str(h5file)

    except Exception as e:
        print(f"Error writing batch script: {e}")
        return None

def input_file(output):
    """
    Generates an input configuration `.dat` file for the simulation.

    Parameters:
        output (str): The base name of the output geometry or mesh file.

    Returns:
        str: The unique identifier for the generated input file.
    """
    try:
        # Generate a unique identifier for the input file
        inputf = f"input_{uuid.uuid4().hex}"
        
        # Define the configuration lines
        lines = [
            'gmsh_filePath ""',  # Assuming files are in the current directory
            f'gmsh_fileName "{output}"',  # Geometry file name
            'mesh_h5_filePath ""',  # Assuming output files stay in the current directory
            f'mesh_h5_fileName "{output}"',  # Mesh file name
            'num_partitions 4',  # Number of partitions for parallel processing
            'lineal_output 1',  # Enable linear output
            'eval_mesh_quality 0',  # Disable mesh quality evaluation
        ]

        # Write the configuration to a file
        config_filename = f"{inputf}.dat"
        with open(config_filename, "w") as file:
            file.write("\n".join(lines) + "\n")

        print(f"Input configuration file written to: {config_filename}")
        return str(inputf)

    except Exception as e:
        print(f"Error writing input configuration file: {e}")
        return None

def partition(input_file):
    """
    Generates a batch script for partitioning the mesh using the tool_meshConversorPar utility.

    Parameters:
        input_file (str): The name of the input configuration file (without extension).

    Returns:
        str: The unique identifier for the toolMCP batch job.
    """
    try:
        # Generate unique identifiers for filenames
        partition = f"partition_{uuid.uuid4().hex}"
        #toolMCP = f"toolMCP_{uuid.uuid4().hex}"
        logfile = f"logfile_{uuid.uuid4().hex}"
        errfile = f"errfile_{uuid.uuid4().hex}"

        # Define the script lines
        lines = [
            "#!/bin/bash",
            "#SBATCH --job-name=toolMCP",
            "#SBATCH --chdir=.",
            f"#SBATCH -o {logfile}.out",
            f"#SBATCH -e {errfile}.err",
            "#SBATCH --ntasks=4",
            "##SBATCH --cpus-per-task=28",
            "#SBATCH --time=0:29:59",
            "#SBATCH --qos=gp_debug",
            "#SBATCH --account=bsc21",
            "#SBATCH --exclusive",
            "module purge",
            "module load openmpi/4.1.5-gcc ucx/1.15.0 hdf5/1.14.1-2-gcc-ompi cmake",
            f"mpirun -np 4 ./tool_meshConversorPar {input_file}.dat",
        ]

        # Write the script to a file
        script_filename = f"{partition}.sh"
        with open(script_filename, "w") as file:
            file.write("\n".join(lines) + "\n")

        print(f"Partition script written to: {script_filename}")
        return str(partition)

    except Exception as e:
        print(f"Error writing partition script: {e}")
        return None

def SolverIncomp(output_geo):
    file_name = "BluffBodySolverIncomp.json"

    # Check if the file exists
    if os.path.isfile(file_name):
        os.remove(file_name)
        print(f"File '{file_name}' has been removed.")
    else:
        print(f"File '{file_name}' does not exist. Nothing to remove.")

    try:
        # Define the JSON structure using lines
        lines = []
        lines.append('{')
        lines.append('    "type": "BluffBodySolverIncomp",')
        lines.append('    "mesh_h5_file_path": "",')
        lines.append(f'    "mesh_h5_file_name": "{output_geo}",')
        lines.append('    "results_h5_file_path": "",')
        lines.append(f'    "results_h5_file_name": "res_{output_geo}",')
        lines.append('    "final_istep": 10001,')
        lines.append('    "doGlobalAnalysis": true,')
        lines.append('    "doTimerAnalysis": false,')
        lines.append('    "save_logFile_first": 1,')
        lines.append('    "save_logFile_step": 10,')
        lines.append('    "save_resultsFile_first": 1,')
        lines.append('    "save_resultsFile_step": 1000,')
        lines.append('    "save_restartFile_first": 1,')
        lines.append('    "save_restartFile_step": 1000,')
        lines.append('    "loadRestartFile": false,')
        lines.append('    "restartFile_to_load": 1,')
        lines.append('    "continue_oldLogs": false,')
        lines.append('    "saveAvgFile": true,')
        lines.append('    "loadAvgFile": false,')
        lines.append('    "saveSurfaceResults": true,')
        lines.append('    "flag_les": 1,')
        lines.append('    "maxIter": 20,')
        lines.append('    "tol": 0.001,')
        lines.append('    "flag_walave": true,')
        lines.append('    "period_walave": 1.0,')
        lines.append('    "cfl_conv": 0.95,')
        lines.append('    "v0": 1.0,')
        lines.append('    "delta": 1.0,')
        lines.append('    "rho0": 1.0,')
        lines.append('    "Re": 300000.0,')
        lines.append('    "aoa": 0.0,')
        lines.append('    "bouCodes": ')
        lines.append('    [')
        lines.append('        {"id": 1, "bc_type": "bc_type_slip_wall_model"},')
        lines.append('        {"id": 2, "bc_type": "bc_type_far_field"},')
        lines.append('        {"id": 3, "bc_type": "bc_type_far_field"},')
        lines.append('        {"id": 4, "bc_type": "bc_type_outlet_incomp"},')
        lines.append('        {"id": 5, "bc_type": "bc_type_far_field"}')
        lines.append('    ],')
        lines.append('    "buffer": ')
        lines.append('    [')
        lines.append('        {"type": "east", "min": 15, "size": 5},')
        lines.append('        {"type": "west", "min": -15, "size": 5},')
        lines.append('        {"type": "north", "min": 15, "size": 5},')
        lines.append('        {"type": "south", "min": -15, "size": 5}')
        lines.append('    ]')
        lines.append('}')

        # Write the JSON to the file
        with open(file_name, "w") as file:
            file.write("\n".join(lines))

        print(f"Configuration script written to: {file_name}")
        return file_name

    except Exception as e:
        print(f"Error writing configuration script: {e}")
        return None

