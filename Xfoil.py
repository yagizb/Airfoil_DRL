
import os
import subprocess
import uuid
from pathlib import Path
import signal

# XFOIL analysis function
def analyze_airfoil(airfoil_file, angle_of_attack,Re_number,work_dir:Path):
    
    work_dir = Path(work_dir).resolve()

    results_file = work_dir / f"xfoil_results_{uuid.uuid4().hex}.dat"

    XFOIL_PATH = '/home/bsc/bsc545758/Xfoil/bin/xfoil'
     # Use xvfb-run for offscreen rendering
    command = ["xvfb-run", "-a", XFOIL_PATH]
    
    # Absolute input airfoil path
    airfoil_path = (work_dir / f"{airfoil_file}.dat").resolve()

    # Start XFOIL process
    xfoil = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, preexec_fn=os.setsid, 
        cwd=str(work_dir),
    )

    commands = f"""
    LOAD {airfoil_path.name}

    PANE

    PPAR
    N 301
    T 0.5
    
        
    OPER
    VISC {Re_number}
    MACH 0.15
    VPAR
    N 7

    ITER 201
    PACC 
    {results_file.name}
      
    ALFA {angle_of_attack}

    PACC
    
    QUIT
    """
    timeout_s = 30
    # Send commands to XFOIL and close stdin automatically with communicate()
    try:
        output, errors = xfoil.communicate(commands, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(xfoil.pid), signal.SIGKILL)
        except Exception:
            xfoil.kill()
        return None, None
    
    if errors:
        print("Errors:", errors)

    # Parse results
    try:
        with open(results_file, 'r') as f:
            lines = f.readlines()
        
        if lines:
            last_line = lines[-1]
            cl_value = float(last_line.split()[1])  # Assuming Cl is the second column
            cd_value = float(last_line.split()[2])  # Assuming Cd is the third column
            
            # Clean up temporary files
            if os.path.exists(results_file):
                    os.remove(results_file)
            return cl_value, cd_value
        else:
            print("Result file is empty.")
            return None, None
    except (FileNotFoundError, ValueError, IndexError) as e:
        print("Error reading results:", e)

    # Clean up temporary files
        if os.path.exists(results_file):
            os.remove(results_file)
    return None, None