import json
from sympy import true

def write_BluffBodySolverIncomp_json(
    base_name,
    loadRestartFile,
    work_dir,
    config
):

    data = {
        "type": "BluffBodySolverIncomp",

        "mesh_h5_file_path": "",
        "mesh_h5_file_name": f"{base_name}_per",

        "results_h5_file_path": "",
        "results_h5_file_name": "results",

        "final_istep": config["final_istep"],

        "doGlobalAnalysis": False,
        "doTimerAnalysis": False,

        "save_logFile_first": 1,
        "save_logFile_step": 10,

        "save_resultsFile_first": 1,
        "save_resultsFile_step": 50000,

        "save_restartFile_first": 1,
        "save_restartFile_step": 10000,

        "loadRestartFile":loadRestartFile,
        "restartFile_to_load": 1,

        "continue_oldLogs": False,

        "saveAvgFile": True,
        "loadAvgFile": False,

        "saveSurfaceResults": False,
        
        "flag_trip_element_upp": True,
        "x_trip_o_upp": 0.05,
        "y_trip_o_upp": 0.035,
        "l_trip_x_upp": 0.003,
        "l_trip_y_upp": 0.003,
        
        "flag_trip_element_low": True,
        "x_trip_o_low": 0.05,
        "y_trip_o_low": -0.035,
        "l_trip_x_low": 0.003,
        "l_trip_y_low": 0.003,
        
        "flag_fsp_jacobian": False,

        "flag_type_les": "les_type_vreman",

        "flag_walave": True,
        "period_walave": 1.0,

        "flag_type_wmles": config["flag_type_wmles"],

        "flag_lps_stab": True,

        "wmles_extype": config["wmles_extype"],
        "wmles_walex": config['wmles_walex'],

        "maxIter": 20,
        "tol": 0.001,

        "cfl_conv": 0.55,

        "v0": 1.0,
        "delta": 1.0,
        "rho0": 1.0,

        "Re": config['Re_number'],
        "aoa": config['angle_of_attack'],

        "bouCodes": [
            {"id": 4,"bc_type": "bc_type_slip_wall_model"},
            {"id": 5,"bc_type": "bc_type_far_field"},
            {"id": 6,"bc_type": "bc_type_outlet_incomp"}
        ],

        "buffer": 
        [
            {"type": "bc_outlet","min": 45,"size": 5}
        ]
    }

    with open(work_dir / "BluffBodySolverIncomp.json", "w") as f:
        json.dump(data, f, indent=4)

    print("Wrote BluffBodySolverIncomp.json")