import torch
import numpy as np
import random
import json
from pathlib import Path

def read_flow_config(config_file="flow_config.json"):
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / config_file

    with open(config_path, "r") as f:
        flow_config = json.load(f)

    return flow_config

flow_config = read_flow_config()
# ---------- General ----------
SEED            = 42
# ---------- Environment ----------
NUM_ENVS        = 32
OBJECTIVE       = "cl"      # "cl" or "cl_cd" 
FIDELITY        = 0         # 0=Xfoil, 1=RANS, 2=LES
TRAIN           = "000"      ## 0=Xfoil, 1=RANS, 2=LES 
AOA             = 0       # Angle of attack (degrees)
RE              = 3.0E6     # Reynolds Number
ACTION_SCALE    = 0.0050     # To scale the action
AIRFOIL_FILE    = "airfoil" # Airfoil data    
MAX_STEPS       = 1         # ??????? Episode always truncated at step 1: max_steps=1 makes every episode end immediately. 
                            # max_steps = 1 → single-step bandit, good for one-shot optimization. PPO chooses an action ,  evaluate it once, reward = CL/CD (or CL), and the episode ends
                            # max_steps = 10 The agent gets multiple actions in one episode. Now the environment behaves more like a sequential design problem,
MAX_NO_IMPROV   = 20000

WORK_ROOT = "runs"
CFL3D_SCRIPT = "Main_airfoil_cfl3d.sh"
SOD2D_SCRIPT = "Main_airfoil_sod2d.sh"
BASENAME = (f"Re{int(RE/1.e06)}M_AoA{int(AOA):02d}_{OBJECTIVE.upper()}")
AIRFOIL_HISTORY_DIR = (f"airfoil_history_{BASENAME}")
CL_CD_HISTORY_DIR   = (f"cl_cd_history_{BASENAME}")
NUM_CONTROL_POINTS  = 18    # Bezier Curves, control points

# ---------- SAC Training ---------
BATCH_SIZE      = 1024                         # Sub-chunks of rollout used during network training
BUFFER_SIZE     = 88907
ENT_COEF        = 0.00017056535745467157       # 0.02 - 0.05 - 0.1 # encourage exploration --- 0.025 ent_coef=0.0,   # ← default
GAMMA           = 0.9843497124285968 #0.9581661681401709           # Discount factor
GRADIENT_STEPS  = 2
LEARNING_RATE   = 0.00009623046369670056
TAU             = 0.0031577581109901317
TRAIN_FREQ      = 4 
LEARNING_STARTS = 100   

# Roll_out        = 2                 # Collects experience from the environment using current policy. 
# N_STEPS_cal     = 2048/(NUM_ENVS*Roll_out)                                
N_STEPS         = 64
               # =0 → no output (silent) , =1 → training progress printed (timesteps, FPS, reward, losses).
TOTAL_TIMESTEPS = N_STEPS * NUM_ENVS * 1    ## Rollout size = N_STEPS × NUM_ENVS = 2048 transitions     
TRAIN_PHASE     = 0                 # =0 first learn, >0 continue to train 
VERBOSE         = 1

# ---------- Logging / Saving ----------
MAX_EPISODES    = 2001
SAVE_INTERVAL   = 3001

 
def set_global_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)