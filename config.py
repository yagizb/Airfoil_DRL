import torch
import numpy as np
import random,os

# ---------- General ----------
SEED            = 42
# ---------- Environment ----------
NUM_ENVS        = 32
OBJECTIVE       = "cl"      # "cl" or "cl_cd" 
FIDELITY        = 0         # 0=Xfoil, 1=RANS, 2=LES
TRAIN           = "000"      ## 0=Xfoil, 1=RANS, 2=LES 
AOA             = 0.0       # Angle of attack (degrees)
RE              = 3.0E6     # Reynolds Number
ACTION_SCALE    = 0.0050     # To scale the action
AIRFOIL_FILE    = "airfoil" # Airfoil data    
MAX_STEPS       = 1         # ??????? Episode always truncated at step 1: max_steps=1 makes every episode end immediately. 
                            # max_steps = 1 → single-step bandit, good for one-shot optimization. PPO chooses an action ,  evaluate it once, reward = CL/CD (or CL), and the episode ends
                            # max_steps = 10 The agent gets multiple actions in one episode. Now the environment behaves more like a sequential design problem,
MAX_NO_IMPROV   = 20000

WORK_ROOT = "runs"
BASENAME = (f"Re{int(RE/1.e06)}M_AoA{int(AOA):02d}_{OBJECTIVE.upper()}")
AIRFOIL_HISTORY_DIR = (f"airfoil_history_{BASENAME}")
CL_CD_HISTORY_DIR   = (f"cl_cd_history_{BASENAME}")
NUM_CONTROL_POINTS  = 18    # Bezier Curves, control points

# ---------- PPO Training ---------
#TARGET_KL       = 0.03
LEARNING_RATE   = 0.000096
CLIP_RANGE      = 0.1
GAMMA           = 0.9582               # Discount factor
GAE_LAMBDA      = 0.95
SDE_SAMPLE_FREQ = 5                 # resample noise every ** steps         
USE_SDE         = False              # (SB3), use_sde controls whether the policy uses State-Dependent Exploration (SDE) instead of the default action noise.
                                    # use_sde=True is most useful for continuous action spaces. (like control points of an airfoil).
                                     #is set to False (which is the default), the policy will use unstructured Gaussian noise for exploration
# Roll_out        = 2                 # Collects experience from the environment using current policy. 
# N_STEPS_cal     = 2048/(NUM_ENVS*Roll_out)                                
N_STEPS         = 256
BATCH_SIZE      = 1024                # Sub-chunks of rollout used during network training.
N_EPOCHS        = 5                # How many passes over the same rollout data.


MAX_GRAD_NORM   = 0.7434
ENT_COEF        = 0.00017  # 0.02 - 0.05 - 0.1             # encourage exploration --- 0.025 ent_coef=0.0,   # ← default                        
VF_COEF         = 0.5                # how much the critic matters in training ---- 0.1  vf_coef=0.5,    # ← default
VERBOSE         = 1                  # =0 → no output (silent) , =1 → training progress printed (timesteps, FPS, reward, losses).
TOTAL_TIMESTEPS = N_STEPS * NUM_ENVS * 1    ## Rollout size = N_STEPS × NUM_ENVS = 2048 transitions     
TRAIN_PHASE     = 0                 # =0 first learn, >0 continue to train 
# ---------- SAC Training ---------
TAU             = 0.005
LEARNING_STARTS = 100   
GRADIENT_STEPS  = 2
TRAIN_FREQ      = 4 
# ---- SAC-compatible training ----
BUFFER_SIZE       = 88907

# ---------- Logging / Saving ----------
MAX_EPISODES    = 501
SAVE_INTERVAL   = 1001

MODEL_BASENAME = f"airfoil_Re{int(RE/1e6)}M_AoA{int(AOA):02d}_{OBJECTIVE.upper()}"
LOG_DIR = "tb_logs"                  # TensorBoard log dir
BEST_DIR = os.path.join("./logs", MODEL_BASENAME)
PATIENCE_EVALS = 100000 
MIN_EVALS_BEFORE_CHECK = 32
EVAL_FREQ = 16
N_EVAL_EPISODES = 10

### 688 e kadar devam etmiş, = 512  176/16 =11
# LOAD_MODE = "predict"               # "train" "predict"  
 
def set_global_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)