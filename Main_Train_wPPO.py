# Main_Train.py (for example)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv,VecMonitor,DummyVecEnv
from pathlib import Path
import multiprocessing as mp

import config
from AirfoilEnv import AirfoilEnv
from AirfoilCallBacks import TensorboardAeroCallback
from Reset import reset_history

# --- GLOBAL SEED SETUP ---
trial_number = 33
SEED = int(config.SEED) + trial_number

def create_env(env_id: int, seed : int=0):
    def _init():
        # directory where the Python script lives
        SCRIPT_DIR = Path(__file__).resolve().parent
        # WORK_ROOT from config, relative to script locationç
        root = SCRIPT_DIR / getattr(config, "WORK_ROOT", "runs")
        work_dir = root / f"env_{env_id}"
        
        env= AirfoilEnv(
            env_id=env_id,
            n_envs=config.NUM_ENVS,
            work_dir=str(work_dir),    
            save_data=True,
            fidelity=config.FIDELITY,
            batch_id=0,
            angle_of_attack=config.AOA,
            Re_number=config.RE,
            scaling_factor=config.ACTION_SCALE,
            airfoil_file=config.AIRFOIL_FILE,
            max_steps=config.MAX_STEPS,
            max_no_improvement_episodes=config.MAX_NO_IMPROV,
            objective=config.OBJECTIVE,
        )
        # --- CRITICAL: per-env seeding ---
        env.reset(seed=seed + env_id)
        
        return env
    return _init

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True) ## is a Python multiprocessing setting that controls how new processes are created.
    
    reset_history(config.AIRFOIL_HISTORY_DIR, config.CL_CD_HISTORY_DIR)

    MODEL_BASENAME = f"airfoil_Re{int(config.RE/1e6)}M_AoA{int(config.AOA):02d}_{config.OBJECTIVE.upper()}"
    
    train_env = SubprocVecEnv(
        [create_env(i, seed=SEED) for i in range(config.NUM_ENVS)],
        start_method="spawn",
    )
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    train_env.seed(SEED)
        
    model = PPO(
        "MlpPolicy",
        train_env,
        batch_size=config.BATCH_SIZE,
        clip_range=config.CLIP_RANGE,
        ent_coef=config.ENT_COEF,
        gae_lambda=config.GAE_LAMBDA,
        gamma=config.GAMMA,
        learning_rate=config.LEARNING_RATE,
        max_grad_norm=config.MAX_GRAD_NORM,
        n_epochs=config.N_EPOCHS,
        n_steps=config.N_STEPS,
        vf_coef=config.VF_COEF,
        use_sde=config.USE_SDE,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=config.VERBOSE,
        seed=SEED,
        tensorboard_log="./tensorboard_logs/",
    )

    cb = TensorboardAeroCallback(log_every=100)

    model.learn(total_timesteps=config.TOTAL_TIMESTEPS,callback=cb, tb_log_name=f"XFOIL001_PPOtr{trial_number}_MaxCL")
    print("Learned")

    ms = MODEL_BASENAME + f"_XFOIL001_PPOtr{trial_number}_MaxCL"
        
    model.save(ms)
    train_env.save(ms + ".pkl")
    train_env.close()
    print("Model saved")