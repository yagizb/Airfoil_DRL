# Main_Train.py (for example)
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv,VecMonitor,DummyVecEnv
from pathlib import Path
import multiprocessing as mp

import config
from AirfoilEnv import AirfoilEnv
from AirfoilCallBacks import TensorboardAeroCallback
from Reset import reset_history


def create_env(env_id: int):
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
        return env
    return _init

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    trial_number = 88 #### optimum hyperparameters found at trial ?????, see Optuna_Results_Summary.txt
    SEED = int(config.SEED) + trial_number
    config.set_global_seeds(config.SEED)
    reset_history(config.AIRFOIL_HISTORY_DIR, config.CL_CD_HISTORY_DIR)

    MODEL_BASENAME = f"airfoil_Re{int(config.RE/1e6)}M_AoA{int(config.AOA):02d}_{config.OBJECTIVE.upper()}"
    
    train_env = SubprocVecEnv(
        [create_env(i) for i in range(config.NUM_ENVS)],
        start_method="spawn",
    )
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    train_env.seed(SEED)
        
    model = SAC(
        "MlpPolicy",
        train_env,
        batch_size=config.BATCH_SIZE,
        buffer_size=config.BUFFER_SIZE,
        ent_coef=config.ENT_COEF,
        gamma=config.GAMMA,
        gradient_steps=config.GRADIENT_STEPS,
        learning_rate=config.LEARNING_RATE,
        tau=config.TAU,
        train_freq=config.TRAIN_FREQ,
        learning_starts=config.LEARNING_STARTS,
        seed=SEED,
        verbose=1,
    )

    cb = TensorboardAeroCallback(log_every=100)

    model.learn(total_timesteps=config.TOTAL_TIMESTEPS,callback=cb, tb_log_name="Roll_001")
    print("Learned")

    ms = MODEL_BASENAME + "_001"
        
    model.save(ms)
    train_env.save(ms + ".pkl")
    train_env.close()
    print("Model saved")