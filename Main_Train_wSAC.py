from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, VecMonitor
from pathlib import Path
import multiprocessing as mp

import numpy as np
import torch
import random

import config
from AirfoilEnv import AirfoilEnv
from AirfoilCallBacks import TensorboardAeroCallback
from Reset import reset_history


# --- GLOBAL SEED SETUP ---
trial_number = 48
SEED = int(config.SEED) + trial_number


def create_env(env_id: int, seed: int = 0):
    def _init():
        SCRIPT_DIR = Path(__file__).resolve().parent
        root = SCRIPT_DIR / getattr(config, "WORK_ROOT", "runs")
        work_dir = root / f"env_{env_id}"

        env = AirfoilEnv(
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
    mp.set_start_method("spawn", force=True)

    reset_history(config.AIRFOIL_HISTORY_DIR, config.CL_CD_HISTORY_DIR)

    MODEL_BASENAME = f"airfoil_Re{int(config.RE/1e6)}M_AoA{int(config.AOA):02d}_{config.OBJECTIVE.upper()}"

    # --- PASS SEED TO EACH ENV ---
    train_env = SubprocVecEnv(
        [create_env(i, seed=SEED) for i in range(config.NUM_ENVS)],
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
        seed=SEED,   # important
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
    )

    cb = TensorboardAeroCallback(log_every=100)

    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=cb,
        tb_log_name=f"XFOIL001_SACtr{trial_number}_MaxCL"
    )

    print("Learned")

    ms = MODEL_BASENAME + f"_XFOIL001_SACtr{trial_number}_MaxCL"

    model.save(ms)
    train_env.save(ms + ".pkl")
    train_env.close()

    print("Model saved")