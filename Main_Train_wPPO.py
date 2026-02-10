import multiprocessing as mp
from pathlib import Path
from typing import List, Callable

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecMonitor

import config
from AirfoilEnv import AirfoilEnv
from AirfoilCallBacks import TensorboardAeroCallback

from Reset import reset_history
import os

def make_env(env_id: int, seed: int):
    def _init():
        script_dir = Path(__file__).resolve().parent
        root = script_dir / getattr(config, "WORK_ROOT", "runs")

        # IMPORTANT: env-specific dir so subprocesses never clash
        work_dir = root / f"env_{env_id}"
        work_dir.mkdir(parents=True, exist_ok=True)

        env = AirfoilEnv(
            env_id=env_id,
            n_envs=config.NUM_ENVS,
            work_dir=str(work_dir),
            save_data=False,
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
        env.reset(seed=seed + env_id)
        return env
    return _init


def build_vec_env(n_envs: int, seed: int, use_subproc: bool) -> VecNormalize:
    env_fns: List[Callable[[], gym.Env]] = [make_env(i, seed) for i in range(n_envs)]

    if use_subproc:
        venv = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        venv = DummyVecEnv(env_fns)

    venv = VecMonitor(venv)

    # Fresh normalization stats (START FROM SCRATCH)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Seed vec env + action space etc.
    venv.seed(seed)
    return venv


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Global seeds
    config.set_global_seeds(config.SEED)   
    
    # IMPORTANT: ensure no old history influences your run
    reset_history(config.AIRFOIL_HISTORY_DIR, config.CL_CD_HISTORY_DIR)

    MODEL_BASENAME = f"airfoil_Re{int(config.RE/1e6)}M_AoA{int(config.AOA):02d}_{config.OBJECTIVE.upper()}"

    trial_number = 92 
    SEED = int(config.SEED) + trial_number
    
    train_env = build_vec_env(
        n_envs=config.NUM_ENVS,
        seed=SEED,
        use_subproc=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config.LEARNING_RATE,
        clip_range=config.CLIP_RANGE,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        use_sde=config.USE_SDE,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        ent_coef=config.ENT_COEF,
        vf_coef=config.VF_COEF,
        max_grad_norm=config.MAX_GRAD_NORM,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=config.VERBOSE,
        seed=SEED,
        tensorboard_log="./tensorboard_logs/",
    )

    cb = TensorboardAeroCallback(log_every=100)

    # Fresh training from timestep 0
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=cb,
        tb_log_name="Roll_001",
        reset_num_timesteps=True,
    )

    print("Learned")

    ms = MODEL_BASENAME + "_001"
    model.save(str(ms))
    train_env.save(str(ms) + ".pkl")
    train_env.close()