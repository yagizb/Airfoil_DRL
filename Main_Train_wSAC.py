import multiprocessing as mp
from pathlib import Path
from typing import List, Callable

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecMonitor

import config
from AirfoilEnv import AirfoilEnv
from AirfoilCallBacks import TensorboardAeroCallback
from Reset import reset_history


def make_env(env_id: int, run_root: Path, hist_root: Path, seed: int):
    def _init():
        work_dir = run_root / f"env_{env_id}"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Make sure history dirs exist (important when save_data=True)
        (hist_root / "airfoils").mkdir(parents=True, exist_ok=True)
        (hist_root / "clcd").mkdir(parents=True, exist_ok=True)

        env = AirfoilEnv(
            env_id=env_id,
            n_envs=config.NUM_ENVS,
            work_dir=str(work_dir),
            save_data=True,                 # OK because history dirs are trial-local
            fidelity=config.FIDELITY,
            batch_id=0,
            angle_of_attack=config.AOA,
            Re_number=config.RE,
            scaling_factor=config.ACTION_SCALE,
            airfoil_file=config.AIRFOIL_FILE,
            max_steps=config.MAX_STEPS,      # if MAX_STEPS=1, each episode=1 step (fine)
            max_no_improvement_episodes=config.MAX_NO_IMPROV,
            objective=config.OBJECTIVE,
            airfoil_history_dir=str(hist_root / "airfoils"),
            cl_cd_history_dir=str(hist_root / "clcd"),
        )
        env.reset(seed=seed + env_id)
        return env
    return _init


def build_vec_env(
    n_envs: int,
    run_root: Path,
    hist_root: Path,
    seed: int,
    use_subproc: bool,
) -> VecNormalize:

    env_fns: List[Callable[[], gym.Env]] = [
        make_env(i, run_root, hist_root, seed) for i in range(n_envs)
    ]

    venv = SubprocVecEnv(env_fns, start_method="spawn") if use_subproc else DummyVecEnv(env_fns)
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    venv.seed(seed)
    return venv

def _params() -> dict:
    return {
        "learning_rate": config.LEARNING_RATE,
        "buffer_size": config.BUFFER_SIZE,
        "batch_size": config.BATCH_SIZE,
        "tau": config.TAU,
        "gamma": config.GAMMA,
        "ent_coef": config.ENT_COEF,
        "train_freq": config.TRAIN_FREQ,
        "gradient_steps": config.GRADIENT_STEPS,
        "policy_kwargs": {"net_arch": [256, 256]},
    }

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    config.set_global_seeds(config.SEED)

    trial_number = 48
    SEED = int(config.SEED) + trial_number

    SCRIPT_DIR = Path(__file__).resolve().parent
    base_root = SCRIPT_DIR / "runs"   # or Path(getattr(config, "WORK_ROOT", "runs")).resolve()

    model_basename = f"airfoil_Re{int(config.RE/1e6)}M_AoA{int(config.AOA):02d}_{config.OBJECTIVE.upper()}"

    trial_root = base_root / model_basename / f"trial_{trial_number:04d}"
    run_root = trial_root / "work"
    hist_root = trial_root / "history"

    # Ensure these exist BEFORE subprocesses start
    run_root.mkdir(parents=True, exist_ok=True)
    hist_root.mkdir(parents=True, exist_ok=True)

    # Clear history directories (only your CSV/airfoil dumps)
    reset_history(str(hist_root / "airfoils"), str(hist_root / "clcd"))

    # 1) Create base env (NO VecNormalize yet)
    train_env = build_vec_env(
        n_envs=config.NUM_ENVS,
        run_root=run_root,
        hist_root=hist_root,
        seed=SEED,
        use_subproc=True,
    )

    params = _params()

    model = SAC(
        "MlpPolicy",
        train_env,
        seed=SEED,
        verbose=0,
        tensorboard_log=str(trial_root / "tb"),
        learning_starts=getattr(config, "LEARNING_STARTS", 100),
        **params,
    )

    cb = TensorboardAeroCallback(log_every=100)

    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=cb,
        tb_log_name=f"trial_{trial_number:04d}",
    )

    # Save artifacts (be explicit with filenames)
    model.save(str(trial_root / "model"))                 # -> model.zip
    train_env.save(str(trial_root / "model.pkl"))  # -> vecnormalize.pkl

    train_env.close()
    