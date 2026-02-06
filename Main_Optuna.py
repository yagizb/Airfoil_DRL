import shutil
import multiprocessing as mp
from pathlib import Path
from typing import Callable, List, Dict, Any, Tuple

import optuna
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

import config
from AirfoilEnv import AirfoilEnv


# -----------------------------
# ENV FACTORY (trial-local dirs)
# -----------------------------
def make_env(env_id: int, run_root: Path, hist_root: Path, seed: int):
    def _init():
        work_dir = run_root / f"env_{env_id}"
        work_dir.mkdir(parents=True, exist_ok=True)

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
            # trial-local history folders
            airfoil_history_dir=str(hist_root / "airfoils"),
            cl_cd_history_dir=str(hist_root / "clcd"),
        )

        # If your env uses numpy randomness internally, this helps.
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

    if use_subproc:
        venv = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        venv = DummyVecEnv(env_fns)

    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    venv.seed(seed)

    return venv


# -----------------------------
# PPO SEARCH SPACE (CFD-friendly)
# -----------------------------

def suggest_ppo_params(trial: optuna.Trial) -> Tuple[Dict[str, Any], int]:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.99)
    gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.10, 0.30)

    ent_coef = trial.suggest_float("ent_coef", 1e-3, 5e-1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.3, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)

    n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 15, 20])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    n_steps = trial.suggest_categorical("n_steps", [64, 128, 256])

    params: Dict[str, Any] = {
        "learning_rate": learning_rate,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "policy_kwargs": dict(net_arch=[256, 256]),
    }

    return params, int(n_steps)

def _sanitize_ppo_params_for_vecenv(params: Dict[str, Any], n_envs: int) -> Dict[str, Any]:
    """
    Ensure PPO constraints:
      - batch_size <= n_steps * n_envs
      - batch_size divides n_steps * n_envs ideally (SB3 will warn otherwise)
    """
    params = dict(params)
    n_steps = int(params["n_steps"])
    batch_size = int(params["batch_size"])
    rollout_size = n_steps * n_envs

    if batch_size > rollout_size:
        # Force batch_size down to a valid value
        params["batch_size"] = max(32, rollout_size // 2)

    # Optional: make batch_size a divisor of rollout_size to avoid SB3 warnings
    batch_size = int(params["batch_size"])
    if rollout_size % batch_size != 0:
        # pick the largest batch_size from {256,128,64,32} that divides rollout_size
        for bs in [256, 128, 64, 32]:
            if bs <= rollout_size and rollout_size % bs == 0:
                params["batch_size"] = bs
                break
        else:
            # fallback: keep it; SB3 will still run
            pass

    return params


# -----------------------------
# OPTUNA OBJECTIVE
# -----------------------------
def objective(trial: optuna.Trial) -> float:
    SCRIPT_DIR = Path(__file__).resolve().parent  
    base_root = SCRIPT_DIR / "runs"
    #base_root = Path(getattr(config, "WORK_ROOT", "runs")).resolve()
    model_basename = f"airfoil_Re{int(config.RE/1e6)}M_AoA{int(config.AOA):02d}_{config.OBJECTIVE.upper()}"

    trial_root = base_root / "optuna_ppo" / model_basename / f"trial_{trial.number:04d}"
    run_root = trial_root / "work"
    hist_root = trial_root / "history"

    if trial_root.exists():
        shutil.rmtree(trial_root, ignore_errors=True)
    run_root.mkdir(parents=True, exist_ok=True)
    hist_root.mkdir(parents=True, exist_ok=True)

    seed = int(config.SEED) + trial.number

    train_env = build_vec_env(
        n_envs=config.NUM_ENVS,
        run_root=run_root,
        hist_root=hist_root,
        seed=seed,
        use_subproc=True,
    )

    # Eval env: reward normalization OFF for clearer evaluation
    eval_env = build_vec_env(
        n_envs=1,
        run_root=trial_root / "eval_work",
        hist_root=trial_root / "eval_history",
        seed=seed + 10_000,
        use_subproc=False,
    )
    eval_env.norm_reward = False

    params, n_steps = suggest_ppo_params(trial)
    params = _sanitize_ppo_params_for_vecenv(params, n_envs=config.NUM_ENVS)

    model = PPO(
        "MlpPolicy",
        train_env,
        seed=seed,
        verbose=0,
        tensorboard_log=str(trial_root / "tb"),
        **params,
    )

    #train_steps = int(getattr(config, "OPTUNA_TRAIN_STEPS", 2048))
    train_steps = int(getattr(config, "OPTUNA_TRAIN_STEPS", 1 * n_steps * config.NUM_ENVS))
    model.learn(total_timesteps=train_steps, tb_log_name=f"trial_{trial.number:04d}")

    # Sync normalization stats from training env to eval env
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms

    n_eval_eps = int(getattr(config, "OPTUNA_EVAL_EPISODES", 21))
    mean_reward, _ = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_eps,
        deterministic=True,         ###  → no action noise (important for fair comparison)
        return_episode_rewards=False,
    )

    # Pruning
    trial.report(float(mean_reward), step=0) # type: ignore # Sends an intermediate result to Optuna, step=0 because this evaluation happens once (or at a fixed point)
    if trial.should_prune():
        train_env.close()
        eval_env.close()
        raise optuna.TrialPruned()

    # Save trial artifacts (optional)
    model.save(str(trial_root / "model"))
    train_env.save(str(trial_root / "model.pkl"))

    train_env.close()
    eval_env.close()
    return float(mean_reward) # type: ignore


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    config.set_global_seeds(config.SEED)

    storage = f"sqlite:///{Path(__file__).resolve().parent / 'POP_optuna_airfoil.db'}"
    study = optuna.create_study(
        study_name="POP_airfoil_RE3_AoA00_maxCL_n99",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=config.SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8),
    )
    ## Optuna will run the objective() function n_trials times.
    study.optimize(objective, n_trials=301, gc_after_trial=True, catch=(Exception,))

    print("\n===== OPTUNA DONE =====")
    print("Best value:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")