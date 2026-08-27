# Main_Train_SAC_Optuna.py
import shutil
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, List, Callable

import gymnasium as gym
import optuna

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

import DRL_config
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
            n_envs=DRL_config.NUM_ENVS,
            work_dir=str(work_dir),
            save_data=True,                 # OK because history dirs are trial-local
            fidelity=DRL_config.FIDELITY,
            batch_id=0,
            angle_of_attack=DRL_config.AOA,
            Re_number=DRL_config.RE,
            scaling_factor=DRL_config.ACTION_SCALE,
            airfoil_file=DRL_config.AIRFOIL_FILE,
            max_steps=DRL_config.MAX_STEPS,      # if MAX_STEPS=1, each episode=1 step (fine)
            max_no_improvement_episodes=DRL_config.MAX_NO_IMPROV,
            objective=DRL_config.OBJECTIVE,
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
# OPTUNA SEARCH SPACE
# -----------------------------
def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    lr = trial.suggest_float("learning_rate", 3e-5, 1e-4, log=True)                 ## Step size of gradient descent for actor and critics
    buffer_size = trial.suggest_int("buffer_size", 20_000, 100_000, log=True)    ## Size of replay buffer (number of past transitions stored)     
    batch_size = trial.suggest_categorical("batch_size", [256, 512,1024])           ## Number of samples used per gradient update
    tau = trial.suggest_float("tau", 0.003, 0.01, log=True)                         ## Speed of soft update of target Q-networks: θ_target ← τ θ + (1 − τ) θ_target   
    gamma = trial.suggest_float("gamma", 0.95, 0.9995)                              ## discount factor, How much the agent values future rewards
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.02, log=True)                 ## Exploration vs exploitation

    train_freq = trial.suggest_categorical("train_freq", [2, 4, 6])                 ## How often the model updates (in env steps)
    gradient_steps = trial.suggest_categorical("gradient_steps", [2, 4])            ## How many gradient updates per training event


    return dict(
        learning_rate=lr,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        ent_coef=ent_coef,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        policy_kwargs=dict(net_arch=[256, 256]),
    )


# -----------------------------
# OPTUNA OBJECTIVE
# -----------------------------
def objective(trial: optuna.Trial) -> float:
    SCRIPT_DIR = Path(__file__).resolve().parent 
    base_root = SCRIPT_DIR / "runs"
    #base_root = Path(getattr(DRL_config, "WORK_ROOT", "runs")).resolve()
    model_basename = f"airfoil_Re{int(DRL_config.RE/1e6)}M_AoA{int(DRL_config.AOA):02d}_{DRL_config.OBJECTIVE.upper()}"

    trial_root = base_root / "optuna" / model_basename / f"trial_{trial.number:04d}"
    run_root = trial_root / "work"
    hist_root = trial_root / "history"

    if trial_root.exists():
        shutil.rmtree(trial_root, ignore_errors=True)
    run_root.mkdir(parents=True, exist_ok=True)
    hist_root.mkdir(parents=True, exist_ok=True)

    seed = int(DRL_config.SEED) + trial.number

    train_env = build_vec_env(
        n_envs=DRL_config.NUM_ENVS,
        run_root=run_root,
        hist_root=hist_root,
        seed=seed,
        use_subproc=True,
    )

    # Eval env: reward normalization OFF (more interpretable)
    eval_env = build_vec_env(
        n_envs=1,
        run_root=trial_root / "eval_work",
        hist_root=trial_root / "eval_history",
        seed=seed + 10_000,
        use_subproc=False,
    )
    eval_env.norm_reward = False

    params = suggest_params(trial)

    model = SAC(
        "MlpPolicy",
        train_env,
        seed=seed,
        verbose=0,
        tensorboard_log=str(trial_root / "tb"),
        learning_starts=getattr(DRL_config, "LEARNING_STARTS", 100),
        **params,
    )

    train_steps = int(getattr(DRL_config, "OPTUNA_TRAIN_STEPS", 2048)) ## CFD calls total per 1 trial, done in parallel batches of n_envs.
    model.learn(total_timesteps=train_steps, tb_log_name=f"trial_{trial.number:04d}")

    # Sync normalization statistics
    
    eval_env.obs_rms = train_env.obs_rms        ## obs_rms: running mean & variance of observations
    eval_env.ret_rms = train_env.ret_rms        ## ret_rms: running mean & variance of returns (rewards)

    n_eval_eps = int(getattr(DRL_config, "OPTUNA_EVAL_EPISODES", 21))
    mean_reward, _ = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_eps,
        deterministic=True,
        return_episode_rewards=False,
    )

    # pruning
    trial.report(float(mean_reward), step=0) # type: ignore
    if trial.should_prune():
        train_env.close()
        eval_env.close()
        raise optuna.TrialPruned()

    # save trial artifacts (optional)
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
    DRL_config.set_global_seeds(DRL_config.SEED)

    storage = f"sqlite:///{Path(__file__).resolve().parent / 'optuna_airfoil.db'}"
    study = optuna.create_study(
        study_name="SAC_airfoil_RE3_AoA00_maxCLCD",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=DRL_config.SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8),
    )
    ## Optuna will run the objective() function n_trials times.
    study.optimize(objective, n_trials=301, gc_after_trial=True, catch=(Exception,))

    print("\n===== OPTUNA DONE =====")
    print("Best value:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")