from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from pathlib import Path
import multiprocessing as mp
import config
from AirfoilEnv import AirfoilEnv
from AirfoilCallBacks import TensorboardAeroCallback
from Reset import reset_history

def create_env(env_id):
    def _init():
        script_dir = Path(__file__).resolve().parent
        root = script_dir / getattr(config, "WORK_ROOT", "runs")
        work_dir = root / f"env_{env_id}"

        env = AirfoilEnv(
            env_id=env_id,
            n_envs=config.NUM_ENVS,
            work_dir=str(work_dir),
            save_data=False,  # OK only if it writes under work_dir (env-specific)
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
    config.set_global_seeds(config.SEED)
    trial_number = 48 #### optimum hyperparameters found at trial 48, see Optuna_Results_Summary.txt
    SEED = int(config.SEED) + trial_number
    reset_history(config.AIRFOIL_HISTORY_DIR, config.CL_CD_HISTORY_DIR)

    MODEL_BASENAME = f"airfoil_Re{int(config.RE/1e6)}M_AoA{int(config.AOA):02d}_{config.OBJECTIVE.upper()}"
    
    # vec_env = SubprocVecEnv([create_env(i) for i in range(config.NUM_ENVS)])
    vec_env = SubprocVecEnv(
        [create_env(i) for i in range(config.NUM_ENVS)],
        start_method="spawn",
    )
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    vec_env.seed(SEED)

    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=config.LEARNING_RATE,
        buffer_size=config.BUFFER_SIZE,
        batch_size=config.BATCH_SIZE,
        tau=config.TAU,
        gamma=config.GAMMA,
        ent_coef=config.ENT_COEF,
        learning_starts=config.LEARNING_STARTS,
        train_freq=config.TRAIN_FREQ,
        gradient_steps=config.GRADIENT_STEPS,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=config.VERBOSE,
        seed=SEED,
        tensorboard_log="./tensorboard_logs/",
    )
    cb = TensorboardAeroCallback(log_every=100)
    
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS,callback=cb, tb_log_name="Roll_001")
    print("Learned")

    ms = MODEL_BASENAME + "_001"
    model.save(str(ms))
    vec_env.save(str(ms) + ".pkl")
    vec_env.close()