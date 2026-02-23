from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize,DummyVecEnv
import config
from AirfoilEnv import AirfoilEnv
from pathlib import Path

from Reset import reset_history

def create_env(env_id: int):
    def _init():
        # directory where the Python script lives
        SCRIPT_DIR = Path(__file__).resolve().parent
        # WORK_ROOT from config, relative to script location
        root = SCRIPT_DIR / getattr(config, "WORK_ROOT", "runs")
        work_dir = root / f"env_{env_id}"
        
        return AirfoilEnv(
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
    return _init

if __name__ == "__main__":
    RE = 3.0E6
    AOA = 0.0
    OBJECTIVE = "cl"
    BASENAME = (f"Re{int(RE/1.e06)}M_AoA{int(AOA):02d}_{OBJECTIVE.upper()}")

    AIRFOIL_HISTORY_DIR = (f"airfoil_history_{BASENAME}")
    CL_CD_HISTORY_DIR   = (f"cl_cd_history_{BASENAME}")
    reset_history(AIRFOIL_HISTORY_DIR, CL_CD_HISTORY_DIR)
    TAG = "001"
   
    #ms = Path(MODEL_BASENAME + "_001")
    SCRIPT_DIR = Path(__file__).resolve().parent
    ms = SCRIPT_DIR/f"airfoil_{BASENAME}_{TAG}"
    NUM_ENVS = 1
    vec_env = DummyVecEnv([create_env(i) for i in range(NUM_ENVS)])
    vec_env = VecNormalize.load(str(ms.with_suffix(".pkl")), vec_env)
    #vec_env = VecNormalize.load(ms + ".pkl", vec_env)
    vec_env.training = False       # evaluation mode
    vec_env.norm_reward = False    # don’t normalize rewards during eval

    # --- load model ---
    model = SAC.load(ms, env=vec_env)
    print("Loaded model with VecNormalize for evaluation")

    # turn on saving of airfoils & metrics
    vec_env.set_attr("save_data", True)
    #vec_env.set_attr("fidelity", 1)

    obs = vec_env.reset()
    episode = 0
    while episode < config.MAX_EPISODES:
        episode += 1
        action, _ = model.predict(obs, deterministic=True) # type: ignore
        obs, rewards, dones, infos = vec_env.step(action)

        for i, info in enumerate(infos):
            print(f"[env {i}] episode={episode} reward={rewards[i]:.6g}")
            if info.get("no_imp_eps", False):
                print(f"[env {i}] stopping: no-improvement threshold met.")
                break

        if episode % config.SAVE_INTERVAL == 0:
            model.save(f"airfoil_{BASENAME}_{TAG}_ep{episode}")
            print(f"Saved model at episode {episode}")

        if any(info.get("no_imp_eps", False) for info in infos):
            break