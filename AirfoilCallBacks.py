import optuna
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TensorboardAeroCallback(BaseCallback):
    def __init__(self, log_every=100):
        super().__init__()
        self.log_every = int(log_every)

    def _on_step(self) -> bool:
        t = int(self.num_timesteps)
        if t % self.log_every != 0:
            return True

        infos = self.locals.get("infos", [])
        rewards_sb3 = self.locals.get("rewards", None)

        cls, cds, lds, objs = [], [], [], []
        raw_rewards, env_rewards = [], []
        fails = []
        invalid_codes = []
        clxs, cdxs = [], []

        for info in infos:
            if not info:
                continue

            failed = bool(info.get("failed", False))
            fails.append(1.0 if failed else 0.0)

            if "invalid_code" in info:
                invalid_codes.append(int(info["invalid_code"]))

            # raw_reward = physics objective (CL or L/D) in your current env
            if "raw_reward" in info:
                raw_rewards.append(float(info["raw_reward"]))

            # reward = env reward returned by your step() (shaped if you shape)
            if "reward" in info:
                env_rewards.append(float(info["reward"]))

            if not failed:
                if "CL" in info:      cls.append(float(info["CL"]))
                if "CD" in info:      cds.append(float(info["CD"]))
                if "LD" in info:      lds.append(float(info["LD"]))
                if "obj_val" in info: objs.append(float(info["obj_val"]))

            if "CLX" in info: clxs.append(float(info["CLX"]))
            if "CDX" in info: cdxs.append(float(info["CDX"]))

        r_sb3 = None
        if rewards_sb3 is not None:
            r_sb3 = np.asarray(rewards_sb3, dtype=np.float64).ravel()

        def log_stats(tag, arr):
            if arr is None:
                return
            a = np.asarray(arr, dtype=np.float64).ravel()
            if a.size == 0:
                return
            self.logger.record(f"step/{tag}_mean", float(a.mean()))
            self.logger.record(f"step/{tag}_max",  float(a.max()))
            self.logger.record(f"step/{tag}_min",  float(a.min()))

        # Physics metrics (valid-only)
        log_stats("CL", cls)
        log_stats("CD", cds)
        log_stats("LD", lds)
        log_stats("obj_val", objs)

        # XFOIL gate diagnostics
        log_stats("CLX", clxs)
        log_stats("CDX", cdxs)

        # Rewards
        log_stats("reward_sb3", r_sb3)
        log_stats("reward_env", env_rewards)
        log_stats("raw_reward_obj", raw_rewards)

        # Health
        log_stats("failed", fails)

        # invalid_code rates (more interpretable than mean)
        if invalid_codes:
            n = len(invalid_codes)
            for c in sorted(set(invalid_codes)):
                rate = sum(x == c for x in invalid_codes) / n
                self.logger.record(f"step/invalid_code_{c}_rate", float(rate))

        self.logger.dump(step=t)
        return True
    
class OptunaEvalTBCallback(BaseCallback):
    def __init__(self, eval_fn, eval_every_steps: int, trial: optuna.Trial):
        super().__init__()
        self.eval_fn = eval_fn
        self.eval_every_steps = int(eval_every_steps)
        self.trial = trial
        self.best = -1e9

    def _on_step(self) -> bool:
        t = int(self.num_timesteps)
        if t <= 0 or (t % self.eval_every_steps) != 0:
            return True

        mean_cl = float(self.eval_fn())
        self.best = max(self.best, mean_cl)

        # TensorBoard scalars
        self.logger.record("optuna/mean_cl", mean_cl)
        self.logger.record("optuna/best_mean_cl", float(self.best))
        self.logger.dump(step=t)

        # Optuna pruning signal
        self.trial.report(mean_cl, step=t)
        if self.trial.should_prune():
            raise optuna.TrialPruned()

        return True