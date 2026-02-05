import optuna
from stable_baselines3.common.callbacks import BaseCallback

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