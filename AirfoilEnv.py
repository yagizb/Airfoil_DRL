# ---- AirfoilEnv.py (PATCH) ----
import os, json
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

from BezierCurv import generate_airfoil
from Xfoil import analyze_airfoil
from CFL3Drun import cfl3d_airfoil
import config
from typing import Optional
from typing import cast

class AirfoilEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        env_id: int,
        n_envs: int,
        work_dir: str,
        save_data: bool,
        fidelity: int,
        batch_id: int,
        angle_of_attack: float,
        Re_number: float,
        scaling_factor: float,
        airfoil_file: str,
        max_steps: int = 1,
        max_no_improvement_episodes: int = config.MAX_NO_IMPROV,
        objective: str = "cl_cd",
        # >>> NEW (trial-local history dirs
        airfoil_history_dir: Optional[str] = None,
        cl_cd_history_dir: Optional[str] = None,
    ):
        super().__init__()

        self.env_id = env_id
        self.n_envs = n_envs
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.airfoil_basename = os.path.basename(airfoil_file)
        self.airfoil_file = self.airfoil_basename

        self.save_data = bool(save_data)
        self.fidelity = int(fidelity)

        self.batch_id = int(batch_id)
        self.angle_of_attack = float(angle_of_attack)
        self.Re_number = float(Re_number)
        self.action_scaling_factor = float(scaling_factor)
        self.max_steps = int(max_steps)
        self.max_no_improvement_episodes = int(max_no_improvement_episodes)
        self.objective = objective.lower().strip()
        assert self.objective in ("cl", "cl_cd")

        # >>> USE trial-local dirs if provided, else fallback to config
        air_hist_root = airfoil_history_dir or config.AIRFOIL_HISTORY_DIR
        clcd_root     = cl_cd_history_dir or config.CL_CD_HISTORY_DIR

        os.makedirs(air_hist_root, exist_ok=True)
        os.makedirs(clcd_root, exist_ok=True)

        self.env_airfoil_history_dir = os.path.join(air_hist_root, f"env_{self.env_id}")
        os.makedirs(self.env_airfoil_history_dir, exist_ok=True)

        self.env_cl_cd_history_path = os.path.join(clcd_root, f"cl_cd_history_env_{self.env_id}.csv")
        if not os.path.exists(self.env_cl_cd_history_path):
            with open(self.env_cl_cd_history_path, "w") as f:
                f.write("Episode, CL, CD, cl_cd, Objective, Objective_Value, Reward, Invalid_code, Failed \n")

        # --- spaces ---
        self.num_control_points = config.NUM_CONTROL_POINTS
        self.cp_dim = self.num_control_points * 2
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.cp_dim,), dtype=np.float32)
        self.observation_space = Box(low=-1.0, high=1.0, shape=(self.cp_dim,), dtype=np.float32)

        # --- state ---
        self.current_step = 0
        self.previous_state = None
        self.episode_number = 0
        self.best_obj = 0.0
        self.no_improvement_episodes = 0

        self.state = self._reset_control_points()

    # (rest unchanged)

    # ---------------- Helpers ----------------
    def _reset_control_points(self) -> np.ndarray:
        """Load initial control points (expects shape (18,2)) and return flat (36,) float32."""
        cps = np.loadtxt("initial_control_points.dat", dtype=np.float32)
        expected = (self.num_control_points, 2)
        if cps.shape != expected:
            raise ValueError(f"initial_control_points.dat must be shape {expected}, got {cps.shape}")
        return cps.reshape(-1).astype(np.float32)

    def _flat_to_cps(self, flat: np.ndarray) -> np.ndarray:
        """(36,) -> (18,2)"""
        return flat.reshape(self.num_control_points, 2)

    def _objective_value(self, CL: float, CD: float, eps: float = 1e-8) -> float:
        if self.objective == "cl":
            return float(CL)
        else:
            return float(CL / max(CD, eps))
    def _csv_val(self, x):
        return "" if x is None else f"{float(x):.6g}"

    def eval_reward_01(self, CL, CD, eps: float = 1e-8):
        # Missing or non-finite → failure
        if (CL is None) or (CD is None) or (not np.isfinite(CL)) or (not np.isfinite(CD)):
            penalty = -1.0 if self.objective == "cl" else -20.0
            return penalty, penalty, None, None

        # Physical / numerical invalid
        LD = CL / max(CD, eps)
        invalid = (CD <= 0.0) or (CL >= 3.0) or (LD >= 200.0)
        if invalid:
            penalty = -1.0 if self.objective == "cl" else -20.0
            return penalty, penalty, None, None

        obj_val = float(CL) if (self.objective == "cl") else float(LD)
        reward = obj_val
        return reward, obj_val, float(CL), float(CD)
    
    # ---------------- Gymnasium API ----------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.save_data and self.previous_state is not None:
            self.state = self.previous_state.copy()
        else:
            self.state = self._reset_control_points()

        self.current_step = 0
        self.no_improvement_episodes = 0
        info = {}
        return self.state.copy(), info

    def step(self, action):
        assert isinstance(self.observation_space, Box)
        obs_space = cast(Box, self.observation_space)
        # --- Action processing ---
        action = np.asarray(action, dtype=np.float32).reshape(self.cp_dim)
        scaled_action = self.action_scaling_factor * action

        next_state = (self.state + scaled_action).astype(np.float32)
        next_state = np.clip(
            next_state,
            obs_space.low,
            obs_space.high,
        ).astype(np.float32)

        # ---------- 1) Geometry generation ----------
        airfoil_data = generate_airfoil(
            airfoil_file=self.airfoil_file,
            control_points=self._flat_to_cps(next_state),
            output_dir=str(self.work_dir),
        )
        if airfoil_data is None:
            raise ValueError("generate_airfoil(...) returned None; check BezierCurv.generate_airfoil.")

        # ---------- 2) Aerodynamics evaluation ----------
        eps = 1e-8
        CL = None
        CD = None
        CLX = None
        CDX = None

        # XFOIL gate + CFL3D
        CLX, CDX = analyze_airfoil(
                self.airfoil_file,
                self.angle_of_attack,
                self.Re_number,
                work_dir=self.work_dir,     
            )
        if self.fidelity == 3:
            # Load from broker_io/results_k.json
            try:
                with open(f"broker_io/results_{self.batch_id}.json", "r") as f:
                    data = json.load(f)
                    for result in data["results"]:
                        if result["env_id"] == self.env_id:
                            CL = float(result["Cl"])
                            CD = float(result["CD"])
                            break
                    else:
                        print(f"[Warning] No match for env_id={self.env_id}")
            except Exception as e:
                print(f"[Error] Failed to load broker_io/results_{self.batch_id}.json: {e}")

        elif self.fidelity == 2:
            # High-fidelity SOD2D (stub for now)
            print(f"[Env {self.env_id}] SOD2D evaluation (stub)")
            CL, CD = 0.0, eps

        elif self.fidelity == 1:
            
            if (CLX is None) or (CDX is None) or (CLX <= 0.0):
                print(f"[Env {self.env_id}] XFOIL CL <= 0 → use coarse result")
                CL, CD = CLX, CDX
                
            else:
                print(f"[Env {self.env_id}] Run CFL3D simulation")
                CL, CD = cfl3d_airfoil(
                    self.env_id,
                    self.n_envs,
                    self.airfoil_file,
                    self.angle_of_attack,
                    self.Re_number,
                    work_dir=self.work_dir,
                )

        else:
            # Low-fidelity XFOIL only
            print(f"[Env {self.env_id}] Run XFOIL simulation")
            CL, CD = CLX, CDX

        # ---------- 3) Reward ----------
        reward, obj_val, CL, CD = self.eval_reward_01(CL, CD, eps=eps)
        # Compute LD only if valid
        LD = float(CL / max(CD, eps)) if (CL is not None and CD is not None) else 0.0

        # Failure detection
        invalid_code = int(self._invalid_code(CL, CD, eps=eps))
        failed = (obj_val == -1.0) or (invalid_code != 0)   # robust
        #terminated = bool(failed)
        terminated = False      ## (penalize but don’t constantly reset) → usually much better for PPO in brittle CFD environments

        # Truncation
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
   
        self.previous_state = next_state.copy()
        self.state = next_state.copy()

        if self.save_data:
            self.episode_number += 1
            with open(self.env_cl_cd_history_path, "a") as f:
                f.write(
                    f"{self.episode_number},"
                    f"{self._csv_val(CL)},"
                    f"{self._csv_val(CD)},"
                    f"{self._csv_val(LD)},"
                    f"{self.objective},"
                    f"{obj_val:.6g},"
                    f"{reward:.6g},"
                    f"{invalid_code},"
                    f"{int(failed)}\n"
                )

            # Save only if "good" reward 
            if reward > 0.0:
                airfoil_path = os.path.join(self.env_airfoil_history_dir, f"airfoil_{self.episode_number}.dat")
                with open(airfoil_path, "w") as f:
                    f.write(airfoil_data) # type: ignore

        # ---------- info ----------
        info = {
            "obj_val": float(obj_val),
            "failed": bool(failed),
            "invalid_code": int(invalid_code),
            "objective": self.objective,
            "raw_reward": float(obj_val),    # physics objective (CL or L/D)
            "reward": float(reward),     # shaped PPO reward    
        }

        # Only attach aero metrics when the step is VALID
        if not failed:
            info.update({
                "CL": float(CL), # type: ignore
                "CD": float(CD), # type: ignore
                "LD": float(LD),
            })

        # Optional: record XFOIL gate values (diagnostics only)
        if CLX is not None:
            info["CLX"] = float(CLX)
        if CDX is not None:
            info["CDX"] = float(CDX)

        return next_state.copy(), float(reward), terminated, bool(truncated), info

    def render(self):
        pass

    def _invalid_code(self, CL, CD, eps=1e-8):
        if (CL is None) or (CD is None):
            return 1  # missing
        if (not np.isfinite(CL)) or (not np.isfinite(CD)):
            return 2  # nan/inf
        # explicit crash / nonsense
        if CL == 0.0:
            return 6  # xfoil failed / converged to nonsense
        if CD <= 0.0:
            return 3  # cd <= 0
        LD = CL / max(CD, eps)
        if CL >= 3.0:
            return 4  # cl too large
        if LD >= 200.0:
            return 5  # ld too large
        return 0