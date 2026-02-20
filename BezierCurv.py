import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# --------- Bezier utilities (vectorized) ---------
def bernstein_basis(n: int, t: np.ndarray) -> np.ndarray:
    """
    Bernstein basis matrix B of shape (len(t), n+1)
    where B[i, j] = C(n, j) * t[i]^j * (1 - t[i])^(n-j)
    """
    t = np.asarray(t, dtype=np.float64)
    j = np.arange(n + 1, dtype=np.float64)
    from math import comb
    C = np.array([comb(n, int(k)) for k in j], dtype=np.float64)
    T = t.reshape(-1, 1)
    B = C * (T ** j) * ((1.0 - T) ** (n - j))
    return B

def bezier_curve(control_points: np.ndarray, n_points: int = 96) -> np.ndarray:
    """
    control_points: (m, 2) with m >= 2 ; n = m-1 is polynomial degree
    returns curve of shape (n_points, 2)
    """
    cps = np.asarray(control_points, dtype=np.float64)
    m = cps.shape[0]
    if cps.ndim != 2 or cps.shape[1] != 2 or m < 2:
        raise ValueError(f"control_points must be (m,2) with m>=2, got {cps.shape}")
    n = m - 1
    t = np.linspace(0.0, 1.0, n_points)
    B = bernstein_basis(n, t)                    # (n_points, m)
    curve = B @ cps                              # (n_points, 2)
    return curve.astype(np.float32)

import numpy as np
from typing import Tuple

def non_overlap_ok(
    upper_curve: np.ndarray,
    lower_curve: np.ndarray,
    eps: float = 1e-4,
    n: int = 401,
) -> Tuple[bool, float]:
    """
    Excludes the first and last points of BOTH upper and lower surfaces
    before performing thickness check.

    Enforces:
        y_upper(x) >= y_lower(x) + eps

    Returns:
        (ok, min_thickness)
    """

    upper_curve = np.asarray(upper_curve, dtype=np.float64)
    lower_curve = np.asarray(lower_curve, dtype=np.float64)

    # ---- Exclude first and last surface points ----
    if upper_curve.shape[0] > 2:
        upper_curve = upper_curve[1:-1]
    if lower_curve.shape[0] > 2:
        lower_curve = lower_curve[1:-1]

    xu, yu = upper_curve[:, 0], upper_curve[:, 1]
    xl, yl = lower_curve[:, 0], lower_curve[:, 1]

    # Common x-grid
    x = np.linspace(0.0, 1.0, n)

    # Sort by x for interpolation
    iu = np.argsort(xu)
    il = np.argsort(xl)

    yu_i = np.interp(x, xu[iu], yu[iu])
    yl_i = np.interp(x, xl[il], yl[il])

    t = yu_i - yl_i
    tmin = float(np.min(t))

    return (tmin >= eps), tmin

# --------- Airfoil generator ---------
def generate_airfoil(
    airfoil_file: str,
    control_points: Optional[np.ndarray] = None,
    n_samples_upper: int = 96,
    n_samples_lower: int = 96,
    lock_endpoints: bool = True,
    default_cp_path: str = "initial_control_points.dat",
    return_array: bool = False,
    output_dir: Optional[str] = None,

    # ---- NEW: overlap constraint knobs ----
    enforce_non_overlap: bool = True,
    thickness_eps: float = 1e-4,
    thickness_grid_n: int = 401,
    raise_on_overlap: bool = True,
):
    """
    Build airfoil from Bezier control points with optional non-overlap constraint.

    Non-overlap constraint:
        y_upper(x) >= y_lower(x) + thickness_eps  on a common x-grid.

    If enforce_non_overlap and violation happens:
      - if raise_on_overlap: raises ValueError 
      - else: writes anyway, but returns info in the string (not recommended for training)

    Writes:  <output_dir>/<airfoil_file>.dat
    Returns: airfoil_data_str (and optionally the (N,2) array if return_array=True)
    """
    # Load default control points (18x2 expected)
    cps_default = np.loadtxt(default_cp_path, dtype=np.float32)
    if cps_default.shape != (18, 2):
        raise ValueError(f"{default_cp_path} must be shape (18,2), got {cps_default.shape}")

    # Parse incoming control points
    if control_points is None:
        cps = cps_default.copy()
    else:
        cp = np.asarray(control_points, dtype=np.float32)
        if cp.ndim == 1:
            if cp.size != 36:
                raise ValueError(f"Flat control_points must have 36 floats, got {cp.size}")
            cp = cp.reshape(18, 2)
        elif cp.shape != (18, 2):
            raise ValueError(f"control_points must be (18,2) or flat 36, got {cp.shape}")

        cps = cps_default.copy()
        if lock_endpoints:
            # Keep endpoints (index 0 and 8 of each surface) fixed from defaults
            cps[1:8, :] = cp[1:8, :]          # upper interior
            cps[10:17, :] = cp[10:17, :]      # lower interior (locks 9 and 17)
        else:
            cps = cp

    # Split into surfaces
    upper_cps = cps[:9, :]    # 9 control points (degree 8)
    lower_cps = cps[9:, :]    # 9 control points (degree 8)

    # Generate curves
    upper_curve = bezier_curve(upper_cps, n_points=n_samples_upper)
    lower_curve = bezier_curve(lower_cps, n_points=n_samples_lower)

    # Reverse if your convention wants TE->LE etc.
    upper_curve = upper_curve[::-1]   # upper: TE -> LE
    lower_curve = lower_curve[::-1]   # lower: TE -> LE

    # ---- enforce non-overlap / positive thickness ----
    if enforce_non_overlap:
        ok, tmin = non_overlap_ok(
            upper_curve, lower_curve,
            eps=thickness_eps,
            n=thickness_grid_n,
        )
        if not ok:
            msg = (
                f"Airfoil self-intersection / overlap detected: "
                f"min_thickness={tmin:.6e} < eps={thickness_eps:.6e}. "
                f"Rejecting geometry."
            )
            if raise_on_overlap:
                raise ValueError(msg)
            else:
                # keep going but include warning in output string
                # (not recommended for RL; better to reject)
                warning_line = f"# WARNING: {msg}\n"
        else:
            warning_line = ""
    else:
        warning_line = ""

    # Stack into a single loop without duplicating the leading edge point
    airfoil = np.vstack([upper_curve, lower_curve[1:]])

    # Determine output path
    if output_dir is None:
        out_dir = Path.cwd()
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{airfoil_file}.dat"

    # Save coordinates
    np.savetxt(out_path, airfoil, fmt="%.6f")

    # Build string for env logging
    airfoil_data_str = warning_line + "\n".join(
        " ".join(f"{x:.6f}" for x in row) for row in airfoil
    )

    if return_array:
        return airfoil_data_str, airfoil
    return airfoil_data_str