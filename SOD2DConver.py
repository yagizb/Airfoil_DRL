import numpy as np
import time
import math


tAVG_start = int(1)

##FUNCTIONS ----------------------------
def compute_runmean(force, time=0, t_start=0):
      
	#Start averaging from t_start
	if t_start > 0:
		mask = np.where(time >= t_start)[0]
		time = time[mask]
		force = force[mask]
      
	#Perform average
	nsample = force.shape[0]
	runmean = np.zeros((nsample,))

	for i in range(nsample):

		if i == 0:
			runmean[i] = force[i]

		else:
			runmean[i] = np.mean(force[:i+1])

	return runmean, time

def compute_runStandardDeviation(force, time=0, t_start=0):

	#Start averaging from t_start
	if t_start > 0:
		mask = np.where(time >= t_start)[0]
		time = time[mask]
		force = force[mask]
      
	#Perform standard deviation
	nsample = force.shape[0]
	runEps = np.zeros((nsample,))

	for i in range(nsample):

		if i == 0:
			runEps[i] = np.std(force[0])

		else:
			runEps[i] = np.std(force[:i+1])

	return runEps, time	
def detect_convergence(
    time,
    avg,
    std,
    window_fraction=0.20,
    tol_avg=0.02,
    tol_std=0.05,
    rms_ratio_limit=None,
    abs_std_limit=None,
):
    """
    Detect convergence using the last window_fraction of the signal.

    Converged if:
    1) running mean changes less than tol_avg
    2) running std changes less than tol_std
    3) RMS/mean ratio is below rms_ratio_limit
    4) ??? absolute std is below abs_std_limit
    """

    n = len(time)

    if n < 20:
        return False, {}

    i0 = int((1.0 - window_fraction) * n)

    avg_win  = avg[i0:]
    std_win  = std[i0:]
    time_win = time[i0:]

    avg_ref = max(abs(np.mean(avg_win)), 1e-12)
    std_ref = max(abs(np.mean(std_win)), 1e-12)

    avg_change = abs(avg_win[-1] - avg_win[0]) / avg_ref
    std_change = abs(std_win[-1] - std_win[0]) / std_ref

    avg_final = avg[-1]
    std_final = std[-1]

    rms_ratio = std_final / max(abs(avg_final), 1e-12)

    converged = (
        (avg_change < tol_avg) and
        (std_change < tol_std)
    )

    # Additional RMS/mean criterion
    if rms_ratio_limit is not None:
        converged = converged and (rms_ratio < rms_ratio_limit)

    # Additional absolute std criterion
    if abs_std_limit is not None:
        converged = converged and (std_final < abs_std_limit)

    info = {
        "time_start": time_win[0],
        "time_end": time_win[-1],
        "avg_change_percent": 100.0 * avg_change,
        "std_change_percent": 100.0 * std_change,
        "avg_final": avg_final,
        "std_final": std_final,
        "rms_ratio": rms_ratio,
    }

    return converged, info

def follow_sod2d_output(
    output_geo,
    angle_of_attack,
    chord,
    Lz,
    rho,
    Uinf,
    max_abs_force=1.0e6,
):
    filename = f"surf_code_4-{output_geo}_per-4.dat"

    S_ref = chord * Lz
    AoA = np.deg2rad(angle_of_attack)
    q_inf = 0.5 * rho * Uinf * Uinf

    surf_data = np.genfromtxt(filename, delimiter=",", skip_header=1)

    if surf_data.size == 0:
        raise ValueError(f"{filename} exists but has no data rows yet")

    surf_data = np.atleast_2d(surf_data)

    if surf_data.shape[1] < 8:
        raise ValueError(
            f"{filename} has invalid shape {surf_data.shape}; expected at least 8 columns"
        )

    # -------------------------------------------------
    # IMPORTANT:
    # Do NOT filter NaN rows.
    # NaN/Inf means SOD2D diverged.
    # -------------------------------------------------
    if not np.all(np.isfinite(surf_data)):
        bad_rows = np.where(~np.isfinite(surf_data).all(axis=1))[0]
        first_bad = bad_rows[0]

        iter_bad = surf_data[first_bad, 0]
        time_bad = surf_data[first_bad, 1]

        raise FloatingPointError(
            f"{filename} contains NaN/Inf at row={first_bad}, "
            f"ITER={iter_bad}, TIME={time_bad}. SOD2D diverged."
        )

    # -------------------------------------------------
    # Force explosion check before CL/CD computation
    # -------------------------------------------------
    force_cols = surf_data[:, 3:9]
    max_force = np.max(np.abs(force_cols))

    if max_force > max_abs_force:
        row_max = np.argmax(np.max(np.abs(force_cols), axis=1))
        iter_bad = surf_data[row_max, 0]
        time_bad = surf_data[row_max, 1]

        raise FloatingPointError(
            f"{filename} force explosion at ITER={iter_bad}, "
            f"TIME={time_bad}, max_force={max_force:.3e}. "
            "SOD2D diverged."
        )

    if surf_data.shape[0] < 5:
        raise ValueError(
            f"{filename} has only {surf_data.shape[0]} rows; waiting for more"
        )

    # -------------------------------------------------
    # Filter backwards/restarted time steps
    # -------------------------------------------------
    raw_time = surf_data[:, 1]

    mask = np.ones_like(raw_time, dtype=bool)
    max_value = raw_time[0]

    for i in range(1, len(raw_time)):
        if raw_time[i] < raw_time[i - 1]:
            mask[i] = False
        else:
            max_value = max(max_value, raw_time[i])
            if raw_time[i] < max_value:
                mask[i] = False

    surf_data = surf_data[mask]

    if surf_data.shape[0] < 5:
        raise ValueError(
            f"{filename} has only {surf_data.shape[0]} valid rows after filtering"
        )

    iter_arr = surf_data[:, 0]
    time_arr = surf_data[:, 1] * Uinf / chord

    Cl_pres = (
        -surf_data[:, 3] * np.sin(AoA)
        + surf_data[:, 4] * np.cos(AoA)
    ) / (q_inf * S_ref)

    Cl_visc = (
        -surf_data[:, 6] * np.sin(AoA)
        + surf_data[:, 7] * np.cos(AoA)
    ) / (q_inf * S_ref)

    Cl = Cl_pres + Cl_visc

    Cd_pres = (
        surf_data[:, 3] * np.cos(AoA)
        + surf_data[:, 4] * np.sin(AoA)
    ) / (q_inf * S_ref)

    Cd_visc = (
        surf_data[:, 6] * np.cos(AoA)
        + surf_data[:, 7] * np.sin(AoA)
    ) / (q_inf * S_ref)

    Cd = Cd_pres + Cd_visc

    Cl_avg, time_Clavg = compute_runmean(
        Cl,
        time=time_arr,
        t_start=tAVG_start,
    )

    Cd_avg, time_Cdavg = compute_runmean(
        Cd,
        time=time_arr,
        t_start=tAVG_start,
    )

    if len(Cl_avg) == 0 or len(Cd_avg) == 0:
        raise ValueError(
            f"Not enough data after tAVG_start={tAVG_start}. "
            f"time range=[{time_arr.min():.6e}, {time_arr.max():.6e}]"
        )

    Cl_std, time_Clstd = compute_runStandardDeviation(
        Cl,
        time=time_arr,
        t_start=tAVG_start,
    )

    Cd_std, time_Cdstd = compute_runStandardDeviation(
        Cd,
        time=time_arr,
        t_start=tAVG_start,
    )

    if len(Cl_std) == 0 or len(Cd_std) == 0:
        raise ValueError("Not enough data to compute running standard deviation")

    Cl_conv, Cl_info = detect_convergence(
        time_Clavg,
        Cl_avg,
        Cl_std,
        window_fraction=0.20,
        tol_avg=0.03,
        tol_std=0.10,
        rms_ratio_limit=0.10,
    )

    Cd_conv, Cd_info = detect_convergence(
        time_Cdavg,
        Cd_avg,
        Cd_std,
        window_fraction=0.20,
        tol_avg=0.05,
        tol_std=0.10,
        abs_std_limit=0.005,
    )

    fully_converged = Cl_conv and Cd_conv

    return fully_converged, float(Cl_avg[-1]), float(Cd_avg[-1])


def _safe_float(x):
    try:
        return float(str(x).strip())
    except Exception:
        return float("nan")

def wait_until_sod2d_converged(
    output_geo,
    angle_of_attack,
    chord,
    Lz,
    rho,
    Uinf,
    time_step=5,
    max_wait_time=7200,
    max_abs_coeff=1.0e2,
):
    start_time = time.time()

    last_CL = None
    last_CD = None

    while True:
        elapsed_time = time.time() - start_time

        if elapsed_time > max_wait_time:
            print(
                f"[SOD2D monitor] Timeout: SOD2D did not converge after "
                f"{elapsed_time:.1f} seconds."
            )
            return "TIMEOUT", last_CL, last_CD

        try:
            fully_converged, CL, CD = follow_sod2d_output(
                output_geo=output_geo,
                angle_of_attack=angle_of_attack,
                chord=chord,
                Lz=Lz,
                rho=rho,
                Uinf=Uinf,
            )

            CL = _safe_float(CL)
            CD = _safe_float(CD)

            if not math.isfinite(CL) or not math.isfinite(CD):
                print(
                    f"[SOD2D monitor] NaN/Inf detected: "
                    f"CL={CL}, CD={CD}. SOD2D diverged."
                )
                return "DIVERGED", None, None

            if abs(CL) > max_abs_coeff or abs(CD) > max_abs_coeff:
                print(
                    f"[SOD2D monitor] Divergence detected: "
                    f"CL={CL}, CD={CD}, limit={max_abs_coeff}."
                )
                return "DIVERGED", None, None

            # Store last valid values
            last_CL = CL
            last_CD = CD

            if fully_converged:
                print(
                    f"[SOD2D monitor] Fully converged: "
                    f"CL={CL:.6f}, CD={CD:.6f}, "
                    f"elapsed={elapsed_time:.1f}s"
                )
                return "CONVERGED", CL, CD

            print(
                f"[SOD2D monitor] Not converged yet: "
                f"CL={CL:.6f}, CD={CD:.6f}, "
                f"elapsed={elapsed_time:.1f}s"
            )

        except FloatingPointError as e:
            print(f"[SOD2D monitor] {e}")
            return "DIVERGED", None, None

        except Exception as e:
            print(f"[SOD2D monitor] Waiting for valid output file: {e}")

        time.sleep(time_step)