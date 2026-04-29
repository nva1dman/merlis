#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.special import voigt_profile

from s00_settings import (
    LINE_FITS_DIR,
    BALMER_DIR,
    BALMER_LINES,
    LINE_HALF_WINDOW,
    CONT_GAP,
    CONT_WIDTH,
    SMOOTH_WIN_LINE,
    SMOOTH_POLY_LINE,
    SNR_DETECT,
    SNR_RELIABLE,
    K_SIGMA_LINE,
    FRAC_DEPTH_LINE,
    DBIC_SIMPLE_EQUIV,
    DBIC_BLEND_STRONG,
    N_MC_LINE,
    N_MC_BROAD_COMPLEX,
    RNG_SEED,
    SAVE_PLOTS,
    OUTPUT_TAG,
    get_line_table,
)

from s02_prepare_spectra import run_prepare_spectra

from s01_spectrum_utils import (
    local_continuum,
    normalise_flux,
    local_sigma,
    find_line_boundaries,
    ew_from_observed,
    ew_err_analytic,
    calc_aic_bic,
    model_fwhm_generic,
    absorption_centroid,
)

from s03_balmer_multicomponent import (
    BALMER_MODELS,
    BALMER_DISPLAY,
    analyse_balmer_line,
    balmer_fit_robust,
    balmer_components,
    balmer_model_fwhm,
    balmer_measure_ew,
)

warnings.filterwarnings("ignore")


# ============================================================
# DETECTION STATUS
# ============================================================

def classify_detection_status(snr_depth, detect=SNR_DETECT, reliable=SNR_RELIABLE):
    if not np.isfinite(snr_depth):
        return "unknown"
    if snr_depth < detect:
        return "not_detected"
    if snr_depth < reliable:
        return "marginal"
    return "detected"


# ============================================================
# BROAD-COMPLEX LOOKUP
# ============================================================

def _canon(s):
    if s is None:
        return ""
    return (
        str(s)
        .strip()
        .lower()
        .replace("$", "")
        .replace("\\", "")
        .replace("{", "")
        .replace("}", "")
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


BALMER_LOOKUP = {}
for li in BALMER_LINES:
    BALMER_LOOKUP[_canon(li["ascii"])] = li
    BALMER_LOOKUP[_canon(li["name"])] = li


def identify_broad_complex_line(line_id, label, wavelength, tol=2.0):
    keys = [_canon(line_id), _canon(label)]

    for k in keys:
        if k in BALMER_LOOKUP:
            return True, BALMER_LOOKUP[k]

    for li in BALMER_LINES:
        if abs(float(wavelength) - float(li["rest"])) <= tol:
            return True, li

    return False, None


# ============================================================
# LINE LIST
# ============================================================

def read_line_list():
    df = get_line_table()

    required = {"line_id", "label", "wavelength"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in line list: {missing}")

    df["line_id"] = df["line_id"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df["wavelength"] = df["wavelength"].astype(float)

    print("\nLoaded line list:")
    print(df.to_string(index=False))

    return df


# ============================================================
# GENERAL LINE-PROFILE MODELS
# ============================================================

def lp_gaussian(w, amp, lam0, sigma):
    return 1.0 - amp * np.exp(-0.5 * ((w - lam0) / sigma) ** 2)


def lp_lorentzian(w, amp, lam0, gamma):
    return 1.0 - amp / (1.0 + ((w - lam0) / gamma) ** 2)


def lp_voigt(w, amp, lam0, sigma, gamma):
    pk = voigt_profile(0.0, sigma, gamma)
    if pk <= 0:
        return np.ones_like(w)
    prof = amp * voigt_profile(w - lam0, sigma, gamma) / pk
    return 1.0 - prof


def lp_double_gaussian(w, a1, l1, s1, a2, l2, s2):
    g1 = a1 * np.exp(-0.5 * ((w - l1) / s1) ** 2)
    g2 = a2 * np.exp(-0.5 * ((w - l2) / s2) ** 2)
    return 1.0 - g1 - g2


LINE_PROFILE_MODELS = {
    "Gaussian": {"func": lp_gaussian, "k": 3},
    "Lorentzian": {"func": lp_lorentzian, "k": 3},
    "Voigt": {"func": lp_voigt, "k": 4},
    "DoubleGaussian": {"func": lp_double_gaussian, "k": 6},
}


# ============================================================
# FIT INITIALIZATION / MODEL FITTING
# ============================================================

def line_profile_init_bounds(model_name, rest, lam_init, depth_init):
    eps = 0.001

    if model_name == "Gaussian":
        p0 = np.array([depth_init, lam_init, 0.25])
        lo = np.array([eps, rest - 2.0, 0.03])
        hi = np.array([1.0, rest + 2.0, 4.0])

    elif model_name == "Lorentzian":
        p0 = np.array([depth_init, lam_init, 0.25])
        lo = np.array([eps, rest - 2.0, 0.03])
        hi = np.array([1.0, rest + 2.0, 4.0])

    elif model_name == "Voigt":
        p0 = np.array([depth_init, lam_init, 0.20, 0.20])
        lo = np.array([eps, rest - 2.0, 0.03, 0.03])
        hi = np.array([1.0, rest + 2.0, 4.0, 4.0])

    elif model_name == "DoubleGaussian":
        p0 = np.array([
            depth_init * 0.60, lam_init - 0.05, 0.18,
            depth_init * 0.40, lam_init + 0.05, 0.35,
        ])
        lo = np.array([eps, rest - 2.0, 0.03, eps, rest - 2.0, 0.03])
        hi = np.array([1.0, rest + 2.0, 4.0, 1.0, rest + 2.0, 4.0])

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return p0, lo, hi


def fit_line_profile_model(model_name, wf, ff, p0, lo, hi, sigma_local):
    func = LINE_PROFILE_MODELS[model_name]["func"]

    scale = hi - lo
    scale = np.where(scale > 0, scale, 1.0)

    def residuals(x):
        p = lo + x * scale
        return func(wf, *p) - ff

    x0 = (p0 - lo) / scale
    f_scale = max(3.0 * sigma_local, 0.01) if np.isfinite(sigma_local) else 0.03

    res = least_squares(
        residuals,
        x0,
        bounds=(np.zeros_like(x0), np.ones_like(x0)),
        loss="huber",
        f_scale=f_scale,
        method="trf",
        max_nfev=30000,
    )

    popt = lo + res.x * scale
    model = func(wf, *popt)
    resid = ff - model
    ok = res.success or res.cost < 1e-4

    return popt, model, resid, ok


def measure_line_profile_from_fit(wf, flux_for_ew, model_name, popt, sigma_local, rest):
    func = LINE_PROFILE_MODELS[model_name]["func"]

    wd = np.linspace(wf.min(), wf.max(), 6000)
    md = func(wd, *popt)

    continuum = 1.0
    center_fit = absorption_centroid(wd, md, continuum=continuum)
    depth_fit = float(np.nanmax(continuum - md))
    dv = (center_fit - rest) / rest * 299792.458 if np.isfinite(center_fit) else np.nan

    fwhm, _, _ = model_fwhm_generic(wd, md, continuum=continuum)

    bounds = find_line_boundaries(
        wd,
        md,
        continuum,
        sigma_local,
        k_sigma=K_SIGMA_LINE,
        frac_depth=FRAC_DEPTH_LINE,
    )

    ew_obs = ew_from_observed(wf, flux_for_ew, bounds["lam_left"], bounds["lam_right"])
    ew_err = ew_err_analytic(sigma_local, wf, bounds["lam_left"], bounds["lam_right"])

    return {
        "continuum": continuum,
        "center_fit": center_fit,
        "depth_fit": depth_fit,
        "fwhm": fwhm,
        "dv": dv,
        "wave_dense": wd,
        "model_dense": md,
        "lam_left": bounds["lam_left"],
        "lam_right": bounds["lam_right"],
        "left_truncated": bounds["left_truncated"],
        "right_truncated": bounds["right_truncated"],
        "truncated": bounds["truncated"],
        "ew_obs": ew_obs,
        "ew_err": ew_err,
    }


def select_best_line_profile_model(rows):
    valid = [r for r in rows if r["Success"] and np.isfinite(r["BIC"])]
    if not valid:
        return None

    valid_sorted = sorted(valid, key=lambda x: x["BIC"])
    best = valid_sorted[0]["Model"]
    best_bic = valid_sorted[0]["BIC"]

    near = [r for r in valid if (r["BIC"] - best_bic) < DBIC_SIMPLE_EQUIV]
    if near:
        best = sorted(near, key=lambda x: (x["k"], x["BIC"]))[0]["Model"]

    return best


def classify_line_profile(rows, recommended):
    if recommended is None:
        return "unknown"

    if recommended != "DoubleGaussian":
        return "single"

    valid_single = [
        r for r in rows
        if r["Success"] and r["Model"] in {"Gaussian", "Lorentzian", "Voigt"} and np.isfinite(r["BIC"])
    ]
    valid_blend = [
        r for r in rows
        if r["Success"] and r["Model"] == "DoubleGaussian" and np.isfinite(r["BIC"])
    ]

    if not valid_blend:
        return "unknown"
    if not valid_single:
        return "blend_like"

    bic_blend = min(r["BIC"] for r in valid_blend)
    bic_single = min(r["BIC"] for r in valid_single)
    d_bic = bic_single - bic_blend

    if d_bic >= DBIC_BLEND_STRONG:
        return "blend_like"
    if d_bic >= DBIC_SIMPLE_EQUIV:
        return "ambiguous"
    return "single"


# ============================================================
# MONTE CARLO: GENERAL LINES
# ============================================================

def _line_profile_mc_bounds(model_name, best_popt):
    p = np.array(best_popt, dtype=float)
    lo = p.copy()
    hi = p.copy()

    if model_name in {"Gaussian", "Lorentzian"}:
        lo[0] = max(0.001, p[0] * 0.4)
        hi[0] = min(1.0, p[0] * 1.8 + 0.02)
        lo[1] = p[1] - 1.0
        hi[1] = p[1] + 1.0
        lo[2] = max(0.03, p[2] * 0.4)
        hi[2] = max(lo[2] + 1e-3, p[2] * 1.8 + 0.05)

    elif model_name == "Voigt":
        lo[0] = max(0.001, p[0] * 0.4)
        hi[0] = min(1.0, p[0] * 1.8 + 0.02)
        lo[1] = p[1] - 1.0
        hi[1] = p[1] + 1.0
        lo[2] = max(0.03, p[2] * 0.4)
        hi[2] = max(lo[2] + 1e-3, p[2] * 1.8 + 0.05)
        lo[3] = max(0.03, p[3] * 0.4)
        hi[3] = max(lo[3] + 1e-3, p[3] * 1.8 + 0.05)

    elif model_name == "DoubleGaussian":
        lo[0] = max(0.001, p[0] * 0.4)
        hi[0] = min(1.0, p[0] * 1.8 + 0.02)
        lo[1] = p[1] - 1.0
        hi[1] = p[1] + 1.0
        lo[2] = max(0.03, p[2] * 0.4)
        hi[2] = max(lo[2] + 1e-3, p[2] * 1.8 + 0.05)
        lo[3] = max(0.001, p[3] * 0.4)
        hi[3] = min(1.0, p[3] * 1.8 + 0.02)
        lo[4] = p[4] - 1.0
        hi[4] = p[4] + 1.0
        lo[5] = max(0.03, p[5] * 0.4)
        hi[5] = max(lo[5] + 1e-3, p[5] * 1.8 + 0.05)

    else:
        raise ValueError(f"Unknown model for MC: {model_name}")

    return lo, hi


def estimate_line_profile_uncertainties(
    wf,
    ff_n,
    model_name,
    best_popt,
    sigma_local,
    rest,
    n_mc=N_MC_LINE,
    rng_seed=RNG_SEED,
):
    if not np.isfinite(sigma_local) or sigma_local <= 0:
        return {
            "center_err": np.nan,
            "dv_err": np.nan,
            "depth_err": np.nan,
            "fwhm_err": np.nan,
            "ew_err_mc": np.nan,
            "lam_left_err": np.nan,
            "lam_right_err": np.nan,
            "n_mc_ok": 0,
        }

    vals_center = []
    vals_dv = []
    vals_depth = []
    vals_fwhm = []
    vals_ew = []
    vals_ll = []
    vals_lr = []

    rng = np.random.default_rng(rng_seed)
    p0_base = np.array(best_popt, dtype=float)
    lo, hi = _line_profile_mc_bounds(model_name, p0_base)

    for _ in range(n_mc):
        ff_mc = ff_n + rng.normal(0.0, sigma_local, size=len(ff_n))

        try:
            popt_mc, _, _, ok_mc = fit_line_profile_model(
                model_name=model_name,
                wf=wf,
                ff=ff_mc,
                p0=p0_base,
                lo=lo,
                hi=hi,
                sigma_local=sigma_local,
            )
        except Exception:
            continue

        if not ok_mc:
            continue

        meas = measure_line_profile_from_fit(
            wf=wf,
            flux_for_ew=ff_mc,
            model_name=model_name,
            popt=popt_mc,
            sigma_local=sigma_local,
            rest=rest,
        )

        vals_center.append(meas["center_fit"])
        vals_dv.append(meas["dv"])
        vals_depth.append(meas["depth_fit"])
        vals_fwhm.append(meas["fwhm"])
        vals_ew.append(meas["ew_obs"])
        vals_ll.append(meas["lam_left"])
        vals_lr.append(meas["lam_right"])

    def _std_or_nan(arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < max(10, int(0.3 * n_mc)):
            return np.nan
        return float(np.std(arr, ddof=1))

    return {
        "center_err": _std_or_nan(vals_center),
        "dv_err": _std_or_nan(vals_dv),
        "depth_err": _std_or_nan(vals_depth),
        "fwhm_err": _std_or_nan(vals_fwhm),
        "ew_err_mc": _std_or_nan(vals_ew),
        "lam_left_err": _std_or_nan(vals_ll),
        "lam_right_err": _std_or_nan(vals_lr),
        "n_mc_ok": len(vals_center),
    }


# ============================================================
# MONTE CARLO: BROAD COMPLEX (BALMER ROUTING)
# ============================================================

def _broad_complex_bounds_from_best(model_name, best_popt):
    p = np.array(best_popt, dtype=float)
    lo = p.copy()
    hi = p.copy()

    if model_name == "M1_1Gauss":
        lo[0] = max(0.90, p[0] - 0.05)
        hi[0] = min(1.10, p[0] + 0.05)
        lo[1] = max(0.001, p[1] * 0.4)
        hi[1] = min(1.000, p[1] * 1.8 + 0.02)
        lo[2] = p[2] - 1.5
        hi[2] = p[2] + 1.5
        lo[3] = max(0.2, p[3] * 0.4)
        hi[3] = max(lo[3] + 1e-3, p[3] * 1.8 + 0.1)

    elif model_name == "M2_1Voigt":
        lo[0] = max(0.90, p[0] - 0.05)
        hi[0] = min(1.10, p[0] + 0.05)
        lo[1] = max(0.001, p[1] * 0.4)
        hi[1] = min(1.000, p[1] * 1.8 + 0.02)
        lo[2] = p[2] - 1.5
        hi[2] = p[2] + 1.5
        lo[3] = max(0.2, p[3] * 0.4)
        hi[3] = max(lo[3] + 1e-3, p[3] * 1.8 + 0.1)
        lo[4] = max(0.2, p[4] * 0.4)
        hi[4] = max(lo[4] + 1e-3, p[4] * 1.8 + 0.1)

    elif model_name == "M3_2Gauss":
        lo[0] = max(0.90, p[0] - 0.05)
        hi[0] = min(1.10, p[0] + 0.05)
        lo[1] = max(0.001, p[1] * 0.4)
        hi[1] = min(1.000, p[1] * 1.8 + 0.02)
        lo[2] = p[2] - 1.5
        hi[2] = p[2] + 1.5
        lo[3] = max(0.3, p[3] * 0.4)
        hi[3] = max(lo[3] + 1e-3, p[3] * 1.8 + 0.1)
        lo[4] = max(0.001, p[4] * 0.4)
        hi[4] = min(1.000, p[4] * 1.8 + 0.02)
        lo[5] = p[5] - 1.0
        hi[5] = p[5] + 1.0
        lo[6] = max(0.05, p[6] * 0.4)
        hi[6] = max(lo[6] + 1e-3, p[6] * 1.8 + 0.05)

    elif model_name == "M4_VoigtGauss":
        lo[0] = max(0.90, p[0] - 0.05)
        hi[0] = min(1.10, p[0] + 0.05)
        lo[1] = max(0.001, p[1] * 0.4)
        hi[1] = min(1.000, p[1] * 1.8 + 0.02)
        lo[2] = p[2] - 1.5
        hi[2] = p[2] + 1.5
        lo[3] = max(0.3, p[3] * 0.4)
        hi[3] = max(lo[3] + 1e-3, p[3] * 1.8 + 0.1)
        lo[4] = max(0.2, p[4] * 0.4)
        hi[4] = max(lo[4] + 1e-3, p[4] * 1.8 + 0.1)
        lo[5] = max(0.001, p[5] * 0.4)
        hi[5] = min(1.000, p[5] * 1.8 + 0.02)
        lo[6] = p[6] - 1.0
        hi[6] = p[6] + 1.0
        lo[7] = max(0.05, p[7] * 0.4)
        hi[7] = max(lo[7] + 1e-3, p[7] * 1.8 + 0.05)

    else:
        raise ValueError(f"Unknown broad-complex model: {model_name}")

    return lo, hi


def _measure_broad_complex_from_fit(wf, flux_for_ew, model_name, popt, sigma_local, rest):
    func, _ = BALMER_MODELS[model_name]

    wd_margin = (wf.max() - wf.min()) * 0.05
    wd = np.linspace(wf.min() - wd_margin, wf.max() + wd_margin, 10000)
    md = func(wd, popt)

    c0 = float(popt[0])
    center_fit = float(wd[np.argmin(md)])
    depth_fit = c0 - float(np.nanmin(md))

    fwhm, _, _ = balmer_model_fwhm(wd, md, c0)
    dv = (center_fit - rest) / rest * 299792.458

    comps = balmer_components(model_name, popt, wd)
    ew = balmer_measure_ew(
        wf=wf,
        fn=flux_for_ew,
        wd=wd,
        md=md,
        sigma_noise=sigma_local,
        cont=c0,
        comps=comps,
    )

    return {
        "continuum": c0,
        "center_fit": center_fit,
        "depth_fit": depth_fit,
        "fwhm": fwhm,
        "dv": dv,
        "wave_dense": wd,
        "model_dense": md,
        "components": comps,
        "lam_left": ew.get("lam_left", np.nan),
        "lam_right": ew.get("lam_right", np.nan),
        "ew_obs": ew.get("ew_obs", np.nan),
        "ew_err": ew.get("ew_err", np.nan),
        "ew_components": ew.get("ew_components", []),
        "truncated": ew.get("truncated", False),
        "left_truncated": ew.get("left_truncated", False),
        "right_truncated": ew.get("right_truncated", False),
    }


def estimate_broad_complex_uncertainties(
    wf,
    ff_n,
    model_name,
    best_popt,
    sigma_local,
    rest,
    n_mc=N_MC_BROAD_COMPLEX,
    rng_seed=RNG_SEED,
):
    if not np.isfinite(sigma_local) or sigma_local <= 0:
        return {
            "center_err": np.nan,
            "dv_err": np.nan,
            "depth_err": np.nan,
            "fwhm_err": np.nan,
            "ew_err_mc": np.nan,
            "lam_left_err": np.nan,
            "lam_right_err": np.nan,
            "n_mc_ok": 0,
        }

    vals_center = []
    vals_dv = []
    vals_depth = []
    vals_fwhm = []
    vals_ew = []
    vals_ll = []
    vals_lr = []

    rng = np.random.default_rng(rng_seed)
    func, _ = BALMER_MODELS[model_name]

    p0_base = np.array(best_popt, dtype=float)
    lo, hi = _broad_complex_bounds_from_best(model_name, p0_base)

    for _ in range(n_mc):
        ff_mc = ff_n + rng.normal(0.0, sigma_local, size=len(ff_n))

        try:
            popt_mc, ok_mc, _, _ = balmer_fit_robust(
                name=model_name,
                func=func,
                wf=wf,
                ff=ff_mc,
                p0=p0_base,
                lo=lo,
                hi=hi,
                sigma_local=sigma_local,
            )
        except Exception:
            continue

        if not ok_mc:
            continue

        meas = _measure_broad_complex_from_fit(
            wf=wf,
            flux_for_ew=ff_mc,
            model_name=model_name,
            popt=popt_mc,
            sigma_local=sigma_local,
            rest=rest,
        )

        vals_center.append(meas["center_fit"])
        vals_dv.append(meas["dv"])
        vals_depth.append(meas["depth_fit"])
        vals_fwhm.append(meas["fwhm"])
        vals_ew.append(meas["ew_obs"])
        vals_ll.append(meas["lam_left"])
        vals_lr.append(meas["lam_right"])

    def _std_or_nan(arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < max(10, int(0.3 * n_mc)):
            return np.nan
        return float(np.std(arr, ddof=1))

    return {
        "center_err": _std_or_nan(vals_center),
        "dv_err": _std_or_nan(vals_dv),
        "depth_err": _std_or_nan(vals_depth),
        "fwhm_err": _std_or_nan(vals_fwhm),
        "ew_err_mc": _std_or_nan(vals_ew),
        "lam_left_err": _std_or_nan(vals_ll),
        "lam_right_err": _std_or_nan(vals_lr),
        "n_mc_ok": len(vals_center),
    }


# ============================================================
# PLOTTING FOR GENERAL LINES
# ============================================================

def _pub():
    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    })


def plot_general_profile(wf, ff, result, line_id, label, rest, output_dir, obj=None):
    _pub()

    wd = result["wave_dense"]
    md = result["model_dense"]
    ll = result["lam_left"]
    lr = result["lam_right"]

    if np.isfinite(ll) and np.isfinite(lr):
        span = lr - ll
        xpad = max(0.35 * span, 0.8)
        x0, x1 = ll - xpad, lr + xpad
    else:
        x0, x1 = wf.min(), wf.max()

    pm = (wd >= x0) & (wd <= x1)
    wm = (wf >= x0) & (wf <= x1)

    wp = wf[wm]
    fp = ff[wm]
    wdp = wd[pm]
    mdp = md[pm]

    fig, ax = plt.subplots(figsize=(12 / 2.54, 7 / 2.54))
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.18, top=0.88)

    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.9, zorder=1)
    ax.plot(wp, fp, color="black", lw=1.2, zorder=5)
    ax.plot(wdp, mdp, color="red", lw=1.8, ls="--", zorder=6)

    ax.axvline(rest, color="#ff1493", lw=1.1, ls=":", zorder=4, alpha=0.95)

    if np.isfinite(ll):
        ax.axvline(ll, color="#006400", lw=1.0, ls="--", zorder=3, alpha=0.9)
    if np.isfinite(lr):
        ax.axvline(lr, color="#006400", lw=1.0, ls="--", zorder=3, alpha=0.9)

    y_candidates = [np.nanmin(fp), np.nanmax(fp), np.nanmin(mdp), np.nanmax(mdp), 1.0]
    y_min = np.nanmin(y_candidates)
    y_max = np.nanmax(y_candidates)

    depth = 1.0 - y_min
    bottom_pad = max(0.15 * depth, 0.004)
    top_pad = max(0.08 * depth, 0.003)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y_min - bottom_pad, min(1.01 + top_pad, y_max + top_pad))

    ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]", labelpad=4)
    ax.set_ylabel(r"$\mathrm{I}/\mathrm{I}_{\mathrm{cont}}$", labelpad=4)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))

    title = label if not obj else f"{obj} — {label}"
    ax.set_title(title, fontsize=11, pad=8)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{line_id}_{rest:.2f}".replace(" ", "_")
    for ext in (".png", ".pdf"):
        fig.savefig(output_dir / f"{stem}{ext}", bbox_inches="tight", facecolor="white")

    plt.close(fig)
    return fig


# ============================================================
# ANALYSIS OF ONE GENERAL LINE
# ============================================================

def analyse_line_profile(wave, flux, line_id, label, rest, output_dir=None, obj=None, make_plot=True):
    fit_window = (rest - LINE_HALF_WINDOW, rest + LINE_HALF_WINDOW)
    cont_left = (fit_window[0] - CONT_GAP - CONT_WIDTH, fit_window[0] - CONT_GAP)
    cont_right = (fit_window[1] + CONT_GAP, fit_window[1] + CONT_GAP + CONT_WIDTH)

    empty_result = {
        "kind": "line",
        "line": {"name": label, "ascii": line_id, "rest": rest},
        "empirical": {"sigma_local": np.nan, "snr_depth": np.nan},
        "models": {},
        "recommended": None,
        "figure": None,
        "status": "not_detected",
        "profile_class": "unknown",
    }

    if wave.min() > cont_left[0] or wave.max() < cont_right[1]:
        return empty_result

    mask = (wave >= fit_window[0]) & (wave <= fit_window[1])
    wf = wave[mask].copy()
    ff = flux[mask].copy()

    if len(wf) < 10:
        return empty_result

    level, slope, ref_wave = local_continuum(wave, flux, cont_left, cont_right)
    ff_n = normalise_flux(wf, ff, level, slope, ref_wave)

    wide_mask = (wave >= cont_left[0]) & (wave <= cont_right[1])
    wave_wide = wave[wide_mask]
    flux_wide_n = normalise_flux(wave_wide, flux[wide_mask], level, slope, ref_wave)
    sigma_local = local_sigma(wave_wide, flux_wide_n, cont_left, cont_right)

    win = min(SMOOTH_WIN_LINE, len(ff_n) - 1)
    win = win if win % 2 == 1 else win - 1
    win = max(win, 3)

    fsmooth = savgol_filter(
        ff_n,
        window_length=win,
        polyorder=min(SMOOTH_POLY_LINE, win - 1),
    )

    idx_min = np.argmin(fsmooth)
    lam_init = float(wf[idx_min])
    depth_init = max(1.0 - float(fsmooth[idx_min]), 0.01)

    snr_depth = depth_init / sigma_local if np.isfinite(sigma_local) and sigma_local > 0 else np.nan
    status = classify_detection_status(snr_depth)

    if status == "not_detected":
        empty_result["empirical"] = {"sigma_local": sigma_local, "snr_depth": snr_depth}
        return empty_result

    print(f"\n{'-' * 56}")
    print(f"  {label} -- Automatic profile model comparison")
    print(f"{'-' * 56}")

    results = {}
    rows = []

    for model_name in LINE_PROFILE_MODELS.keys():
        p0, lo, hi = line_profile_init_bounds(model_name, rest, lam_init, depth_init)
        popt, _, resid, ok = fit_line_profile_model(
            model_name=model_name,
            wf=wf,
            ff=ff_n,
            p0=p0,
            lo=lo,
            hi=hi,
            sigma_local=sigma_local,
        )

        _, aic, bic, rms = calc_aic_bic(
            resid,
            LINE_PROFILE_MODELS[model_name]["k"],
            len(wf),
        )

        meas = measure_line_profile_from_fit(
            wf=wf,
            flux_for_ew=ff_n,
            model_name=model_name,
            popt=popt,
            sigma_local=sigma_local,
            rest=rest,
        )

        results[model_name] = {
            "success": ok,
            "popt": popt,
            "sigma_local": sigma_local,
            "snr_depth": snr_depth,
            "rms": rms,
            "aic": aic,
            "bic": bic,
            "k": LINE_PROFILE_MODELS[model_name]["k"],
            **meas,
        }

        rows.append({
            "Model": model_name,
            "k": LINE_PROFILE_MODELS[model_name]["k"],
            "Success": ok,
            "AIC": aic,
            "BIC": bic,
            "RMS": rms,
        })

        status_fit = "OK" if ok else "FAIL"
        print(f"  {status_fit:4} {model_name:<14} RMS={rms:.5f}  AIC={aic:.1f}  BIC={bic:.1f}")

    best = select_best_line_profile_model(rows)
    profile_class = classify_line_profile(rows, best)

    if best is None:
        return {
            "kind": "line",
            "line": {"name": label, "ascii": line_id, "rest": rest},
            "empirical": {"sigma_local": sigma_local, "snr_depth": snr_depth},
            "models": {},
            "recommended": None,
            "figure": None,
            "status": status,
            "profile_class": "unknown",
        }

    mc_err = estimate_line_profile_uncertainties(
        wf=wf,
        ff_n=ff_n,
        model_name=best,
        best_popt=results[best]["popt"],
        sigma_local=sigma_local,
        rest=rest,
        n_mc=N_MC_LINE,
        rng_seed=RNG_SEED,
    )
    results[best].update(mc_err)

    fig = None
    if make_plot and output_dir is not None:
        fig = plot_general_profile(
            wf=wf,
            ff=ff_n,
            result=results[best],
            line_id=line_id,
            label=label,
            rest=rest,
            output_dir=Path(output_dir),
            obj=obj,
        )

    return {
        "kind": "line",
        "line": {"name": label, "ascii": line_id, "rest": rest},
        "empirical": {"sigma_local": sigma_local, "snr_depth": snr_depth},
        "models": results,
        "recommended": best,
        "figure": fig,
        "status": status,
        "profile_class": profile_class,
    }


# ============================================================
# TARGET-SCOPED ANALYSIS
# ============================================================

def _attach_spectrum_metadata(res, target, src_line):
    if res is None:
        return None

    res["source_line"] = {
        "line_id": src_line["line_id"],
        "label": src_line["label"],
        "rest_input": src_line["rest_input"],
    }

    res["source_spectrum"] = {
        "spectrum_id": target["spectrum_id"],
        "spectrum_kind": target["spectrum_kind"],
        "filename": target.get("filename"),
        "object_name": target.get("object_name"),
    }

    return res


def analyse_target_lines(target, df=None):
    wave = target["wavelength"]
    flux = target["flux"]
    obj_name = target.get("object_name", None)
    spectrum_id = target["spectrum_id"]

    if df is None:
        df = read_line_list()

    all_results = []

    print("\n" + "=" * 70)
    print(f"  LINE ANALYSIS : {spectrum_id} [{target['spectrum_kind']}]")
    print("=" * 70)

    line_output_dir = LINE_FITS_DIR / spectrum_id
    balmer_output_dir = BALMER_DIR / spectrum_id

    for i, row in df.iterrows():
        line_id = str(row["line_id"]).strip()
        label = str(row["label"]).strip()
        rest = float(row["wavelength"])

        is_broad_complex, broad_info = identify_broad_complex_line(line_id, label, rest)

        print(f"\n[{i + 1}/{len(df)}] {line_id} | {label} | {rest:.2f} A")

        if is_broad_complex:
            print("    -> Broad-complex line block")

            plot_prefix = balmer_output_dir / f"broad_complex_line_{OUTPUT_TAG}" if OUTPUT_TAG else balmer_output_dir / "broad_complex_line"

            res = analyse_balmer_line(
                wave,
                flux,
                broad_info,
                plot=SAVE_PLOTS,
                output_path=plot_prefix if SAVE_PLOTS else None,
                obj=obj_name,
            )

            if res is None:
                res = {
                    "kind": "broad_complex",
                    "line": {
                        "name": broad_info["name"],
                        "ascii": broad_info["ascii"],
                        "rest": broad_info["rest"],
                    },
                    "empirical": {"sigma_local": np.nan, "snr_depth": np.nan},
                    "models": {},
                    "recommended": None,
                    "figure": None,
                    "status": "not_detected",
                    "profile_class": "broad_complex",
                }

            res["kind"] = "broad_complex"
            res["profile_class"] = "broad_complex"

            broad_snr = res.get("empirical", {}).get("snr_depth", np.nan)
            res["status"] = classify_detection_status(broad_snr)

            rec = res.get("recommended", None)
            if rec is not None:
                emp = res["empirical"]
                mr = res["models"][rec]

                mc_err = estimate_broad_complex_uncertainties(
                    wf=emp["wave_fit"],
                    ff_n=emp["flux_fit"],
                    model_name=rec,
                    best_popt=mr["popt"],
                    sigma_local=emp["sigma_local"],
                    rest=broad_info["rest"],
                    n_mc=N_MC_BROAD_COMPLEX,
                    rng_seed=RNG_SEED,
                )
                res["models"][rec].update(mc_err)

        else:
            print("    -> Automatic profile model selection")
            res = analyse_line_profile(
                wave=wave,
                flux=flux,
                line_id=line_id,
                label=label,
                rest=rest,
                output_dir=line_output_dir if SAVE_PLOTS else None,
                obj=obj_name,
                make_plot=SAVE_PLOTS,
            )

        src_line = {
            "line_id": line_id,
            "label": label,
            "rest_input": rest,
        }
        res = _attach_spectrum_metadata(res, target, src_line)
        all_results.append(res)

    print("\nDone.")
    return all_results


# ============================================================
# DRIVER
# ============================================================

def run_general_line_analysis(analysis_targets=None):
    LINE_FITS_DIR.mkdir(parents=True, exist_ok=True)
    BALMER_DIR.mkdir(parents=True, exist_ok=True)

    df = read_line_list()

    if analysis_targets is None:
        prep = run_prepare_spectra()
        analysis_targets = prep["analysis_targets"]

    all_results = []

    for target in analysis_targets:
        target_results = analyse_target_lines(target, df=df)
        all_results.extend(target_results)

    return all_results


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "classify_detection_status",
    "identify_broad_complex_line",
    "read_line_list",
    "LINE_PROFILE_MODELS",
    "line_profile_init_bounds",
    "fit_line_profile_model",
    "measure_line_profile_from_fit",
    "select_best_line_profile_model",
    "classify_line_profile",
    "estimate_line_profile_uncertainties",
    "estimate_broad_complex_uncertainties",
    "analyse_line_profile",
    "analyse_target_lines",
    "run_general_line_analysis",
]


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    all_results = run_general_line_analysis()

