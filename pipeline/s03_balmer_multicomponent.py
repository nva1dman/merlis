#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.special import voigt_profile

from s00_settings import (
    AVERAGED_FITS_PATH,
    BALMER_DIR,
    BALMER_LINES,
    DENSE_NPTS_BALMER,
    SMOOTH_WIN_BALMER,
    SMOOTH_POLY_BALMER,
    K_SIGMA_BALMER,
    FRAC_DEPTH_BALMER,
    SNR_DETECT,
    SNR_RELIABLE,
)

from s01_spectrum_utils import (
    load_averaged_fits,
    local_continuum,
    normalise_flux,
    local_sigma,
    find_line_boundaries,
    ew_from_observed,
    ew_err_analytic,
    calc_aic_bic,
    model_fwhm_generic,
    trapz_integral,
)

warnings.filterwarnings("ignore")


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def gauss_profile(w, amplitude, center, sigma):
    return amplitude * np.exp(-0.5 * ((w - center) / sigma) ** 2)


def voigt_peak_profile(w, amplitude, center, sigma, gamma):
    pk = voigt_profile(0.0, sigma, gamma)
    if pk <= 0:
        return np.zeros_like(w)
    return amplitude * voigt_profile(w - center, sigma, gamma) / pk


def model_single_gauss(w, p):
    c0, amp, lam0, sigma = p
    return c0 - gauss_profile(w, amp, lam0, sigma)


def model_single_voigt(w, p):
    c0, amp, lam0, sigma, gamma = p
    return c0 - voigt_peak_profile(w, amp, lam0, sigma, gamma)


def model_double_gauss(w, p):
    c0, amp_b, lam_b, sig_b, amp_n, lam_n, sig_n = p
    return c0 - gauss_profile(w, amp_b, lam_b, sig_b) - gauss_profile(w, amp_n, lam_n, sig_n)


def model_voigt_plus_gauss(w, p):
    c0, amp_b, lam_b, sig_b, gam_b, amp_n, lam_n, sig_n = p
    return c0 - voigt_peak_profile(w, amp_b, lam_b, sig_b, gam_b) - gauss_profile(w, amp_n, lam_n, sig_n)


BALMER_MODELS = {
    "M1_1Gauss":     (model_single_gauss, 4),
    "M2_1Voigt":     (model_single_voigt, 5),
    "M3_2Gauss":     (model_double_gauss, 7),
    "M4_VoigtGauss": (model_voigt_plus_gauss, 8),
}

BALMER_DISPLAY = {
    "M1_1Gauss":     "Single Gaussian",
    "M2_1Voigt":     "Single Voigt",
    "M3_2Gauss":     "Double Gaussian",
    "M4_VoigtGauss": "Voigt + Gaussian",
}

BALMER_SHORT = {
    "M1_1Gauss":     "1-Gauss",
    "M2_1Voigt":     "1-Voigt",
    "M3_2Gauss":     "2-Gauss",
    "M4_VoigtGauss": "Voigt+Gauss",
}


# ============================================================
# EMPIRICAL MEASUREMENTS
# ============================================================

def _fwhm_empirical(wave, flux_smooth, continuum=1.0):
    f_min = np.nanmin(flux_smooth)
    depth = continuum - f_min
    if depth <= 0:
        return np.nan, np.nan, np.nan

    f_half = continuum - depth / 2.0
    idx_min = np.argmin(flux_smooth)

    def _cross(sw, sf, lev):
        above = sf >= lev
        idx = np.where(np.diff(above.astype(int)))[0]
        if len(idx) == 0:
            return np.nan

        i = idx[-1]
        if sf[i + 1] == sf[i]:
            return sw[i]

        return sw[i] + (lev - sf[i]) / (sf[i + 1] - sf[i]) * (sw[i + 1] - sw[i])

    wl_l = _cross(wave[:idx_min + 1], flux_smooth[:idx_min + 1], f_half)
    wl_r = _cross(wave[idx_min:][::-1], flux_smooth[idx_min:][::-1], f_half)

    if not (np.isfinite(wl_l) and np.isfinite(wl_r)):
        return np.nan, np.nan, np.nan

    return float(wl_r - wl_l), float(wl_l), float(wl_r)


def balmer_measure_empirical(
    wave_full,
    flux_full,
    line_info,
    smooth_win=SMOOTH_WIN_BALMER,
    smooth_poly=SMOOTH_POLY_BALMER,
):
    rest = line_info["rest"]
    cont_left = line_info["cont_left"]
    cont_right = line_info["cont_right"]
    fit_window = line_info["fit_window"]

    mask = (wave_full >= fit_window[0]) & (wave_full <= fit_window[1])
    wf = wave_full[mask].copy()
    ff = flux_full[mask].copy()

    if len(wf) < 10:
        return None

    level, slope, ref_wave = local_continuum(
        wave_full, flux_full, cont_left, cont_right
    )
    ff_n = normalise_flux(wf, ff, level, slope, ref_wave)

    wide_mask = (wave_full >= cont_left[0]) & (wave_full <= cont_right[1])
    wave_wide = wave_full[wide_mask]
    flux_wide_n = normalise_flux(
        wave_wide, flux_full[wide_mask], level, slope, ref_wave
    )

    sigma_local = local_sigma(wave_wide, flux_wide_n, cont_left, cont_right)

    win = min(smooth_win, len(ff_n) - 1)
    win = win if win % 2 == 1 else win - 1
    win = max(win, 3)

    fsmooth = savgol_filter(
        ff_n,
        window_length=win,
        polyorder=min(smooth_poly, win - 1),
    )

    idx_min = np.argmin(fsmooth)
    center_obs = float(wf[idx_min])
    depth_obs = float(1.0 - fsmooth[idx_min])

    fwhm_obs, wl_l, wl_r = _fwhm_empirical(wf, fsmooth)

    dv_obs = (center_obs - rest) / rest * 299792.458
    snr_depth = (
        depth_obs / sigma_local
        if np.isfinite(sigma_local) and sigma_local > 0
        else np.nan
    )

    line_name = line_info["name"]
    print(f"\n{'=' * 56}")
    print(f"  {line_name} -- Empirical (smoothed)")
    print(f"{'=' * 56}")
    print(f"  Rest          : {rest:.2f} A")
    print(f"  Centre (obs)  : {center_obs:.3f} A  (dv={dv_obs:+.1f} km/s)")
    print(f"  Depth (obs)   : {depth_obs:.4f}")
    if np.isfinite(fwhm_obs):
        print(f"  FWHM (obs)    : {fwhm_obs:.3f} A")
    print(f"  Local sigma   : {sigma_local:.5f}")
    if np.isfinite(snr_depth):
        flag = ""
        if snr_depth < SNR_DETECT:
            flag = "  !! below detection"
        elif snr_depth < SNR_RELIABLE:
            flag = "  ! marginal"
        print(f"  Depth/sigma   : {snr_depth:.1f}{flag}")
    print(f"{'=' * 56}")

    return {
        "wave_fit": wf,
        "flux_fit": ff_n,
        "flux_smooth": fsmooth,
        "wave_wide": wave_wide,
        "flux_wide_n": flux_wide_n,
        "center_obs": center_obs,
        "depth_obs": depth_obs,
        "fwhm_obs": fwhm_obs,
        "continuum_level": level,
        "continuum_slope": slope,
        "ref_wave": ref_wave,
        "dv_obs": dv_obs,
        "sigma_local": sigma_local,
        "snr_depth": snr_depth,
    }


# ============================================================
# FIT INITIALIZATION / FITTING
# ============================================================

def balmer_init_bounds(model_name, lam_min, amp0, wf, ff, rest):
    l0 = lam_min
    ll, lh = rest - 6.0, rest + 6.0
    eps = 0.02

    wing_mask = (np.abs(wf - l0) >= 3.0) & (np.abs(wf - l0) <= 6.0)
    if wing_mask.sum() >= 4:
        amp_b = float(np.clip(1.0 - np.nanmedian(ff[wing_mask]), eps, 0.7))
    else:
        amp_b = amp0 * 0.4

    amp_n = float(np.clip(amp0 - amp_b, eps, 0.8))

    sig_n_min, sig_n_max = 0.3, 2.0
    sig_b_min, sig_b_max = 2.0, 15.0

    if model_name == "M1_1Gauss":
        p0 = [1.0, amp0, l0, 3.0]
        lo = [0.95, eps, ll, 0.8]
        hi = [1.05, 1.0, lh, 15.0]

    elif model_name == "M2_1Voigt":
        p0 = [1.0, amp0, l0, 2.5, 1.5]
        lo = [0.95, eps, ll, 0.3, 0.3]
        hi = [1.05, 1.0, lh, 12.0, 12.0]

    elif model_name == "M3_2Gauss":
        p0 = [1.0, amp_b, l0, 4.0, amp_n, l0, 0.8]
        lo = [0.95, eps, ll, sig_b_min, eps, l0 - 1.0, sig_n_min]
        hi = [1.05, 0.8, lh, sig_b_max, 0.8, l0 + 1.0, sig_n_max]

    elif model_name == "M4_VoigtGauss":
        p0 = [1.0, amp_b, l0, 2.5, 1.5, amp_n, l0, 0.8]
        lo = [0.95, eps, ll, sig_b_min, 0.3, eps, l0 - 1.0, sig_n_min]
        hi = [1.05, 0.8, lh, sig_b_max, sig_b_max, 0.8, l0 + 1.0, sig_n_max]

    else:
        raise ValueError(f"Unknown Balmer model: {model_name}")

    return np.array(p0, dtype=float), np.array(lo, dtype=float), np.array(hi, dtype=float)


def balmer_fit_robust(name, func, wf, ff, p0, lo, hi, sigma_local):
    scale = hi - lo
    scale = np.where(scale > 0, scale, 1.0)

    def residuals(x):
        p = np.clip(lo + x * scale, lo, hi)
        return func(wf, p) - ff

    x0 = (p0 - lo) / scale
    f_scale = max(3.0 * sigma_local, 0.01) if np.isfinite(sigma_local) else 0.05

    try:
        res = least_squares(
            residuals,
            x0,
            bounds=(np.zeros_like(x0), np.ones_like(x0)),
            loss="huber",
            f_scale=f_scale,
            max_nfev=50000,
            method="trf",
        )
        popt = lo + res.x * scale
        ok = res.success or res.cost < 1e-4
    except Exception as exc:
        print(f"  [{name}] FAIL: {exc}")
        return p0, False, np.full(len(ff), np.nan), np.full(len(ff), np.nan)

    model = func(wf, popt)
    residual = ff - model
    return popt, ok, residual, model


def balmer_components(model_name, p, w):
    if model_name == "M1_1Gauss":
        return [("Gaussian", gauss_profile(w, p[1], p[2], p[3]))]

    if model_name == "M2_1Voigt":
        return [("Voigt", voigt_peak_profile(w, p[1], p[2], p[3], p[4]))]

    if model_name == "M3_2Gauss":
        return [
            ("broad Gaussian", gauss_profile(w, p[1], p[2], p[3])),
            ("narrow Gaussian", gauss_profile(w, p[4], p[5], p[6])),
        ]

    if model_name == "M4_VoigtGauss":
        return [
            ("broad Voigt", voigt_peak_profile(w, p[1], p[2], p[3], p[4])),
            ("narrow Gaussian", gauss_profile(w, p[5], p[6], p[7])),
        ]

    return []


def balmer_model_fwhm(wd, md, continuum):
    return model_fwhm_generic(wd, md, continuum=continuum)


# ============================================================
# EQUIVALENT WIDTH
# ============================================================

def _ew_from_components(wd, components, continuum, lam_left, lam_right):
    mask = (wd >= lam_left) & (wd <= lam_right)
    out = []

    for label, comp in components:
        out.append((label, float(trapz_integral(comp[mask] / continuum, wd[mask]))))

    return out


def balmer_measure_ew(
    wf,
    fn,
    wd,
    md,
    sigma_noise,
    cont,
    comps,
    k_sigma=K_SIGMA_BALMER,
    frac_depth=FRAC_DEPTH_BALMER,
):
    bounds = find_line_boundaries(
        wd,
        md,
        cont,
        sigma_noise,
        k_sigma=k_sigma,
        frac_depth=frac_depth,
    )

    ll = bounds["lam_left"]
    lr = bounds["lam_right"]

    return {
        "lam_left": ll,
        "lam_right": lr,
        "ew_obs": ew_from_observed(wf, fn, ll, lr, continuum=cont),
        "ew_err": ew_err_analytic(sigma_noise, wf, ll, lr),
        "ew_components": _ew_from_components(wd, comps, cont, ll, lr),
        "sigma_noise": sigma_noise,
        "truncated": bounds["truncated"],
        "left_truncated": bounds["left_truncated"],
        "right_truncated": bounds["right_truncated"],
        "threshold": bounds["threshold"],
    }


# ============================================================
# ANALYSE ONE BALMER LINE
# ============================================================

def _print_balmer_comparison(rows, recommended, line_name):
    sep = "-" * 88
    print(f"\n{sep}")
    print(f"  {line_name} -- Model comparison")
    print(sep)
    print(f"  {'Model':<22} {'k':>2}  {'centre':>8}  {'depth':>6}  {'FWHM':>6}  {'EW':>7}  {'dAIC':>7}  {'dBIC':>7}")

    def fmt(v, spec):
        return f"{v:{spec}}" if np.isfinite(v) else "  -  "

    for r in sorted(rows, key=lambda x: x.get("BIC", np.inf)):
        disp = BALMER_DISPLAY.get(r["Model"], r["Model"])
        flag = " <<" if r["Model"] == recommended else ""
        print(
            f"  {disp:<22} {r['k']:>2}  "
            f"{fmt(r['center'], '8.3f')}  "
            f"{fmt(r['depth'], '.4f')}  "
            f"{fmt(r['fwhm'], '.3f')}  "
            f"{fmt(r['ew_obs'], '.3f')}  "
            f"{fmt(r['dAIC'], '+.1f')}  "
            f"{fmt(r['dBIC'], '+.1f')}{flag}"
        )

    print(sep)
    print(f"  Best: {BALMER_DISPLAY.get(recommended, recommended)}")
    print(sep)


def analyse_balmer_line(wave, flux, line_info, plot=True, output_path=None, obj=None):
    rest = line_info["rest"]
    cont_left = line_info["cont_left"]
    cont_right = line_info["cont_right"]
    line_name = line_info["name"]

    if wave.min() > cont_left[0] or wave.max() < cont_right[1]:
        print(f"\n  !! {line_name} outside spectral range")
        return None

    empirical = balmer_measure_empirical(wave, flux, line_info)
    if empirical is None:
        return None

    wf = empirical["wave_fit"]
    ff = empirical["flux_fit"]
    sigma_local = empirical["sigma_local"]
    snr = empirical["snr_depth"]

    if np.isfinite(snr) and snr < SNR_DETECT:
        print(f"  !! {line_name} depth/sigma={snr:.1f} < {SNR_DETECT}, skip")
        return None

    print(f"\n{'-' * 56}")
    print(f"  {line_name} -- Model comparison (Huber, TRF)")
    print(f"{'-' * 56}")

    idx_min = np.argmin(ff)
    lam_min = float(wf[idx_min])
    amp0 = max(1.0 - float(ff[idx_min]), 0.02)

    model_results = {}
    rows = []

    for model_name, (func, kpars) in BALMER_MODELS.items():
        p0, lo, hi = balmer_init_bounds(model_name, lam_min, amp0, wf, ff, rest)
        popt, ok, residuals, mdata = balmer_fit_robust(
            model_name, func, wf, ff, p0, lo, hi, sigma_local
        )

        wd_margin = (wf.max() - wf.min()) * 0.05
        wd = np.linspace(wf.min() - wd_margin, wf.max() + wd_margin, DENSE_NPTS_BALMER)
        md = func(wd, popt)
        cont = float(popt[0])

        redchi, aic, bic, rms = calc_aic_bic(residuals, kpars, len(wf))
        depth_fit = cont - float(np.nanmin(md))
        fwhm, wl_left, wl_right = balmer_model_fwhm(wd, md, cont)
        center_fit = float(wd[np.argmin(md)])
        dv_fit = (center_fit - rest) / rest * 299792.458

        components = balmer_components(model_name, popt, wd)

        if ok:
            ew = balmer_measure_ew(
                wf, ff, wd, md, sigma_local, cont, components
            )
        else:
            ew = {
                "lam_left": np.nan,
                "lam_right": np.nan,
                "ew_obs": np.nan,
                "ew_err": np.nan,
                "ew_components": [],
                "sigma_noise": sigma_local,
                "truncated": False,
                "left_truncated": False,
                "right_truncated": False,
                "threshold": np.nan,
            }

        model_results[model_name] = {
            "success": ok,
            "popt": popt,
            "k": kpars,
            "n": len(wf),
            "residuals": residuals,
            "wave_dense": wd,
            "model_dense": md,
            "components": components,
            "continuum": cont,
            "center_fit": center_fit,
            "depth_fit": depth_fit,
            "fwhm": fwhm,
            "wl_left": wl_left,
            "wl_right": wl_right,
            "rss_dof": redchi,
            "aic": aic,
            "bic": bic,
            "rms": rms,
            "dv": dv_fit,
            **ew,
        }

        rows.append({
            "Model": model_name,
            "k": kpars,
            "Success": ok,
            "center": center_fit,
            "dv": dv_fit,
            "depth": depth_fit,
            "fwhm": fwhm,
            "ew_obs": ew["ew_obs"],
            "rss_dof": redchi,
            "AIC": aic,
            "BIC": bic,
            "RMS": rms,
        })

        status = "OK" if ok else "FAIL"
        ew_str = f"EW={ew['ew_obs']:.3f}" if np.isfinite(ew["ew_obs"]) else "EW=-"
        trunc = " !!TRUNC" if ew.get("truncated", False) else ""
        print(
            f"  {status:4} {BALMER_DISPLAY.get(model_name, model_name):<22} "
            f"k={kpars}  RMS={rms:.5f}  AIC={aic:.1f}  BIC={bic:.1f}  {ew_str}{trunc}"
        )

    valid = [r for r in rows if r["Success"] and np.isfinite(r["BIC"])]
    if not valid:
        print(f"  All fits failed for {line_name}")
        return {
            "line": line_info,
            "empirical": empirical,
            "models": model_results,
            "stats": rows,
            "recommended": None,
            "figure": None,
        }

    best_bic = min(valid, key=lambda r: r["BIC"])["Model"]
    min_bic = min(r["BIC"] for r in valid)
    min_aic = min(r["AIC"] for r in valid)

    for r in rows:
        r["dAIC"] = r["AIC"] - min_aic
        r["dBIC"] = r["BIC"] - min_bic

    recommended = best_bic
    near = [r for r in rows if r["Success"] and (r["dBIC"] < 2.0)]
    if near:
        recommended = sorted(near, key=lambda x: (x["k"], x["BIC"]))[0]["Model"]

    _print_balmer_comparison(rows, recommended, line_name)

    fig = None
    if plot:
        fig = plot_balmer_line(
            wf,
            ff,
            empirical,
            model_results,
            line_info,
            recommended,
            obj=obj,
            output_path=output_path,
        )

    return {
        "line": line_info,
        "empirical": empirical,
        "models": model_results,
        "stats": rows,
        "recommended": recommended,
        "figure": fig,
    }


# ============================================================
# PLOTTING
# ============================================================

def _pub():
    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "font.size": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "grid.color": "#CCCCCC",
        "grid.linewidth": 0.3,
        "grid.alpha": 0.25,
        "axes.grid": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#999999",
    })


def _nice_limits(ax):
    fig = ax.get_figure()
    fig.canvas.draw()

    yticks = ax.yaxis.get_major_locator().tick_values(*ax.get_ylim())
    yticks = yticks[(yticks >= ax.get_ylim()[0] - 0.01) & (yticks <= ax.get_ylim()[1] + 0.01)]
    if len(yticks) >= 2:
        step = yticks[1] - yticks[0]
        ax.set_ylim(yticks[0] - step * 0.45, yticks[-1] + step * 0.45)

    xticks = ax.xaxis.get_major_locator().tick_values(*ax.get_xlim())
    xticks = xticks[(xticks >= ax.get_xlim()[0] - 1) & (xticks <= ax.get_xlim()[1] + 1)]
    if len(xticks) >= 2:
        step = xticks[1] - xticks[0]
        ax.set_xlim(xticks[0] - step * 0.45, xticks[-1] + step * 0.45)


def plot_balmer_line(wf, ff, empirical, model_results, line_info, recommended, obj=None, output_path=None):
    _pub()

    result = model_results[recommended]
    rest = line_info["rest"]
    line_name = line_info["name"]
    model_label = BALMER_DISPLAY.get(recommended, recommended)

    ll = result.get("lam_left", np.nan)
    lr = result.get("lam_right", np.nan)

    if np.isfinite(ll) and np.isfinite(lr):
        span = lr - ll
        margin = max(span * 0.15, 1.5)
        plot_window = (ll - margin, lr + margin)
    else:
        plot_window = (line_info["cont_left"][0], line_info["cont_right"][1])

    fig, ax = plt.subplots(1, 1, figsize=(14 / 2.54, 8 / 2.54))
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.14, top=0.90)

    mask_obs = (wf >= plot_window[0]) & (wf <= plot_window[1])
    wp = wf[mask_obs]
    fp = ff[mask_obs]

    if len(wp) < 3:
        wp, fp = wf, ff
        plot_window = (wf.min(), wf.max())

    ax.axhline(1.0, color="#696969", ls="--", lw=0.9, zorder=1)
    ax.plot(wp, fp, color="black", lw=1.0, zorder=5, label="Observed")

    if result["success"]:
        cont = result["continuum"]
        wd = result["wave_dense"]
        md = result["model_dense"]

        mask_model = (wd >= plot_window[0]) & (wd <= plot_window[1])

        comp_colors = ["#800080", "#0000FF", "#00AA00", "#FF1493"]
        for i, (clab, comp) in enumerate(result["components"]):
            cc = comp_colors[i % len(comp_colors)]
            ax.plot(
                wd[mask_model],
                cont - comp[mask_model],
                color=cc,
                lw=0.9,
                ls="-.",
                alpha=0.75,
                zorder=3,
                label=clab,
            )

        ax.plot(
            wd[mask_model],
            md[mask_model],
            color="red",
            lw=1.8,
            ls="--",
            zorder=6,
            label=f"{model_label} fit",
        )

        if np.isfinite(ll):
            ax.axvline(ll, color="#006400", ls="--", lw=1.2, alpha=0.95, zorder=7)
        if np.isfinite(lr):
            ax.axvline(lr, color="#006400", ls="--", lw=1.2, alpha=0.95, zorder=7)

    ax.axvline(
        rest,
        color="#FF1493",
        lw=1.3,
        ls=":",
        zorder=4,
        alpha=0.95,
        label=rf"$\lambda_{{\rm lab}}$ = {rest:.2f} Å",
    )

    center_fit = result.get("center_fit", np.nan)
    if result["success"] and np.isfinite(center_fit) and abs(center_fit - rest) > 0.05:
        ax.axvline(
            center_fit,
            color="#1f77b4",
            lw=0.8,
            ls=":",
            zorder=4,
            alpha=0.6,
            label=rf"$\lambda_{{\rm obs}}$ = {center_fit:.2f} Å",
        )

    yvals = [np.nanmin(fp), np.nanmax(fp), 1.0]
    if result["success"]:
        yvals.extend([cont, np.nanmin(md[mask_model]), np.nanmax(md[mask_model])])
        for _, comp in result["components"]:
            yvals.extend([
                np.nanmin(cont - comp[mask_model]),
                np.nanmax(cont - comp[mask_model]),
            ])

    ylo = min(yvals) - 0.04
    yhi = max(max(yvals) + 0.03, 1.03)

    ax.set_ylim(ylo, yhi)
    ax.set_xlim(plot_window)
    ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]", labelpad=3)
    ax.set_ylabel(r"$\mathrm{I} / \mathrm{I}_{\mathrm{cont}}$")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))

    _nice_limits(ax)

    title = line_name if not obj else f"{obj} — {line_name}"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        short_name = line_info.get("ascii", "line")

        for ext in (".png", ".pdf"):
            fp_out = p.parent / f"{p.stem}_{short_name}_{recommended}{ext}"
            fig.savefig(
                fp_out,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            print(f"  -> {fp_out}")

    plt.close(fig)
    return fig


# ============================================================
# OUTPUT SUMMARY
# ============================================================

def print_balmer_summary(all_results):
    sep = "=" * 92
    print(f"\n{sep}")
    print("  BALMER SERIES SUMMARY")
    print(sep)
    print(f"  {'Line':<8} {'rest':>7} {'Model':<20} {'centre':>8} {'dv':>7} {'depth':>6} {'FWHM':>6} {'EW':>7} {'RMS':>7}")

    def fmt(v, spec):
        return f"{v:{spec}}" if (v is not None and np.isfinite(float(v))) else "  -  "

    for res in all_results:
        if res is None:
            continue

        li = res["line"]
        rec = res["recommended"]

        if rec is None:
            print(f"  {li['ascii']:<8} {li['rest']:>7.2f} {'(no fit)':^20}")
            continue

        mr = res["models"][rec]
        print(
            f"  {li['ascii']:<8} {li['rest']:>7.2f} "
            f"{BALMER_DISPLAY.get(rec, rec):<20} "
            f"{fmt(mr['center_fit'], '8.3f')} "
            f"{fmt(mr['dv'], '+6.1f')} "
            f"{fmt(mr['depth_fit'], '.4f')} "
            f"{fmt(mr['fwhm'], '.3f')} "
            f"{fmt(mr.get('ew_obs', np.nan), '.3f')} "
            f"{fmt(mr.get('rms', np.nan), '.5f')}"
        )

        for cl, ec in mr.get("ew_components", []):
            print(
                f"  {'':8} {'':7} {'  -> ' + cl:<20} "
                f"{'':8} {'':7} {'':6} {'':6} {fmt(ec, '.3f')}"
            )

    print(sep)


def save_balmer_csv(all_results, output_prefix):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    fp_csv = output_prefix.parent / f"{output_prefix.stem}_results.csv"

    fields = [
        "line",
        "rest",
        "model",
        "centre",
        "dv_km_s",
        "depth",
        "fwhm",
        "ew",
        "ew_err",
        "lam_left",
        "lam_right",
        "truncated",
        "rms",
        "aic",
        "bic",
        "sigma_local",
        "snr_depth",
    ]

    rows = []
    for res in all_results:
        if res is None or res["recommended"] is None:
            continue

        li = res["line"]
        rec = res["recommended"]
        mr = res["models"][rec]
        emp = res["empirical"]

        rows.append({
            "line": li["ascii"],
            "rest": li["rest"],
            "model": BALMER_DISPLAY.get(rec, rec),
            "centre": mr["center_fit"],
            "dv_km_s": mr["dv"],
            "depth": mr["depth_fit"],
            "fwhm": mr["fwhm"],
            "ew": mr.get("ew_obs", np.nan),
            "ew_err": mr.get("ew_err", np.nan),
            "lam_left": mr.get("lam_left", np.nan),
            "lam_right": mr.get("lam_right", np.nan),
            "truncated": mr.get("truncated", False),
            "rms": mr.get("rms", np.nan),
            "aic": mr.get("aic", np.nan),
            "bic": mr.get("bic", np.nan),
            "sigma_local": emp.get("sigma_local", np.nan),
            "snr_depth": emp.get("snr_depth", np.nan),
        })

    with open(fp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  CSV -> {fp_csv}")


# ============================================================
# DRIVER
# ============================================================

def run_balmer_block():
    BALMER_DIR.mkdir(parents=True, exist_ok=True)

    wave, flux, hdr, obj_name = load_averaged_fits(AVERAGED_FITS_PATH)

    print(f"\n{'=' * 60}")
    print("  BALMER SERIES ANALYSIS")
    print(f"  Coverage: {wave.min():.1f} - {wave.max():.1f} A")
    print(f"{'=' * 60}")

    all_results = []
    output_prefix = BALMER_DIR / "balmer_mc"

    for line_info in BALMER_LINES:
        res = analyse_balmer_line(
            wave,
            flux,
            line_info,
            plot=True,
            output_path=output_prefix,
            obj=obj_name,
        )
        all_results.append(res)

    print_balmer_summary(all_results)
    save_balmer_csv(all_results, output_prefix)

    print("\nDone.")
    return all_results


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "BALMER_MODELS",
    "BALMER_DISPLAY",
    "BALMER_SHORT",
    "balmer_measure_empirical",
    "balmer_init_bounds",
    "balmer_fit_robust",
    "balmer_components",
    "balmer_model_fwhm",
    "balmer_measure_ew",
    "analyse_balmer_line",
    "print_balmer_summary",
    "save_balmer_csv",
    "run_balmer_block",
]


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run_balmer_block()


# In[ ]:




