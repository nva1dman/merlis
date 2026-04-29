#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# jupyter nbconvert --to script s00_settings.ipynb


# In[23]:


import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.signal import correlate, correlation_lags

from s00_settings import (
    FITS_DIR,
    FILE_PATTERNS,
    STACKED_DIR,
    DISPLAY_RANGE,
    SNR_REGIONS,
    LINE_LABELS,
    ANALYSIS_MODE,
    SAVE_STACKED,
    SAVE_PLOTS,
    SAVE_FITS,
    OUTPUT_TAG,
    APPLY_RV_CHECK,
    APPLY_RV_CORRECTION,
    RV_SHIFT_WARN_KMS,
    USE_SIGMA_CLIP,
    SIGMA_CLIP,
)

from s01_spectrum_utils import (
    read_wavelength_from_header,
    snr_from_continuum_windows,
    measure_snr_table,
    print_snr_table,
)

warnings.filterwarnings("ignore")


# ============================================================
# FITS LOADING
# ============================================================

def read_fits_spectrum(filepath):
    """
    Read one 1D spectrum from FITS.

    Returns
    -------
    dict or None
        {
            "wavelength": wave,
            "flux": flux,
            "header": header,
            "filename": filename,
            "object_name": object_name,
        }
    """
    filepath = Path(filepath)

    try:
        with fits.open(filepath) as hdul:
            data = None
            header = None

            for hdu in hdul:
                if hdu.data is None:
                    continue

                raw = np.asarray(hdu.data)
                raw = np.squeeze(raw)

                if raw.ndim == 1:
                    data = raw.astype(np.float64)
                    header = hdu.header
                    break
                elif raw.ndim == 0:
                    continue
                else:
                    print(
                        f"  Warning: unsupported data shape {raw.shape} "
                        f"in {filepath.name}"
                    )

            if data is None:
                print(f"  Warning: no usable 1D spectrum in {filepath.name}")
                return None

            wave = read_wavelength_from_header(header, len(data))
            if wave is None:
                print(f"  Warning: no WCS in {filepath.name}")
                return None

            ok = np.isfinite(wave) & np.isfinite(data)
            wave = wave[ok]
            data = data[ok]

            if len(wave) < 100:
                print(f"  Warning: too few valid points in {filepath.name}")
                return None

            if not np.all(np.diff(wave) > 0):
                sort_idx = np.argsort(wave)
                wave = wave[sort_idx]
                data = data[sort_idx]

                uniq = np.concatenate(([True], np.diff(wave) > 0))
                wave = wave[uniq]
                data = data[uniq]

                print(f"  Warning: wavelength axis re-sorted in {filepath.name}")

            object_name = header.get("OBJECT", None)

            return {
                "wavelength": wave,
                "flux": data,
                "header": header,
                "filename": filepath.name,
                "object_name": object_name,
            }

    except Exception as exc:
        print(f"  Error reading {filepath.name}: {exc}")
        return None


def load_all_spectra(data_dir, file_patterns):
    """
    Load all spectra from FITS directory using configured patterns.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_path}")

    files = sorted({f for pat in file_patterns for f in data_path.glob(pat)})

    print(f"Found {len(files)} FITS files")

    spectra = []
    for f in files:
        spec = read_fits_spectrum(f)
        if spec is not None:
            spectra.append(spec)

    print(f"Loaded {len(spectra)} spectra")
    return spectra


# ============================================================
# RV CROSS-CORRELATION
# ============================================================

def _cc_velocity(wave_ref, flux_ref, wave_obs, flux_obs, max_kms=300.0):
    """
    Estimate relative RV shift using cross-correlation in log-lambda space.
    """
    c_kms = 299792.458
    cc_range = (4400.0, 6000.0)

    wl0 = max(wave_ref.min(), wave_obs.min(), cc_range[0])
    wl1 = min(wave_ref.max(), wave_obs.max(), cc_range[1])

    if wl1 <= wl0:
        return 0.0, 0.0

    def _to_log_grid(w, f):
        m = (w >= wl0) & (w <= wl1)
        lw = np.log10(w[m])
        ff = f[m]

        if len(lw) < 50:
            return None, None, None

        lgrid = np.linspace(lw.min(), lw.max(), len(lw))
        dlg = lgrid[1] - lgrid[0]

        fi = interp1d(
            lw, ff,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        return lgrid, fi(lgrid), dlg

    lg_r, fr, dlg = _to_log_grid(wave_ref, flux_ref)
    lg_o, fo, _ = _to_log_grid(wave_obs, flux_obs)

    if lg_r is None or lg_o is None:
        return 0.0, 0.0

    fi2 = interp1d(
        lg_o, fo,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    fo_c = fi2(lg_r)

    ok = np.isfinite(fr) & np.isfinite(fo_c)
    if ok.sum() < 50:
        return 0.0, 0.0

    def _norm(x):
        x = x - np.mean(x)
        s = np.std(x)
        return x / s if s > 0 else x

    cc = correlate(_norm(fr[ok]), _norm(fo_c[ok]), mode="full")
    lags = correlation_lags(ok.sum(), ok.sum(), mode="full")

    max_lag = int(max_kms / (c_kms * np.log(10) * dlg)) + 1
    center = len(lags) // 2
    sl = slice(
        max(0, center - max_lag),
        min(len(lags), center + max_lag + 1),
    )

    if len(cc[sl]) == 0:
        return 0.0, 0.0

    pk = np.argmax(cc[sl])
    velocity = c_kms * np.log(10) * lags[sl][pk] * dlg
    quality = cc[sl][pk] / ok.sum()

    return float(velocity), float(quality)


def check_rv_shifts(spectra, warn_kms=5.0):
    """
    Cross-correlate all spectra against the first spectrum in the list.
    """
    ref = spectra[0]
    results = []

    print(f"\n  RV shift check  (reference: {ref['filename']})")
    print(f"  {'File':<38s}  {'Δv (km/s)':>10}  {'CC quality':>10}")
    print(f"  {'────':<38s}  {'─────────':>10}  {'──────────':>10}")

    any_large = False

    for spec in spectra:
        if spec is ref:
            print(f"  {spec['filename']:<38s}  {'0.00':>10}  {'(reference)':>10}")
            results.append((spec["filename"], 0.0, 1.0))
            continue

        v, q = _cc_velocity(
            ref["wavelength"], ref["flux"],
            spec["wavelength"], spec["flux"],
        )

        flag = "  ← WARNING" if abs(v) > warn_kms else ""
        print(f"  {spec['filename']:<38s}  {v:>+10.2f}  {q:>10.3f}{flag}")

        results.append((spec["filename"], v, q))

        if abs(v) > warn_kms:
            any_large = True

    if any_large:
        print(f"\n  *** WARNING: shift > {warn_kms:.1f} km/s detected. ***")
    else:
        print(f"\n  All shifts within ±{warn_kms:.1f} km/s.")

    max_shift = max(abs(r[1]) for r in results) if results else 0.0
    print(f"  Max |Δv| = {max_shift:.2f} km/s")

    return results


# ============================================================
# STACKING
# ============================================================

def _sigma_clip_stack(all_flux, n_sigma=3.0):
    """
    Pixel-wise sigma clipping on a stack with shape (N_spec, N_pix).
    Replaces outliers with NaN.
    """
    n_frames = np.sum(np.any(np.isfinite(all_flux), axis=1))
    if n_frames < 4:
        print(
            f"  Sigma clipping skipped: only {n_frames} valid frames "
            f"(need ≥ 4 for reliable outlier estimation)."
        )
        return all_flux

    median_pix = np.nanmedian(all_flux, axis=0)
    mad_pix = np.nanmedian(np.abs(all_flux - median_pix[None, :]), axis=0)
    sigma_pix = 1.4826 * mad_pix

    zero_sigma = sigma_pix <= 0
    sigma_pix = np.where(zero_sigma, np.nan, sigma_pix)

    clipped = all_flux.copy()
    bad = np.abs(all_flux - median_pix[None, :]) > n_sigma * sigma_pix[None, :]
    bad[:, zero_sigma] = False
    clipped[bad] = np.nan

    n_clipped = int(np.sum(bad))
    if n_clipped > 0:
        frac = 100.0 * n_clipped / all_flux.size
        print(f"  Sigma clipping ({n_sigma}σ): {n_clipped} pixels masked ({frac:.2f} % of stack)")
    else:
        print(f"  Sigma clipping ({n_sigma}σ): no outliers found.")

    return clipped


def average_spectra_weighted(spectra, snr_regions, wavelength_range=None, sigma_clip=None):
    """
    Build SNR^2-weighted average on a common linear wavelength grid.
    """
    wl_min = max(s["wavelength"].min() for s in spectra)
    wl_max = min(s["wavelength"].max() for s in spectra)

    if wavelength_range is not None:
        wl_min = max(wl_min, wavelength_range[0])
        wl_max = min(wl_max, wavelength_range[1])

    if wl_max <= wl_min:
        raise ValueError("Invalid common wavelength range for averaging.")

    dw = min(np.median(np.abs(np.diff(s["wavelength"]))) for s in spectra)
    grid = np.arange(wl_min, wl_max, dw)

    all_flux = []
    snr_values = []

    for s in spectra:
        snr_c = snr_from_continuum_windows(
            s["wavelength"], s["flux"], snr_regions
        )
        snr_values.append(snr_c)

        fi = interp1d(
            s["wavelength"], s["flux"],
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        all_flux.append(fi(grid))

    all_flux = np.array(all_flux)
    snr_arr = np.array(snr_values, dtype=float)

    bad_w = ~np.isfinite(snr_arr) | (snr_arr <= 0)
    if bad_w.any():
        print(
            f"  Warning: {bad_w.sum()} frame(s) with invalid SNR — "
            f"excluded from averaging."
        )
        all_flux[bad_w] = np.nan
        snr_arr[bad_w] = 0.0

    weights = snr_arr ** 2

    if sigma_clip is not None and sigma_clip > 0 and (~bad_w).sum() > 2:
        all_flux = _sigma_clip_stack(all_flux, n_sigma=sigma_clip)

    wsum = np.nansum(all_flux * weights[:, None], axis=0)
    wnorm = np.nansum(weights[:, None] * np.isfinite(all_flux), axis=0)

    avg_flux = np.where(wnorm > 0, wsum / wnorm, np.nan)

    w2d = weights[:, None] * np.isfinite(all_flux)
    residual = (all_flux - avg_flux[None, :]) ** 2
    wvar = np.nansum(w2d * residual, axis=0) / np.where(wnorm > 0, wnorm, 1.0)
    flux_scatter = np.sqrt(wvar)

    n_spectra = np.sum(np.isfinite(all_flux), axis=0).astype(int)

    return {
        "wavelength": grid,
        "flux": avg_flux,
        "flux_scatter": flux_scatter,
        "n_spectra": n_spectra,
        "snr_values": snr_values,
        "all_flux": all_flux,
    }


# ============================================================
# PLOTTING
# ============================================================

def _pub_style():
    rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "legend.frameon": False,
    })


def _find_line_core(wave, flux, wl_lab, search_half_window=2.0):
    """
    Find the observed local minimum near a laboratory wavelength.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    m = (
        np.isfinite(wave)
        & np.isfinite(flux)
        & (wave >= wl_lab - search_half_window)
        & (wave <= wl_lab + search_half_window)
    )

    if np.sum(m) < 3:
        m = (
            np.isfinite(wave)
            & np.isfinite(flux)
            & (wave >= wl_lab - 5.0)
            & (wave <= wl_lab + 5.0)
        )

    if np.sum(m) < 3:
        return float(wl_lab), 1.0

    ww = wave[m]
    ff = flux[m]

    imin = np.argmin(ff)

    return float(ww[imin]), float(ff[imin])


def _draw_line_labels(ax, wave, flux, line_dict, wavelength_range,
                      fontsize=6.2, color="#B00020",
                      search_half_window=2.0):
    """
    Draw top atlas-style labels.

    Uses only line_dict passed from s00_settings.LINE_LABELS.
    LINE_LABELS must be built from input.dat [LINES].
    """
    if not line_dict:
        print("  No LINE_LABELS provided; no line labels drawn.")
        return

    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if wavelength_range is None:
        wl0 = np.nanmin(wave)
        wl1 = np.nanmax(wave)
    else:
        wl0, wl1 = wavelength_range

    visible = []

    for name, wl in line_dict.items():
        try:
            wl = float(wl)
        except Exception:
            continue

        if wl0 <= wl <= wl1:
            x_core, y_core = _find_line_core(
                wave=wave,
                flux=flux,
                wl_lab=wl,
                search_half_window=search_half_window,
            )

            visible.append({
                "name": str(name),
                "wl_lab": wl,
                "x_core": x_core,
                "y_core": y_core,
            })

    visible = sorted(visible, key=lambda x: x["wl_lab"])

    if not visible:
        print("  No input.dat lines fall inside DISPLAY_RANGE; no line labels drawn.")
        return

    print("\n  Lines drawn on overview plot:")
    for item in visible:
        print(f"    {item['name']:<18s}  lab={item['wl_lab']:.2f}  core={item['x_core']:.2f}")

    # --------------------------------------------------------
    # Uniform label placement: all labels on one height
    # --------------------------------------------------------
    positioned = []

    for item in visible:
        item["x_label"] = item["x_core"]
        positioned.append(item)

    trans = ax.get_xaxis_transform()

    tick_bottom_y = 1.005
    tick_top_y = 1.070
    y_label_axes = 1.110

    for item in positioned:
        name = item["name"]
        x_core = item["x_core"]
        y_core = item["y_core"]
        x_label = item["x_label"]

        # Red dashed guide line from line core to label
        ax.annotate(
            "",
            xy=(x_label, tick_bottom_y),
            xycoords=trans,
            xytext=(x_core, y_core),
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                lw=0.55,
                linestyle=(0, (2, 3)),
                shrinkA=0,
                shrinkB=0,
            ),
            clip_on=False,
            zorder=10,
        )

        # Red point at the detected local minimum
        ax.plot(
            x_core,
            y_core,
            marker="o",
            markersize=3.2,
            markeredgewidth=0,
            color=color,
            clip_on=True,
            zorder=11,
        )

        # Small vertical red tick near the top axis
        ax.plot(
            [x_label, x_label],
            [tick_bottom_y, tick_top_y],
            transform=trans,
            color=color,
            linewidth=0.65,
            linestyle="-",
            clip_on=False,
            zorder=10,
        )

        # Vertical label above the spectrum
        ax.text(
            x_label,
            y_label_axes,
            name,
            transform=trans,
            fontsize=fontsize,
            ha="center",
            va="bottom",
            rotation=90,
            color=color,
            clip_on=False,
            zorder=12,
        )


def plot_publication(main_spec, spectra_raw, is_averaged,
                     snr_regions=None, output_path=None,
                     wavelength_range=None, object_name=None,
                     line_labels=None):
    _pub_style()

    color_indiv = "#C8C8C8"
    color_band = "#2E5FA3"
    color_main = "#1A3A6B"
    color_cont = "#BBBBBB"
    color_annot = "#1A1A1A"

    fig, ax = plt.subplots(figsize=(18.0 / 2.54, 8.5 / 2.54))

    wave = main_spec["wavelength"].copy()
    flux = main_spec["flux"].copy()
    flux_std = main_spec.get("flux_scatter")

    if wavelength_range is not None:
        m = (wave >= wavelength_range[0]) & (wave <= wavelength_range[1])
        wave = wave[m]
        flux = flux[m]

        if flux_std is not None:
            flux_std = flux_std[m]

    # Individual spectra in the background.
    if is_averaged and spectra_raw:
        for spec in spectra_raw:
            sw = spec["wavelength"].copy()
            sf = spec["flux"].copy()

            if wavelength_range is not None:
                mm = (sw >= wavelength_range[0]) & (sw <= wavelength_range[1])
                sw = sw[mm]
                sf = sf[mm]

            ax.plot(
                sw,
                sf,
                color=color_indiv,
                alpha=0.30,
                linewidth=0.15,
                zorder=1,
                rasterized=True,
            )

    # SNR continuum windows.
    if snr_regions:
        wl0_plot = wavelength_range[0] if wavelength_range else -np.inf
        wl1_plot = wavelength_range[1] if wavelength_range else np.inf

        for _, (_, rw0, rw1) in snr_regions.items():
            if rw1 < wl0_plot or rw0 > wl1_plot:
                continue

            for xv in (rw0, rw1):
                ax.axvline(
                    xv,
                    color="#AAAAAA",
                    linewidth=0.4,
                    linestyle="--",
                    alpha=0.6,
                    zorder=0,
                )

    # Scatter band.
    if flux_std is not None:
        ax.fill_between(
            wave,
            flux - flux_std,
            flux + flux_std,
            color=color_band,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )

    # Main averaged spectrum.
    ax.plot(
        wave,
        flux,
        color=color_main,
        linewidth=0.9,
        zorder=4,
    )

    # Continuum level.
    ax.axhline(
        1.0,
        color=color_cont,
        linestyle="--",
        linewidth=0.4,
        alpha=0.8,
        zorder=0,
    )

    # --------------------------------------------------------
    # y-limits
    # --------------------------------------------------------
    balmer_anchors = {
        r"H$\alpha$": 6562.80,
        r"H$\beta$": 4861.33,
        r"H$\gamma$": 4340.47,
    }

    core_mins = []

    for _, wl_c in balmer_anchors.items():
        m = (wave >= wl_c - 15.0) & (wave <= wl_c + 15.0)

        if np.sum(m) > 3:
            core_mins.append(np.nanmin(flux[m]))

    if core_mins:
        y_floor = min(core_mins)
        y_bot = y_floor - 0.04
    else:
        y_bot = np.nanpercentile(flux, 0.5)

    y_top = np.nanpercentile(flux, 99.8)

    if not np.isfinite(y_top):
        y_top = np.nanmax(flux)

    if not np.isfinite(y_bot):
        y_bot = np.nanmin(flux)

    dy = y_top - y_bot

    if not np.isfinite(dy) or dy <= 0:
        y_bot = 0.0
        y_top = 1.1
        dy = y_top - y_bot

    ax.set_ylim(
        y_bot - 0.01 * dy,
        y_top + 0.05 * dy,
    )

    if wavelength_range is not None:
        ax.set_xlim(wavelength_range)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=5,
        width=0.7,
        top=True,
        right=True,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=2.5,
        width=0.5,
        top=True,
        right=True,
    )

    ax.set_xlabel(
        r"Wavelength ($\mathrm{\AA}$)",
        labelpad=3,
    )

    ax.set_ylabel(
        r"$\mathrm{I} / \mathrm{I}_{\mathrm{cont}}$",
        labelpad=3,
    )

    # --------------------------------------------------------
    # Draw only lines passed from input.dat through LINE_LABELS.
    # --------------------------------------------------------
    print("\n  LINE_LABELS received by plot_publication:")
    if line_labels:
        for k, v in line_labels.items():
            print(f"    {k:<18s} {float(v):10.2f}")

        _draw_line_labels(
            ax=ax,
            wave=wave,
            flux=flux,
            line_dict=line_labels,
            wavelength_range=wavelength_range,
            fontsize=5.0,
            color="#B00020",
            search_half_window=2.0,
        )
    else:
        print("    None")

    # Object name only. Do not print N.
    if object_name:
        ax.text(
            0.012,
            0.06,
            object_name,
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            style="italic",
            color=color_annot,
            zorder=5,
        )

    # More top space for vertical labels.
    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.15,
        top=0.70,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        for ext in (".pdf", ".png"):
            fig.savefig(
                output_path.with_suffix(ext),
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )

            print(f"  Figure → {output_path.with_suffix(ext)}")

    plt.close(fig)

    return fig, ax
    

# ============================================================
# FITS SAVING
# ============================================================

def save_fits(wavelength, flux, output_path,
              ref_header=None, flux_std=None, n_spectra=None, history=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hdr = fits.Header()

    if ref_header is not None:
        for key in ("OBJECT", "TELESCOP", "INSTRUME", "DATE-OBS",
                    "EXPTIME", "OBSERVER", "BUNIT"):
            if key in ref_header:
                hdr[key] = ref_header[key]

    hdr["CTYPE1"] = "LINEAR"
    hdr["CRPIX1"] = 1.0
    hdr["CRVAL1"] = float(wavelength[0])
    hdr["CDELT1"] = float(np.median(np.diff(wavelength)))
    hdr["CD1_1"] = hdr["CDELT1"]
    hdr["WCSDIM"] = 1
    hdr["DC-FLAG"] = 0
    hdr["DISPAXIS"] = 1
    hdr["CUNIT1"] = "Angstrom"
    hdr["WSTART"] = float(wavelength[0])
    hdr["WEND"] = float(wavelength[-1])
    hdr["WDELTA"] = hdr["CDELT1"]

    if history:
        for h in history:
            hdr["HISTORY"] = h

    fits.PrimaryHDU(flux.astype(np.float32), hdr).writeto(output_path, overwrite=True)
    print(f"  FITS → {output_path}")

    if flux_std is not None:
        p = output_path.with_name(output_path.stem + "_scatter.fits")
        h2 = hdr.copy()
        h2["BUNIT"] = "FLUX_SCATTER"
        h2["HISTORY"] = "Weighted stddev of input stack (frame-to-frame scatter)"
        h2["HISTORY"] = "NOT the error of the mean"
        fits.PrimaryHDU(flux_std.astype(np.float32), h2).writeto(p, overwrite=True)
        print(f"  scatter → {p.name}")

    if n_spectra is not None:
        p = output_path.with_name(output_path.stem + "_nspec.fits")
        h3 = hdr.copy()
        h3["BUNIT"] = "N_SPECTRA"
        fits.PrimaryHDU(n_spectra.astype(np.int16), h3).writeto(p, overwrite=True)
        print(f"  Nsp → {p.name}")

    return output_path


# ============================================================
# ANALYSIS TARGET BUILDING
# ============================================================

def make_individual_targets(spectra):
    targets = []

    for spec in spectra:
        targets.append({
            "spectrum_id": Path(spec["filename"]).stem,
            "spectrum_kind": "individual",
            "filename": spec["filename"],
            "object_name": spec.get("object_name"),
            "header": spec["header"],
            "wavelength": spec["wavelength"],
            "flux": spec["flux"],
        })

    return targets


def make_averaged_target(avg, spectra):
    ref_header = spectra[0]["header"] if spectra else None
    object_name = spectra[0].get("object_name") if spectra else None

    return {
        "spectrum_id": "average",
        "spectrum_kind": "averaged",
        "filename": "averaged_spectrum.fits",
        "object_name": object_name,
        "header": ref_header,
        "wavelength": avg["wavelength"],
        "flux": avg["flux"],
        "flux_scatter": avg.get("flux_scatter"),
        "n_spectra": avg.get("n_spectra"),
    }


def build_analysis_targets(spectra, averaged_result=None, analysis_mode="AVERAGED"):
    mode = str(analysis_mode).strip().upper()
    targets = []

    if mode in {"AVERAGED", "BOTH"}:
        if averaged_result is None:
            raise ValueError("AVERAGED/BOTH mode requires averaged_result.")
        targets.append(make_averaged_target(averaged_result, spectra))

    if mode in {"INDIVIDUAL", "BOTH"}:
        targets.extend(make_individual_targets(spectra))

    if not targets:
        raise ValueError(f"Unsupported ANALYSIS_MODE: {analysis_mode}")

    return targets


# ============================================================
# DRIVER
# ============================================================

def run_prepare_spectra():
    """
    End-to-end preparation block:
    - load FITS
    - estimate per-frame continuum SNR
    - optional RV check
    - optional weighted average
    - optional stacked outputs
    - build analysis targets according to ANALYSIS_MODE
    """
    STACKED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PREPARE SPECTRA")
    print(f"  ANALYSIS_MODE     : {ANALYSIS_MODE}")
    print(f"  SAVE_STACKED      : {SAVE_STACKED}")
    print(f"  SAVE_PLOTS        : {SAVE_PLOTS}")
    print(f"  SAVE_FITS         : {SAVE_FITS}")
    print(f"  OUTPUT_TAG        : {OUTPUT_TAG}")
    print(f"  STACKED_DIR       : {STACKED_DIR}")
    print(f"  APPLY_RV_CHECK    : {APPLY_RV_CHECK}")
    print(f"  APPLY_RV_CORR     : {APPLY_RV_CORRECTION}")
    print(f"  USE_SIGMA_CLIP    : {USE_SIGMA_CLIP}")
    print(f"  DISPLAY_RANGE     : {DISPLAY_RANGE[0]}–{DISPLAY_RANGE[1]} Å")
    print("  LINE_LABELS       :")

    for label, wl in LINE_LABELS.items():
        print(f"    {label:<18s} {float(wl):10.2f}")

    print("=" * 60)

    print("\n[1] Loading spectra …")
    spectra = load_all_spectra(FITS_DIR, FILE_PATTERNS)

    if not spectra:
        raise ValueError("No valid spectra found.")

    snr_pre = np.array([
        snr_from_continuum_windows(s["wavelength"], s["flux"], SNR_REGIONS)
        for s in spectra
    ], dtype=float)

    snr_sort = np.where(np.isfinite(snr_pre), snr_pre, -np.inf)
    order = np.argsort(snr_sort)[::-1]

    spectra = [spectra[i] for i in order]
    snr_pre = [snr_pre[i] for i in order]

    print("\n  Per-frame continuum SNR:")
    for idx, (s, sv) in enumerate(zip(spectra, snr_pre)):
        tag = "  ← reference" if idx == 0 else ""
        sv_str = f"{sv:.1f}" if np.isfinite(sv) else "—"
        print(f"    {s['filename']:<38s}  SNR = {sv_str}{tag}")

    rv_results = []

    if APPLY_RV_CHECK and len(spectra) > 1:
        print("\n[2] Checking mutual RV shifts …")
        rv_results = check_rv_shifts(spectra, warn_kms=RV_SHIFT_WARN_KMS)
    else:
        print("\n[2] RV check skipped.")

    if APPLY_RV_CORRECTION:
        raise NotImplementedError("APPLY_RV_CORRECTION=YES is not implemented yet.")

    need_average = (ANALYSIS_MODE in {"AVERAGED", "BOTH"}) or SAVE_STACKED
    avg = None
    is_averaged = False

    if need_average:
        if len(spectra) == 1:
            print("\n[3] Single spectrum: averaged product = original spectrum.")

            avg = {
                "wavelength": spectra[0]["wavelength"].copy(),
                "flux": spectra[0]["flux"].copy(),
                "flux_scatter": np.full_like(spectra[0]["flux"], np.nan, dtype=float),
                "n_spectra": np.ones_like(spectra[0]["flux"], dtype=int),
                "snr_values": [snr_pre[0]],
                "all_flux": np.array([spectra[0]["flux"].copy()]),
            }

            is_averaged = True

        else:
            print("\n[3] Computing SNR²-weighted average …")

            sigma_clip_value = SIGMA_CLIP if USE_SIGMA_CLIP else None

            avg = average_spectra_weighted(
                spectra=spectra,
                snr_regions=SNR_REGIONS,
                wavelength_range=DISPLAY_RANGE,
                sigma_clip=sigma_clip_value,
            )

            snr_after = measure_snr_table(
                avg["wavelength"],
                avg["flux"],
                SNR_REGIONS,
            )

            median_snr_after = np.nanmedian(list(snr_after.values()))

            print(f"\n  Frames used      : {len(spectra)}")
            print(f"  Median SNR after : {median_snr_after:.1f}")

            is_averaged = True

    else:
        print("\n[3] Averaging not requested by current settings.")

    print("\n[4] Measuring SNR in continuum windows …")

    if avg is not None:
        tbl = measure_snr_table(avg["wavelength"], avg["flux"], SNR_REGIONS)
        print_snr_table(tbl, SNR_REGIONS, title="averaged spectrum")

    for spec in spectra:
        tbl = measure_snr_table(spec["wavelength"], spec["flux"], SNR_REGIONS)
        print_snr_table(tbl, SNR_REGIONS, title=spec["filename"])

    fig = None
    ax = None

    if avg is not None and SAVE_PLOTS:
        print("\n[5] Building overview figure …")

        main_spec = avg
        object_name = spectra[0]["header"].get("OBJECT", None)

        fig_stem = (
            f"averaged_spectrum_{OUTPUT_TAG}"
            if OUTPUT_TAG
            else "averaged_spectrum"
        )

        fig, ax = plot_publication(
            main_spec=main_spec,
            spectra_raw=spectra,
            is_averaged=is_averaged,
            snr_regions=SNR_REGIONS,
            output_path=STACKED_DIR / fig_stem,
            wavelength_range=DISPLAY_RANGE,
            object_name=object_name,
            line_labels=LINE_LABELS,
        )

    if avg is not None and SAVE_STACKED and SAVE_FITS:
        print("\n[6] Saving stacked FITS products …")

        max_rv = max(abs(r[1]) for r in rv_results) if rv_results else 0.0

        hist = [
            f"N frames averaged   : {len(spectra)}",
            "Method              : SNR^2-weighted mean",
            "Weights             : median continuum-window SNR",
            f"Sigma clipping      : {'OFF' if not USE_SIGMA_CLIP else f'{SIGMA_CLIP} sigma pixel-wise'}",
            "RV alignment        : NOT applied",
            f"Max RV shift found  : {max_rv:.2f} km/s (diagnostic only)",
            "flux_scatter        : weighted stddev of input stack",
        ] + [
            f"{s['filename']}  cont-SNR={'NA' if not np.isfinite(sv) else f'{sv:.1f}'}"
            for s, sv in zip(spectra, avg["snr_values"])
        ]

        fits_name = (
            f"averaged_spectrum_{OUTPUT_TAG}.fits"
            if OUTPUT_TAG
            else "averaged_spectrum.fits"
        )

        save_fits(
            wavelength=avg["wavelength"],
            flux=avg["flux"],
            output_path=STACKED_DIR / fits_name,
            ref_header=spectra[0]["header"],
            flux_std=avg["flux_scatter"],
            n_spectra=avg["n_spectra"],
            history=hist,
        )

    analysis_targets = build_analysis_targets(
        spectra=spectra,
        averaged_result=avg,
        analysis_mode=ANALYSIS_MODE,
    )

    print("\n[7] Analysis targets …")

    for t in analysis_targets:
        print(f"  {t['spectrum_kind']:<10s}  {t['spectrum_id']}")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)

    return {
        "spectra": spectra,
        "averaged": avg,
        "rv_results": rv_results,
        "analysis_targets": analysis_targets,
        "figure": (fig, ax),
    }

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run_prepare_spectra()


# In[ ]:




