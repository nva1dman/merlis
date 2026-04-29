#!/usr/bin/env python
# coding: utf-8

# In[13]:


from pathlib import Path
import numpy as np
from astropy.io import fits


# ============================================================
# BASIC NUMERICAL HELPER
# ============================================================

def trapz_integral(y, x):
    """Safe trapezoidal integral for old/new NumPy versions."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


# ============================================================
# FITS / WAVELENGTH HELPERS
# ============================================================

def read_wavelength_from_header(header, npts):
    """
    Reconstruct wavelength array from FITS header.

    Supports:
    - CRVAL1
    - CDELT1 or CD1_1
    - CRPIX1
    - optional logarithmic axis if CTYPE1 contains 'LOG'
    """
    crval = header.get("CRVAL1")
    cdelt = header.get("CDELT1")
    if cdelt is None:
        cdelt = header.get("CD1_1")
    crpix = header.get("CRPIX1", 1.0)

    if crval is None or cdelt is None:
        return None

    pix = np.arange(npts, dtype=float) + 1.0
    wave = crval + (pix - crpix) * cdelt

    ctype1 = str(header.get("CTYPE1", "")).upper()
    if "LOG" in ctype1:
        wave = 10.0 ** wave

    return wave


def load_averaged_fits(path):
    """
    Load 1D averaged spectrum from FITS and reconstruct wavelength axis.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FITS not found: {path}")

    with fits.open(path) as hdul:
        hdr = hdul[0].header
        flux = np.asarray(hdul[0].data, dtype=np.float64).ravel()

    npts = len(flux)
    wave = read_wavelength_from_header(hdr, npts)
    if wave is None:
        raise ValueError(f"No wavelength calibration found in header: {path}")

    ok = np.isfinite(wave) & np.isfinite(flux)
    wave = wave[ok]
    flux = flux[ok]

    if len(wave) < 10:
        raise ValueError(f"Too few valid data points in: {path}")

    if not np.all(np.diff(wave) > 0):
        sort_idx = np.argsort(wave)
        wave = wave[sort_idx]
        flux = flux[sort_idx]

        uniq = np.concatenate(([True], np.diff(wave) > 0))
        wave = wave[uniq]
        flux = flux[uniq]

    obj_name = hdr.get("OBJECT", None)
    return wave, flux, hdr, obj_name


# ============================================================
# SNR HELPERS
# ============================================================

def der_snr(flux):
    """
    DER_SNR estimator (Stoehr et al. 2008).

    Robust local SNR estimate based on the second derivative.
    """
    flux = np.asarray(flux, dtype=float)
    n = len(flux)

    if n < 5:
        return np.nan

    signal = np.median(flux)
    noise = 1.482602 / np.sqrt(6.0) * np.median(
        np.abs(2.0 * flux[2:n-2] - flux[0:n-4] - flux[4:n])
    )

    if not np.isfinite(noise) or noise <= 0:
        return np.nan

    return signal / noise


def snr_from_continuum_windows(wavelength, flux, regions):
    """
    Compute one robust SNR estimate as the median DER_SNR over valid
    continuum windows.
    """
    values = []

    for _, (_, wl0, wl1) in regions.items():
        mask = (wavelength >= wl0) & (wavelength <= wl1)
        if mask.sum() >= 10:
            s = der_snr(flux[mask])
            if np.isfinite(s):
                values.append(s)

    if not values:
        return np.nan

    return float(np.median(values))


def measure_snr_table(wavelength, flux, regions):
    """
    Measure DER_SNR separately in each continuum window.
    Returns dict {window_name: snr}.
    """
    out = {}

    for name, (_, wl0, wl1) in regions.items():
        mask = (wavelength >= wl0) & (wavelength <= wl1)
        if mask.sum() >= 10:
            out[name] = der_snr(flux[mask])
        else:
            out[name] = np.nan

    return out


def print_snr_table(snr_dict, snr_regions, title="SNR per spectral region"):
    """
    Pretty-print SNR measurements for visual QC.
    """
    width = 16

    print(f"\n{'─' * 58}")
    print(f"  {title}")
    print(f"  Method: DER_SNR (Stoehr et al. 2008)")
    print(f"{'─' * 58}")
    print(f"  {'Window':>{width}}   center    width    SNR")
    print(f"  {'──────':>{width}}   ──────    ─────    ───")

    for name, val in snr_dict.items():
        if name not in snr_regions:
            continue

        cen, w0, w1 = snr_regions[name]
        span = w1 - w0
        val_str = f"{val:6.1f}" if np.isfinite(val) else "    —"
        print(f"  {name:>{width}}   {cen:6.0f} Å   {span:3.0f} Å   {val_str}")

    print(f"{'─' * 58}")


# ============================================================
# LOCAL CONTINUUM / NORMALISATION
# ============================================================

def local_continuum(wave, flux, cont_left, cont_right):
    """
    Estimate local linear continuum from left/right continuum windows.

    Returns
    -------
    level : float
        Continuum value at reference wavelength.
    slope : float
        Linear slope dF/dλ.
    ref_wave : float
        Reference wavelength where 'level' is defined.
    """
    ml = (wave >= cont_left[0]) & (wave <= cont_left[1])
    mr = (wave >= cont_right[0]) & (wave <= cont_right[1])

    vl = flux[ml]
    vr = flux[mr]

    if len(vl) < 3 or len(vr) < 3:
        ref_wave = float(np.nanmedian(wave))
        level = float(np.nanmedian(flux))
        slope = 0.0
        return level, slope, ref_wave

    fl = np.nanmedian(vl)
    fr = np.nanmedian(vr)
    wl = np.nanmedian(wave[ml])
    wr = np.nanmedian(wave[mr])

    slope = (fr - fl) / (wr - wl)
    ref_wave = 0.5 * (wl + wr)
    level = fl + slope * (ref_wave - wl)

    return float(level), float(slope), float(ref_wave)


def make_continuum(wave, level, slope, ref_wave):
    """
    Build linear continuum array.
    """
    return level + slope * (wave - ref_wave)


def normalise_flux(wave, flux, level, slope, ref_wave):
    """
    Divide spectrum by local linear continuum.
    """
    cont = make_continuum(wave, level, slope, ref_wave)
    cont = np.where(cont > 0, cont, 1.0)
    return flux / cont


def local_sigma(wave, flux_norm, cont_left, cont_right):
    """
    Estimate local noise sigma from the normalised continuum windows.
    """
    ml = (wave >= cont_left[0]) & (wave <= cont_left[1])
    mr = (wave >= cont_right[0]) & (wave <= cont_right[1])

    vals = np.concatenate([flux_norm[ml], flux_norm[mr]])
    if len(vals) < 5:
        return np.nan

    return float(np.std(vals - np.median(vals)))


# ============================================================
# LINE GEOMETRY / METRICS
# ============================================================

def find_line_boundaries(wave_dense, model_dense, continuum, sigma_noise,
                         k_sigma, frac_depth):
    """
    Find left/right line boundaries where model absorption falls below
    threshold = max(k_sigma * sigma_noise, frac_depth * peak_depth).
    """
    absorption = continuum - model_dense
    peak = float(np.nanmax(absorption))
    thr = max(k_sigma * sigma_noise, frac_depth * peak)

    im = int(np.argmin(model_dense))

    il = None
    for i in range(im, -1, -1):
        if absorption[i] < thr:
            il = i
            break

    ir = None
    for i in range(im, len(wave_dense)):
        if absorption[i] < thr:
            ir = i
            break

    left_truncated = il is None
    right_truncated = ir is None

    if il is None:
        il = 0
    if ir is None:
        ir = len(wave_dense) - 1

    return {
        "lam_left": float(wave_dense[il]),
        "lam_right": float(wave_dense[ir]),
        "idx_left": il,
        "idx_right": ir,
        "left_truncated": left_truncated,
        "right_truncated": right_truncated,
        "truncated": left_truncated or right_truncated,
        "threshold": float(thr),
    }


def ew_from_observed(wf, flux_norm, lam_left, lam_right, continuum=1.0):
    """
    Equivalent width from observed normalised spectrum.
    """
    mask = (wf >= lam_left) & (wf <= lam_right)
    if mask.sum() < 2:
        return np.nan

    return float(trapz_integral(1.0 - flux_norm[mask] / continuum, wf[mask]))


def ew_err_analytic(sigma, wf, lam_left, lam_right):
    """
    Simple analytic EW uncertainty estimate.
    """
    mask = (wf >= lam_left) & (wf <= lam_right)
    w = wf[mask]

    if len(w) < 2 or not np.isfinite(sigma):
        return np.nan

    dl = float(np.median(np.diff(w)))
    return float(sigma * dl * np.sqrt(2.0 * len(w)))


def calc_aic_bic(residuals, k, n):
    """
    Compute reduced chi-square proxy, AIC, BIC, RMS.
    """
    rss = float(np.sum(np.asarray(residuals, dtype=float) ** 2))

    if rss <= 0 or n <= k:
        return np.nan, np.nan, np.nan, np.nan

    redchi = rss / (n - k)
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    rms = np.sqrt(rss / n)

    return float(redchi), float(aic), float(bic), float(rms)


def model_fwhm_generic(wave_dense, model_dense, continuum=1.0):
    """
    Measure FWHM from dense model profile.
    """
    depth = continuum - np.nanmin(model_dense)
    if depth <= 0:
        return np.nan, np.nan, np.nan

    half = continuum - depth / 2.0
    im = np.argmin(model_dense)

    def _cross(sw, sf, lev):
        above = sf >= lev
        idx = np.where(np.diff(above.astype(int)))[0]
        if len(idx) == 0:
            return np.nan

        i = idx[-1]
        if sf[i + 1] == sf[i]:
            return sw[i]

        return sw[i] + (lev - sf[i]) / (sf[i + 1] - sf[i]) * (sw[i + 1] - sw[i])

    wl = _cross(wave_dense[:im+1], model_dense[:im+1], half)
    wr = _cross(wave_dense[im:][::-1], model_dense[im:][::-1], half)

    if not (np.isfinite(wl) and np.isfinite(wr)):
        return np.nan, np.nan, np.nan

    return float(wr - wl), float(wl), float(wr)


def absorption_centroid(wave_dense, model_dense, continuum=1.0):
    """
    Absorption-weighted centroid of a model line profile.
    """
    a = continuum - model_dense
    a = np.where(a > 0, a, 0.0)

    s = trapz_integral(a, wave_dense)
    if s <= 0:
        return np.nan

    num = trapz_integral(wave_dense * a, wave_dense)
    return float(num / s)


# ============================================================
# OPTIONAL EXPORT LIST
# ============================================================

__all__ = [
    "trapz_integral",
    "read_wavelength_from_header",
    "load_averaged_fits",
    "der_snr",
    "snr_from_continuum_windows",
    "measure_snr_table",
    "print_snr_table",
    "local_continuum",
    "make_continuum",
    "normalise_flux",
    "local_sigma",
    "find_line_boundaries",
    "ew_from_observed",
    "ew_err_analytic",
    "calc_aic_bic",
    "model_fwhm_generic",
    "absorption_centroid",
]


# In[ ]:




