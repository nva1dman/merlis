#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd

from s00_settings import (
    TABLES_DIR,
    ROUND_DIGITS,
)

from s04_general_line_fitting import run_general_line_analysis


# ============================================================
# RESULT COLLECTION
# ============================================================

def collect_result_row(res):
    """
    Convert one result dictionary from s04_general_line_fitting.py
    into a flat row for the output table.
    """
    if res is None:
        return None

    rec = res.get("recommended", None)
    li = res.get("line", {})
    src = res.get("source_line", {})
    spec = res.get("source_spectrum", {})
    status = res.get("status", "unknown")
    profile_class = res.get("profile_class", None)

    if rec is None:
        mr = {}
        model_name = None
    else:
        mr = res["models"][rec]
        model_name = rec

    return {
        "spectrum_id": spec.get("spectrum_id", ""),
        "spectrum_kind": spec.get("spectrum_kind", ""),
        "filename": spec.get("filename", ""),
        "object_name": spec.get("object_name", ""),

        "line": src.get("line_id", li.get("ascii", "")),
        "label": src.get("label", li.get("name", "")),
        "rest": src.get("rest_input", li.get("rest", np.nan)),

        "status": status,
        "profile_class": profile_class,
        "model": model_name,

        "center": mr.get("center_fit", np.nan),
        "center_err": mr.get("center_err", np.nan),

        "dv_km_s": mr.get("dv", np.nan),
        "dv_err": mr.get("dv_err", np.nan),

        "depth": mr.get("depth_fit", np.nan),
        "depth_err": mr.get("depth_err", np.nan),

        "fwhm": mr.get("fwhm", np.nan),
        "fwhm_err": mr.get("fwhm_err", np.nan),

        "ew": mr.get("ew_obs", np.nan),
        "ew_err": mr.get("ew_err_mc", mr.get("ew_err", np.nan)),

        "lam_left": mr.get("lam_left", np.nan),
        "lam_left_err": mr.get("lam_left_err", np.nan),

        "lam_right": mr.get("lam_right", np.nan),
        "lam_right_err": mr.get("lam_right_err", np.nan),

        "sigma_local": res.get("empirical", {}).get("sigma_local", np.nan),
        "snr_depth": res.get("empirical", {}).get("snr_depth", np.nan),

        "aic": mr.get("aic", np.nan),
        "bic": mr.get("bic", np.nan),
        "rms": mr.get("rms", np.nan),
        "n_mc_ok": mr.get("n_mc_ok", np.nan),
    }


# ============================================================
# ROUNDING HELPERS
# ============================================================

def _round_value(x, ndigits=ROUND_DIGITS):
    if isinstance(x, (float, np.floating)):
        if np.isfinite(x):
            return round(float(x), ndigits)
        return np.nan
    return x


def round_numeric_dataframe(df, ndigits=ROUND_DIGITS):
    """
    Round all numeric columns in a DataFrame.
    """
    df2 = df.copy()

    for col in df2.columns:
        if pd.api.types.is_numeric_dtype(df2[col]):
            df2[col] = df2[col].apply(lambda x: _round_value(x, ndigits))

    return df2


# ============================================================
# STRING FORMATTERS FOR SHORT TABLE
# ============================================================

def pm_string(val, err, ndigits=ROUND_DIGITS):
    """
    Format value ± error string.
    """
    if pd.isna(val):
        return ""

    if pd.isna(err):
        return f"{val:.{ndigits}f}"

    return f"{val:.{ndigits}f} ± {err:.{ndigits}f}"


def build_short_table(df_full, ndigits=ROUND_DIGITS):
    """
    Build reduced publication-friendly table with selected parameters only.
    """
    rows = []

    for _, r in df_full.iterrows():
        rows.append({
            "spectrum_id": r.get("spectrum_id", ""),
            "filename": r.get("filename", ""),
            "line": r.get("line", ""),
            "center": pm_string(r.get("center", np.nan), r.get("center_err", np.nan), ndigits),
            "EW": pm_string(r.get("ew", np.nan), r.get("ew_err", np.nan), ndigits),
            "FWHM": pm_string(r.get("fwhm", np.nan), r.get("fwhm_err", np.nan), ndigits),
            "depth": pm_string(r.get("depth", np.nan), r.get("depth_err", np.nan), ndigits),
        })

    return pd.DataFrame(rows)


# ============================================================
# SAVE TABLES
# ============================================================

def save_tables(rows, output_dir=TABLES_DIR):
    """
    Save:
    - full rounded CSV
    - short reduced CSV
    """
    if not rows:
        print("No rows to save.")
        return None, None, None, None

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_full = pd.DataFrame(rows)

    sort_cols = [c for c in ["spectrum_kind", "spectrum_id", "filename", "rest"] if c in df_full.columns]
    if sort_cols:
        df_full = df_full.sort_values(sort_cols).reset_index(drop=True)

    df_full_round = round_numeric_dataframe(df_full, ROUND_DIGITS)
    df_short = build_short_table(df_full_round, ROUND_DIGITS)

    fp_full = outdir / "line_measurements.csv"
    fp_short = outdir / "line_measurements_short.csv"

    df_full_round.to_csv(fp_full, index=False)
    df_short.to_csv(fp_short, index=False)

    print(f"  Full CSV  -> {fp_full}")
    print(f"  Short CSV -> {fp_short}")

    return df_full, df_full_round, df_short, (fp_full, fp_short)


# ============================================================
# HIGH-LEVEL DRIVER
# ============================================================

def build_rows_from_results(all_results):
    """
    Convert full result list into flat row list.
    """
    rows = []

    for res in all_results:
        rr = collect_result_row(res)
        if rr is not None:
            rows.append(rr)

    return rows


def run_tables_export():
    """
    Full table-export step:
    1. run general line analysis
    2. collect rows
    3. save full + short CSV tables
    """
    all_results = run_general_line_analysis()
    rows = build_rows_from_results(all_results)

    df_full, df_full_round, df_short, filepaths = save_tables(rows, TABLES_DIR)

    print("\nFull summary:")
    print(df_full_round.to_string(index=False))

    print("\nShort summary:")
    print(df_short.to_string(index=False))

    return {
        "all_results": all_results,
        "rows": rows,
        "df_full": df_full,
        "df_full_round": df_full_round,
        "df_short": df_short,
        "filepaths": filepaths,
    }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "collect_result_row",
    "round_numeric_dataframe",
    "pm_string",
    "build_short_table",
    "save_tables",
    "build_rows_from_results",
    "run_tables_export",
]


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run_tables_export()


# In[ ]:




