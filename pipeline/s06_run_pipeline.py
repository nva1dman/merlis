#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path

from s00_settings import (
    OUTPUT_DIR,
    STACKED_DIR,
    BALMER_DIR,
    LINE_FITS_DIR,
    TABLES_DIR,
    LOGS_DIR,
    ANALYSIS_MODE,
    OUTPUT_TAG,
)

from s02_prepare_spectra import run_prepare_spectra
from s04_general_line_fitting import run_general_line_analysis
from s05_tables import build_rows_from_results, save_tables


# ============================================================
# DIRECTORY SETUP
# ============================================================

def ensure_directories():
    """
    Create all required output directories.
    """
    for path in [
        OUTPUT_DIR,
        STACKED_DIR,
        BALMER_DIR,
        LINE_FITS_DIR,
        TABLES_DIR,
        LOGS_DIR,
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


# ============================================================
# MAIN DRIVER
# ============================================================

def run_pipeline():
    """
    Full pipeline:
    1. prepare spectra and build analysis targets
    2. analyse all requested spectra/targets
    3. build and save summary tables
    """
    ensure_directories()

    print("\n" + "=" * 72)
    print("  FULL SPECTRAL PIPELINE")
    print("=" * 72)
    print(f"  ANALYSIS_MODE : {ANALYSIS_MODE}")
    print(f"  OUTPUT_TAG    : {OUTPUT_TAG}")

    print("\n[1/3] Preparing spectra...")
    prep_result = run_prepare_spectra()
    analysis_targets = prep_result["analysis_targets"]

    print("\n[2/3] Running line analysis...")
    all_results = run_general_line_analysis(analysis_targets=analysis_targets)

    print("\n[3/3] Exporting tables...")
    rows = build_rows_from_results(all_results)
    df_full, df_full_round, df_short, filepaths = save_tables(rows, TABLES_DIR)

    print("\n" + "=" * 72)
    print("  PIPELINE FINISHED")
    print("=" * 72)
    print(f"  Analysis targets processed : {len(analysis_targets)}")
    print(f"  Total line-result rows     : {len(rows)}")

    if filepaths is not None:
        fp_full, fp_short = filepaths
        print(f"  Full table : {fp_full}")
        print(f"  Short table: {fp_short}")

    return {
        "prepare_result": prep_result,
        "analysis_targets": analysis_targets,
        "all_results": all_results,
        "rows": rows,
        "df_full": df_full,
        "df_full_round": df_full_round,
        "df_short": df_short,
        "filepaths": filepaths,
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run_pipeline()

