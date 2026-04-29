#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT ROOT DETECTION
# ============================================================

def get_project_root():
    """
    Find project root automatically.

    Works in:
    - regular .py scripts
    - Jupyter / ipynb
    - interactive console

    Expected project structure:
        object/
            fits/
            config/
            pipeline/
            output/
    """
    if "__file__" in globals():
        start = Path(__file__).resolve().parent
    else:
        start = Path.cwd().resolve()

    for p in [start] + list(start.parents):
        has_pipeline = (p / "pipeline").exists()
        has_fits = (p / "fits").exists()
        has_config = (p / "config").exists()

        if has_pipeline and (has_fits or has_config):
            return p

    if start.name == "pipeline":
        return start.parent

    return start


# ============================================================
# ROOT PATHS
# ============================================================

ROOT = get_project_root()

FITS_DIR = ROOT / "fits"
CONFIG_DIR = ROOT / "config"
OUTPUT_DIR = ROOT / "output"

STACKED_DIR = OUTPUT_DIR / "stacked"
BALMER_DIR = OUTPUT_DIR / "balmer"
LINE_FITS_DIR = OUTPUT_DIR / "line_profile_fits"
TABLES_DIR = OUTPUT_DIR / "tables"
LOGS_DIR = OUTPUT_DIR / "logs"

INPUT_CONFIG_PATH = CONFIG_DIR / "input.dat"
FILE_PATTERNS = ["*.fits", "*.fit", "*.FITS"]


# ============================================================
# CONFIG PARSER
# ============================================================

def _to_bool(val, default=False):
    if val is None:
        return default

    s = str(val).strip().upper()

    if s in {"YES", "TRUE", "1", "ON", "Y"}:
        return True

    if s in {"NO", "FALSE", "0", "OFF", "N"}:
        return False

    return default


def _to_float(val, default):
    try:
        return float(val)
    except Exception:
        return default


def _to_int(val, default):
    try:
        return int(float(val))
    except Exception:
        return default


def _clean_line(line):
    """
    Remove empty lines and comment-only lines.

    Inline comments are also supported:
        KEY = VALUE   # comment
        KEY = VALUE   ; comment
    """
    line = line.strip()

    if not line:
        return ""

    if line.startswith("#") or line.startswith(";"):
        return ""

    # Remove inline comments.
    for marker in ("#", ";"):
        if marker in line:
            line = line.split(marker, 1)[0].strip()

    return line


def parse_input_file(path):
    """
    Parse combined input.dat file with sections:

    [PIPELINE]
    KEY = VALUE
    ...

    [LINES]
    line_id    label    wavelength
    Halpha     Halpha   6562.80
    Hgamma     Hgamma   4340.47
    ...

    Returns
    -------
    pipeline_dict : dict
        Dictionary with pipeline control parameters.

    df_lines : pandas.DataFrame
        Line table with columns:
        line_id, label, wavelength
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input config not found: {path}")

    pipeline_dict = {}
    lines_block = []

    section = None

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = _clean_line(raw)

            if not line:
                continue

            upper = line.upper()

            if upper == "[PIPELINE]":
                section = "PIPELINE"
                continue

            if upper == "[LINES]":
                section = "LINES"
                continue

            if line.startswith("[") and line.endswith("]"):
                section = None
                continue

            if section == "PIPELINE":
                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                pipeline_dict[key.strip().upper()] = value.strip()

            elif section == "LINES":
                lines_block.append(line)

    if not lines_block:
        raise ValueError("No [LINES] section or empty line list in input.dat")

    header = lines_block[0].split()
    rows = []

    for row in lines_block[1:]:
        parts = row.split()

        if len(parts) == 0:
            continue

        if len(parts) != len(header):
            raise ValueError(
                "Bad row in [LINES] section:\n"
                f"  {row}\n"
                f"Expected {len(header)} columns: {header}, got {len(parts)}"
            )

        rows.append(parts)

    df_lines = pd.DataFrame(rows, columns=header)

    required = {"line_id", "label", "wavelength"}
    missing = required - set(df_lines.columns)

    if missing:
        raise ValueError(f"Missing required columns in [LINES]: {missing}")

    df_lines["line_id"] = df_lines["line_id"].astype(str).str.strip()
    df_lines["label"] = df_lines["label"].astype(str).str.strip()
    df_lines["wavelength"] = df_lines["wavelength"].astype(float)

    if df_lines.empty:
        raise ValueError("[LINES] section was found, but no valid lines were parsed.")

    return pipeline_dict, df_lines


PIPELINE_CFG, INPUT_LINE_DF = parse_input_file(INPUT_CONFIG_PATH)


# ============================================================
# PIPELINE CONTROL KEYS
# ============================================================

ANALYSIS_MODE = str(
    PIPELINE_CFG.get("ANALYSIS_MODE", "AVERAGED")
).strip().upper()

SAVE_STACKED = _to_bool(
    PIPELINE_CFG.get("SAVE_STACKED", "YES"),
    True
)

SAVE_PLOTS = _to_bool(
    PIPELINE_CFG.get("SAVE_PLOTS", "YES"),
    True
)

SAVE_FITS = _to_bool(
    PIPELINE_CFG.get("SAVE_FITS", "YES"),
    True
)

OUTPUT_TAG = str(
    PIPELINE_CFG.get("OUTPUT_TAG", "run")
).strip()

APPLY_RV_CHECK = _to_bool(
    PIPELINE_CFG.get("APPLY_RV_CHECK", "YES"),
    True
)

APPLY_RV_CORRECTION = _to_bool(
    PIPELINE_CFG.get("APPLY_RV_CORRECTION", "NO"),
    False
)

RV_SHIFT_WARN_KMS = _to_float(
    PIPELINE_CFG.get("RV_SHIFT_WARN_KMS", 5.0),
    5.0
)

USE_SIGMA_CLIP = _to_bool(
    PIPELINE_CFG.get("USE_SIGMA_CLIP", "NO"),
    False
)

SIGMA_CLIP = _to_float(
    PIPELINE_CFG.get("SIGMA_CLIP", 3.0),
    3.0
)

SNR_DETECT = _to_float(
    PIPELINE_CFG.get("SNR_DETECT", 3.0),
    3.0
)

SNR_RELIABLE = _to_float(
    PIPELINE_CFG.get("SNR_RELIABLE", 5.0),
    5.0
)

GROUP_MODE = str(
    PIPELINE_CFG.get("GROUP_MODE", "NONE")
).strip().upper()

GROUP_KEY = str(
    PIPELINE_CFG.get("GROUP_KEY", "PHASE")
).strip()

ROUND_DIGITS = _to_int(
    PIPELINE_CFG.get("ROUND_DIGITS", 2),
    2
)


# ============================================================
# LEGACY / COMPATIBILITY KEYS
# ============================================================

INPUT_LINE_LIST = INPUT_CONFIG_PATH

AVERAGED_FITS_PATH = STACKED_DIR / (
    f"averaged_spectrum_{OUTPUT_TAG}.fits"
    if OUTPUT_TAG
    else "averaged_spectrum.fits"
)

USE_AVERAGING = ANALYSIS_MODE in {"AVERAGED", "BOTH"}


# ============================================================
# DISPLAY SETTINGS
# ============================================================

DISPLAY_RANGE = (4250, 6700)
RNG_SEED = 42


# ============================================================
# SNR WINDOWS FOR STACKING / QC
# ============================================================

SNR_REGIONS = {
    "cont_4515": (4515, 4510, 4520),
    "cont_4715": (4715, 4710, 4720),
    "cont_5010": (5010, 5005, 5015),
    "cont_5545": (5545, 5540, 5550),
    "cont_6090": (6090, 6085, 6095),
}


# ============================================================
# LINE LABELS FOR OVERVIEW PLOT
# ============================================================

def _pretty_line_label(label):
    """
    Convert input.dat line labels into publication-style plot labels.

    Input examples:
        Halpha -> H$\\alpha$
        Hgamma -> H$\\gamma$
        MgII   -> Mg II
        FeII   -> Fe II
        SiII   -> Si II
    """
    label = str(label).strip()

    mapping = {
        "Halpha": r"H$\alpha$",
        "Hbeta": r"H$\beta$",
        "Hgamma": r"H$\gamma$",
        "Hdelta": r"H$\delta$",
        "Hepsilon": r"H$\epsilon$",
        "MgI": r"Mg$\,$I",
        "MgII": r"Mg$\,$II",
        "FeI": r"Fe$\,$I",
        "FeII": r"Fe$\,$II",
        "SiI": r"Si$\,$I",
        "SiII": r"Si$\,$II",
        "CaI": r"Ca$\,$I",
        "CaII": r"Ca$\,$II",
        "HeI": r"He$\,$I",
        "HeII": r"He$\,$II",
    }

    return mapping.get(label, label)


def build_line_labels_from_input(df_lines):
    """
    Build plot labels from the [LINES] block in input.dat.

    Now wavelength is appended to EVERY label for a uniform style:
        Halpha 6563
        Hgamma 4340
        Mg II 4481
        Mg I 5184
        Fe II 5317
        Si II 6347
        Si II 6371
    """
    line_labels = {}
    used_labels = {}

    for _, row in df_lines.iterrows():
        pretty_label = _pretty_line_label(row["label"])
        wavelength = float(row["wavelength"])

        # wavelength is always added
        plot_label = rf"{pretty_label} {wavelength:.0f}"

        # safety in case of accidental exact duplicates
        if plot_label in used_labels:
            used_labels[plot_label] += 1
            unique_label = f"{plot_label}_{used_labels[plot_label]}"
        else:
            used_labels[plot_label] = 1
            unique_label = plot_label

        line_labels[unique_label] = wavelength

    return line_labels


# This is what s02_prepare_spectra.py imports for the overview plot.
# It is now built directly from the [LINES] block in input.dat.
LINE_LABELS = build_line_labels_from_input(INPUT_LINE_DF)


# ============================================================
# BALMER-LINE DEFINITIONS
# ============================================================

BALMER_LINES = [
    dict(
        name=r"H$\alpha$",
        ascii="Halpha",
        rest=6562.80,
        fit_window=(6530.0, 6600.0),
        cont_left=(6520.0, 6532.0),
        cont_right=(6598.0, 6612.0),
    ),
    dict(
        name=r"H$\beta$",
        ascii="Hbeta",
        rest=4861.33,
        fit_window=(4835.0, 4890.0),
        cont_left=(4828.0, 4837.0),
        cont_right=(4888.0, 4900.0),
    ),
    dict(
        name=r"H$\gamma$",
        ascii="Hgamma",
        rest=4340.47,
        fit_window=(4315.0, 4368.0),
        cont_left=(4308.0, 4317.0),
        cont_right=(4366.0, 4378.0),
    ),
    dict(
        name=r"H$\delta$",
        ascii="Hdelta",
        rest=4101.74,
        fit_window=(4078.0, 4128.0),
        cont_left=(4072.0, 4080.0),
        cont_right=(4126.0, 4136.0),
    ),
    dict(
        name=r"H$\epsilon$",
        ascii="Hepsilon",
        rest=3970.07,
        fit_window=(3948.0, 3996.0),
        cont_left=(3942.0, 3950.0),
        cont_right=(3994.0, 4004.0),
    ),
    dict(
        name="H8",
        ascii="H8",
        rest=3889.05,
        fit_window=(3868.0, 3912.0),
        cont_left=(3862.0, 3870.0),
        cont_right=(3910.0, 3920.0),
    ),
    dict(
        name="H9",
        ascii="H9",
        rest=3835.39,
        fit_window=(3816.0, 3856.0),
        cont_left=(3810.0, 3818.0),
        cont_right=(3854.0, 3864.0),
    ),
    dict(
        name="H10",
        ascii="H10",
        rest=3797.90,
        fit_window=(3780.0, 3818.0),
        cont_left=(3774.0, 3782.0),
        cont_right=(3816.0, 3826.0),
    ),
]


# ============================================================
# BALMER MULTICOMPONENT SETTINGS
# ============================================================

DENSE_NPTS_BALMER = 10_000
SMOOTH_WIN_BALMER = 11
SMOOTH_POLY_BALMER = 3

K_SIGMA_BALMER = 1.5
FRAC_DEPTH_BALMER = 0.01


# ============================================================
# GENERAL LINE-FITTING SETTINGS
# ============================================================

LINE_HALF_WINDOW = 6.0
CONT_GAP = 2.0
CONT_WIDTH = 3.0

SMOOTH_WIN_LINE = 9
SMOOTH_POLY_LINE = 2

K_SIGMA_LINE = 0.7
FRAC_DEPTH_LINE = 0.003


# ============================================================
# MODEL SELECTION / MONTE CARLO
# ============================================================

DBIC_SIMPLE_EQUIV = 2.0
DBIC_BLEND_STRONG = 6.0

N_MC_LINE = 300
N_MC_BROAD_COMPLEX = 120


# ============================================================
# HELPER ACCESSORS
# ============================================================

def get_pipeline_config():
    return dict(PIPELINE_CFG)


def get_line_table():
    return INPUT_LINE_DF.copy()


def get_line_labels():
    return dict(LINE_LABELS)


# ============================================================
# OPTIONAL CHECK
# ============================================================

if __name__ == "__main__" or "__file__" not in globals():
    print("ROOT =", ROOT)
    print("INPUT_CONFIG_PATH =", INPUT_CONFIG_PATH)
    print("ANALYSIS_MODE =", ANALYSIS_MODE)
    print("SAVE_STACKED =", SAVE_STACKED)
    print("SAVE_PLOTS =", SAVE_PLOTS)
    print("SAVE_FITS =", SAVE_FITS)
    print("OUTPUT_TAG =", OUTPUT_TAG)

    print("\nLINES FROM input.dat:")
    print(INPUT_LINE_DF.to_string(index=False))

    print("\nLINE_LABELS USED FOR OVERVIEW PLOT:")
    for label, wavelength in LINE_LABELS.items():
        print(f"{label:20s} {wavelength:10.2f}")


# In[ ]:




