from pathlib import Path

# =========================================================
# PROJECT PATH
# =========================================================

# Project root
ROOT_DIR = Path(__file__).resolve().parent

# =========================================================
# DATA PATH
# =========================================================
DATA_DIR = ROOT_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURE_DATA_DIR = DATA_DIR / "features"

# =========================================================
# RESULT PATH
# =========================================================
RESULT_DIR = ROOT_DIR / "results"

INTERP_DIR = RESULT_DIR / "interpolation"
OUTLIER_DIR = RESULT_DIR / "outlier"
FILTER_DIR = RESULT_DIR / "filtering"
BACKGROUND_DIR = RESULT_DIR / "background"
MOTION_DIR = RESULT_DIR / "motion"
FEATURE_DIR = RESULT_DIR / "feature"
CLASSIFICATION_DIR = RESULT_DIR / "classification"
BENCHMARK_DIR = RESULT_DIR / "benchmark"

# Create folders automatically
ALL_DIRS = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FEATURE_DATA_DIR,
    RESULT_DIR,
    INTERP_DIR,
    OUTLIER_DIR,
    FILTER_DIR,
    BACKGROUND_DIR,
    MOTION_DIR,
    FEATURE_DIR,
    CLASSIFICATION_DIR,
    BENCHMARK_DIR
]

for folder in ALL_DIRS:
    folder.mkdir(parents=True, exist_ok=True)

# =========================================================
# DATASET
# =========================================================
DATASETS = {
    "static": RAW_DATA_DIR / "csi_log.csv",
    "motion": RAW_DATA_DIR / "csi_log1.csv"
}

# =========================================================
# METHODS
# =========================================================
METHODS = ["butterworth", "median", "moving_average"]

# =========================================================
# CSV FORMAT
# =========================================================
CSV_SEPARATOR = ";"
TIME_COLUMN = "time_ms"
LABEL_COLUMN = "label"
FEATURE_COLUMNS = [
    "amp_mean",
    "amp_std",
    "amp_max",
    "phase_std",
    "i_mean",
    "q_mean"
]

# =========================================================
# RESAMPLING
# =========================================================
TARGET_FS = 10.0
INTERPOLATION_METHOD = "linear"

# =========================================================
# VISUALIZATION
# =========================================================
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
PLOT_STYLE = "default"

# =========================================================
# COLORS
# =========================================================
STATIC_COLOR = "#1f77b4"
MOTION_COLOR = "#d62728"
INTERP_COLOR = "#2ca02c"

# =========================================================
# QUALITY METRICS
# =========================================================
ENABLE_SQI = True
ENABLE_MCI = True
ENABLE_PSI = True
ENABLE_TCI = True
ENABLE_PACKET_LOSS_ESTIMATION = True
ENABLE_LOOCV = True

# =========================================================
# REPORT
# =========================================================
SAVE_REPORT = True
SAVE_STATISTICS = True
SAVE_PROCESSED_CSV = True
SAVE_SUMMARY_CSV = True

# =========================================================
# RANDOM
# =========================================================
RANDOM_SEED = 42
