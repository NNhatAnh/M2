import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

ROOT_DIR = Path(__file__).resolve().parent
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

STEPS: List[Tuple[str, str, Path]] = [
    ("interpolation", "01_interpolation", ROOT_DIR / "01_interpolation.py"),
    ("outlier", "02_outlier_removal", ROOT_DIR / "02_outlier_removal.py"),
    ("filtering", "03_noise_filtering", ROOT_DIR / "03_noise_filtering.py"),
    ("background", "04_background_estimation", ROOT_DIR / "04_background_estimation.py"),
    ("motion", "05_motion_extraction", ROOT_DIR / "05_motion_extraction.py"),
    ("feature", "06_feature_extraction", ROOT_DIR / "06_feature_extraction.py"),
    ("classification", "07_classification", ROOT_DIR / "07_classification.py"),
]


def build_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("pipeline_runner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def normalize_steps(raw_value: str | None) -> List[str]:
    if not raw_value:
        return [step[0] for step in STEPS]

    selected = []
    for item in raw_value.split(","):
        name = item.strip().lower()
        if name:
            selected.append(name)
    return selected


def run_pipeline(selected_steps: List[str], logger: logging.Logger) -> int:
    step_map = {name: (label, path) for name, label, path in STEPS}

    for step_name in selected_steps:
        if step_name not in step_map:
            logger.error("Unknown step '%s'. Valid steps: %s", step_name, ", ".join(step_map.keys()))
            return 2

    logger.info("Pipeline started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Working directory: %s", ROOT_DIR)
    logger.info("Steps to run: %s", ", ".join(selected_steps))

    for step_name in selected_steps:
        label, script_path = step_map[step_name]
        logger.info("=" * 80)
        logger.info("Starting step: %s (%s)", label, script_path.name)
        start_time = time.perf_counter()

        try:
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=ROOT_DIR,
                text=True,
                capture_output=True,
                check=False,
            )
        except Exception as exc:
            logger.exception("Step '%s' crashed: %s", label, exc)
            return 1

        if completed.stdout:
            for line in completed.stdout.splitlines():
                logger.info("[%s] %s", label, line)

        if completed.stderr:
            for line in completed.stderr.splitlines():
                logger.error("[%s] %s", label, line)

        elapsed = time.perf_counter() - start_time
        if completed.returncode == 0:
            logger.info("Completed step: %s in %.2f seconds", label, elapsed)
        else:
            logger.error("Step '%s' failed with exit code %s in %.2f seconds", label, completed.returncode, elapsed)
            return completed.returncode

    logger.info("=" * 80)
    logger.info("Pipeline finished successfully.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full CSI pipeline step by step")
    parser.add_argument(
        "--steps",
        default=None,
        help="Comma-separated step names: interpolation,outlier,filtering,background,motion,feature,classification",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional custom log path. Defaults to logs/pipeline_<timestamp>.log",
    )
    args = parser.parse_args()

    selected_steps = normalize_steps(args.steps)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(args.log_file) if args.log_file else LOG_DIR / f"pipeline_{timestamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = build_logger(log_file)
    logger.info("Log file: %s", log_file)
    return run_pipeline(selected_steps, logger)


if __name__ == "__main__":
    raise SystemExit(main())
