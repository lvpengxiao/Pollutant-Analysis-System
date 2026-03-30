from __future__ import annotations

import os
import tempfile
from pathlib import Path
from contextlib import contextmanager
from typing import Generator


LEARNING_CURVE_TRAIN_POINTS = 5
SHAP_MAX_SAMPLES = 100
HEATMAP_MAX_FEATURES = 20
PDF_HEATMAP_MAX_FEATURES = 15
DISTRIBUTION_PLOT_MAX_FEATURES = 24
ANALYSIS_FIT_POLL_INTERVAL = 0.5
ANALYSIS_FIT_TIMEOUT_SECONDS = 600

MATPLOTLIB_FONT_FAMILIES = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]

APP_CONFIG_DIR_NAME = "PollutantAnalysis"
APP_CONFIG_FILE_NAME = "pollutant_analysis_config.json"
APP_LOG_FILE_NAME = "pollutant_analysis.log"


def get_app_config_dir() -> Path:
    preferred_dir: Path
    try:
        from platformdirs import user_config_dir

        preferred_dir = Path(user_config_dir(APP_CONFIG_DIR_NAME, roaming=True))
    except Exception:
        base_dir = os.getenv("APPDATA") or os.getenv("XDG_CONFIG_HOME")
        if not base_dir:
            base_dir = str(Path.home() / ".config")
        preferred_dir = Path(base_dir) / APP_CONFIG_DIR_NAME
    return _ensure_writable_dir(
        preferred_dir,
        Path(tempfile.gettempdir()) / APP_CONFIG_DIR_NAME,
    )


def get_app_config_path() -> Path:
    return get_app_config_dir() / APP_CONFIG_FILE_NAME


def ensure_app_config_dir() -> Path:
    return get_app_config_dir()


def get_app_log_dir() -> Path:
    preferred_dir: Path
    try:
        from platformdirs import user_log_dir

        preferred_dir = Path(user_log_dir(APP_CONFIG_DIR_NAME, roaming=True))
    except Exception:
        preferred_dir = ensure_app_config_dir() / "logs"
    return _ensure_writable_dir(
        preferred_dir,
        ensure_app_config_dir() / "logs",
    )


def get_app_log_path() -> Path:
    return get_app_log_dir() / APP_LOG_FILE_NAME


def ensure_app_log_dir() -> Path:
    return get_app_log_dir()


def _ensure_writable_dir(preferred_dir: Path, fallback_dir: Path) -> Path:
    cwd_fallback = Path.cwd() / APP_CONFIG_DIR_NAME
    for candidate in (preferred_dir, fallback_dir, cwd_fallback):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except Exception:
            continue
    return Path(tempfile.gettempdir())


def configure_matplotlib(plt_module) -> None:
    plt_module.rcParams["font.sans-serif"] = MATPLOTLIB_FONT_FAMILIES
    plt_module.rcParams["axes.unicode_minus"] = False


@contextmanager
def managed_figure(figsize: tuple[float, float] = (10, 6), dpi: int = 100) -> Generator:
    """
    统一的 matplotlib Figure 管理器：自动创建和清理 Figure。
    
    示例：
        with managed_figure((12, 8)) as fig:
            ax = fig.add_subplot(111)
            ax.plot([1, 2, 3])
            # 使用 fig 和 ax 绘图
            # 退出时自动调用 plt.close(fig)
    """
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    try:
        yield fig
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass  # 安全清理：即使失败也继续
