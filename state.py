from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class AppState:
    df: pd.DataFrame | None = None
    df_backup: pd.DataFrame | None = None
    file_path: str | None = None
    all_columns: list[str] = field(default_factory=list)

    analysis_results: dict[str, pd.DataFrame] = field(default_factory=dict)
    prediction_cache: dict[Any, Any] = field(default_factory=dict)
    model_cache: dict[Any, Any] = field(default_factory=dict)
    X_cache: dict[Any, Any] = field(default_factory=dict)

    cv_results: dict[str, Any] = field(default_factory=dict)
    cv_fold_df: pd.DataFrame | None = None
    cv_detail_df: pd.DataFrame | None = None
    cv_summary_df: pd.DataFrame | None = None

    spearman_results_df: pd.DataFrame | None = None
    comparison_df: pd.DataFrame | None = None
    performance_df: pd.DataFrame | None = None

    label_encoders: dict[str, Any] = field(default_factory=dict)
    preprocessing_applied: dict[str, Any] | None = None
    log_transformed: bool = False
