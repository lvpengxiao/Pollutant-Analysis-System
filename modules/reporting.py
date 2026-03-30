from __future__ import annotations

import sys
from datetime import datetime
from typing import Any, Iterable

import customtkinter as ctk
import numpy as np
import pandas as pd
import sklearn


DEFAULT_RANDOM_SEED = 42


def _normalize_feature_names(selected_features: Iterable[Any] | None) -> list[str]:
    names: list[str] = []
    if not selected_features:
        return names
    for item in selected_features:
        if isinstance(item, tuple) and item:
            names.append(str(item[0]))
        else:
            names.append(str(item))
    return names


def _normalize_model_names(enabled_models: Any) -> list[str]:
    if not enabled_models:
        return []
    if isinstance(enabled_models, dict):
        return [str(name) for name, enabled in enabled_models.items() if enabled]
    return [str(name) for name in enabled_models]


def format_preprocessing_summary(
    preprocessing_applied: dict[str, Any] | None,
    log_transformed: bool = False,
) -> str:
    if preprocessing_applied:
        method = preprocessing_applied.get("method", preprocessing_applied.get("type", "Unknown"))
        columns = preprocessing_applied.get("columns") or []
        if columns:
            return f"{method} -> {len(columns)}个变量"
        return str(method)
    if log_transformed:
        return "已启用对数变换"
    return "无"


def build_reproducibility_rows(
    *,
    file_path: str | None = None,
    test_size: Any = None,
    preprocessing_applied: dict[str, Any] | None = None,
    log_transformed: bool = False,
    selected_features: Iterable[Any] | None = None,
    selected_targets: Iterable[Any] | None = None,
    enabled_models: Any = None,
    output_dir: str | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    extra_rows: Iterable[tuple[str, Any]] | None = None,
) -> list[tuple[str, str]]:
    feature_names = _normalize_feature_names(selected_features)
    target_names = [str(item) for item in (selected_targets or [])]
    model_names = _normalize_model_names(enabled_models)

    rows: list[tuple[str, str]] = [
        ("导出时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Python 版本", sys.version.split()[0]),
        ("sklearn 版本", getattr(sklearn, "__version__", "未知")),
        ("pandas 版本", getattr(pd, "__version__", "未知")),
        ("numpy 版本", getattr(np, "__version__", "未知")),
        ("customtkinter 版本", getattr(ctk, "__version__", "未知")),
        ("随机种子", str(random_seed)),
        ("测试集比例", str(test_size) if test_size is not None else "未记录"),
        ("数据文件", file_path or "未记录"),
        ("预处理操作", format_preprocessing_summary(preprocessing_applied, log_transformed)),
        ("特征数量", str(len(feature_names))),
        ("特征列表", ", ".join(feature_names) if feature_names else "未记录"),
        ("目标数量", str(len(target_names))),
        ("目标列表", ", ".join(target_names) if target_names else "未记录"),
        ("启用模型", ", ".join(model_names) if model_names else "未记录"),
        ("输出目录", output_dir or "未记录"),
    ]
    if extra_rows:
        for key, value in extra_rows:
            rows.append((str(key), str(value)))
    return rows


def reproducibility_dataframe(**kwargs: Any) -> pd.DataFrame:
    rows = build_reproducibility_rows(**kwargs)
    return pd.DataFrame(rows, columns=["项目", "值"])
