"""
机器学习分析引擎
提供模型构建、评估、特征重要性提取等核心计算功能。
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

logger = logging.getLogger(__name__)


def prepare_feature_frame(
    df: pd.DataFrame,
    selected_features: list[tuple[str, str]]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    """Normalize feature dtypes while keeping the raw feature frame for pipelines."""
    df_work = df.copy()
    feat_names: list[str] = []
    numeric_features: list[str] = []
    categorical_features: list[str] = []

    for col, feature_type in selected_features:
        feat_names.append(col)
        if feature_type == "numeric":
            numeric_features.append(col)
            df_work[col] = pd.to_numeric(df_work[col], errors="coerce")
        else:
            categorical_features.append(col)
            df_work[col] = df_work[col].astype(str)

    return df_work, df_work[feat_names].copy(), feat_names, numeric_features, categorical_features


def build_feature_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str]
) -> ColumnTransformer:
    """Create the shared preprocessing pipeline used by analysis and CV."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",
    )


def build_model_pipeline(
    model: Any,
    numeric_features: list[str],
    categorical_features: list[str]
) -> Pipeline:
    """Wrap a model with the shared feature preprocessor."""
    return Pipeline(steps=[
        ("preprocessor", build_feature_preprocessor(numeric_features, categorical_features)),
        ("model", model),
    ])


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict[str, float]:
    """
    计算回归模型的多个评估指标。
    
    Args:
        y_true: 真实目标值数组
        y_pred: 预测目标值数组
        
    Returns:
        包含 R², RMSE, SCC, PCC 的字典
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    scc, _ = spearmanr(y_true, y_pred)
    pcc, _ = pearsonr(y_true, y_pred)
    return {
        'R²': round(r2, 4),
        'RMSE': round(rmse, 4),
        'SCC': round(scc, 4),
        'PCC': round(pcc, 4),
    }


def normalize_importances(raw: np.ndarray) -> np.ndarray:
    """
    将重要性数组归一化为 [0, 1] 范围。
    
    Args:
        raw: 原始重要性数组
        
    Returns:
        归一化后的重要性数组
    """
    clipped = np.maximum(raw, 0)
    total = clipped.sum()
    if total < 1e-10:
        return clipped
    return clipped / total


def spearman_for_target(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    feature_names: list[str],
    target_name: str,
    category: str
) -> list[dict[str, Any]]:
    """
    计算所有特征与目标变量的 Spearman 相关系数。
    
    Args:
        X: 特征 DataFrame
        y: 目标变量向量
        feature_names: 特征名称列表
        target_name: 目标变量名称
        category: 目标变量类别标签
        
    Returns:
        包含相关性结果的字典列表，每个字典包含目标变量、类别、影响因素、
        Spearman_r、P值和显著性标记
    """
    results = []
    
    # Check if y is numeric
    try:
        y_numeric = pd.to_numeric(y, errors='coerce')
    except Exception as e:
        logger.warning(f"无法将目标变量转换为数值: {e}, 使用原始值")
        y_numeric = y
        
    for feat in feature_names:
        # Create a mask for valid (non-NaN) values in both X[feat] and y
        try:
            x_raw = X[feat]

            if pd.api.types.is_numeric_dtype(x_raw):
                x_feat = pd.to_numeric(x_raw, errors='coerce')
            else:
                # Spearman for categorical features uses label ranks as an approximation.
                x_series = x_raw.astype(str).fillna('missing')
                encoder = LabelEncoder()
                x_feat = pd.Series(encoder.fit_transform(x_series), index=x_series.index, dtype=float)
        except Exception as e:
            logger.debug(f"特征 {feat} 无法转换为数值: {e}，跳过")
            continue
            
        valid = ~(np.isnan(x_feat) | np.isnan(y_numeric))
        if valid.sum() < 2:
            r, p = np.nan, np.nan
        else:
            r, p = spearmanr(x_feat[valid], y_numeric[valid])
            
        results.append({
            '目标变量': target_name,
            '类别': category,
            '影响因素': feat,
            'Spearman_r': round(r, 4) if not np.isnan(r) else np.nan,
            'P值': p,
            '显著性': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns',
            '|r|': abs(r) if not np.isnan(r) else np.nan,
        })
    return results
