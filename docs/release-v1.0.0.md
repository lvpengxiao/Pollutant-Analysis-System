# v1.0.0 - Pollutant Analysis System (RI-focused)

## 版本简介
本版本聚焦于新污染物影响因素相对重要性（RI）评估流程的可用性与结果可解释性，提供从数据导入、预处理、建模、交叉验证到图表导出的完整桌面化分析链路。

## 主要亮点
- RI 导向的分析流程：围绕影响因素相对重要性排序与解释输出。
- 多模型支持：RF / AdaBoost / XGBoost / LightGBM / CatBoost / GAM / Stacking。
- 可解释性输出：Permutation Importance、Spearman、SHAP 图表。
- 端到端桌面流程：导入 -> 预处理 -> 分析 -> CV -> 可视化 -> 导出。

## 使用方式
### 普通用户
1. 在 Releases 下载最新 Windows ZIP 包。
2. 解压后双击主程序运行。

### 开发者
1. 安装依赖：`pip install -r requirements.txt`
2. 源码运行：`python run.py`

## 结果解释边界
- 模型重要性与统计相关性不直接等同于因果关系。
- 结果受样本规模、数据质量、特征工程与参数设置影响。
- 建议结合领域机理与独立验证进行结论确认。

## 修复与改进（本轮）
- 修复重要性归一化逻辑（负值截断后归一化）。
- 移除树模型路径中不必要缩放，提高口径一致性。
- 增强小样本场景下 PI 稳定性。
- 增加 CatBoost 原生分类特征训练路径。
- 修复 Spearman 分类特征处理。
- 修复数据加载后预处理页状态同步问题。

## English Summary
This release focuses on RI-oriented pollutant analysis with reproducible workflow and explainability. It provides an end-to-end desktop pipeline from data loading to preprocessing, modeling, cross-validation, visualization, and report export.
