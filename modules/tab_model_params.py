"""
模型参数标签页模块
提供模型选择开关、通用超参数和良好模型判定阈值配置。
"""

import customtkinter as ctk

from .theme import (C, FONTS, make_card, make_inner_frame, make_section_title,
                    make_hint, make_entry, make_checkbox, make_scrollframe)


XGBOOST_AVAILABLE = True
LIGHTGBM_AVAILABLE = True
CATBOOST_AVAILABLE = True
GAM_AVAILABLE = True
try:
    import xgboost
except ImportError:
    XGBOOST_AVAILABLE = False
try:
    import lightgbm
except ImportError:
    LIGHTGBM_AVAILABLE = False
try:
    import catboost
except ImportError:
    CATBOOST_AVAILABLE = False
try:
    import pygam
except ImportError:
    GAM_AVAILABLE = False


class ModelParamsTab:

    def __init__(self, parent, app):
        self.app = app
        self.parent = parent
        self._build()

    def _build(self):
        scroll = make_scrollframe(self.parent)
        scroll.pack(fill="both", expand=True, padx=6, pady=6)

        self._build_model_card(scroll)
        self._build_param_card(scroll)
        self._build_threshold_card(scroll)

    # ── 模型选择 ──

    def _build_model_card(self, parent):
        card = make_card(parent)
        card.pack(fill="x", padx=12, pady=6)

        make_section_title(card, "选择模型", icon="🤖").pack(
            anchor="w", padx=18, pady=(16, 10))

        grid = make_inner_frame(card)
        grid.pack(fill="x", padx=18, pady=(0, 16))

        models = [
            ('RandomForest', '🌲  随机森林', True,
             '经典集成学习，鲁棒性强，适合特征重要性分析'),
            ('AdaBoost', '🚀  AdaBoost', True,
             '自适应提升，组合多个弱学习器'),
            ('XGBoost', '⚡  XGBoost', XGBOOST_AVAILABLE,
             '极端梯度提升，精度优秀'),
            ('LightGBM', '💡  LightGBM', LIGHTGBM_AVAILABLE,
             '直方图算法，训练速度极快'),
            ('CatBoost', '🐱  CatBoost', CATBOOST_AVAILABLE,
             '原生支持分类特征'),
            ('GAM', '📈  GAM 广义加性模型', GAM_AVAILABLE,
             '半参数模型，可解释性强'),
            ('Stacking', '🔗  Stacking 融合模型', True,
             '自动融合上述已勾选的模型，通常能获得最佳性能'),
        ]

        for key, name, avail, desc in models:
            var = ctk.BooleanVar(value=avail)
            self.app.model_vars[key] = var

            row = ctk.CTkFrame(grid, fg_color=C["bg_tertiary"], corner_radius=8)
            row.pack(fill="x", pady=2)

            cb = make_checkbox(row, text=name, variable=var)
            if not avail:
                cb.configure(state="disabled",
                             text_color=C["text_muted"])
            cb.pack(side="left", padx=14, pady=7)

            ctk.CTkLabel(row, text=desc, font=FONTS["small"](),
                         text_color=C["text_secondary"]).pack(side="left", padx=6)

            if not avail:
                ctk.CTkLabel(row, text="[未安装]", font=FONTS["small"](),
                             text_color=C["error"]).pack(side="right", padx=14)

    # ── 超参数 ──

    def _build_param_card(self, parent):
        card = make_card(parent)
        card.pack(fill="x", padx=12, pady=6)

        make_section_title(card, "通用超参数", icon="🔧").pack(
            anchor="w", padx=18, pady=(16, 10))

        grid = make_inner_frame(card)
        grid.pack(fill="x", padx=18, pady=(0, 16))

        self.app.test_size_var = ctk.DoubleVar(value=0.3)
        self._row(grid, "测试集比例", self.app.test_size_var,
                  "0.1~0.5", "小数据集 0.2，中大数据集 0.3")

        self.app.n_estimators_var = ctk.IntVar(value=100)
        self._row(grid, "树数量 (n_estimators)", self.app.n_estimators_var,
                  "50~500", "决策树数量，默认 100")

        self.app.max_depth_var = ctk.IntVar(value=6)
        self._row(grid, "最大深度 (max_depth)", self.app.max_depth_var,
                  "3~20", "控制树深度，推荐 4~10")

        self.app.learning_rate_var = ctk.DoubleVar(value=0.05)
        self._row(grid, "学习率 (learning_rate)", self.app.learning_rate_var,
                  "0.01~0.3", "Boosting 模型学习率")

        self.app.min_samples_split_var = ctk.IntVar(value=5)
        self._row(grid, "最小分裂样本数", self.app.min_samples_split_var,
                  "2~30", "推荐 5~20")

        self.app.min_samples_leaf_var = ctk.IntVar(value=2)
        self._row(grid, "最小叶子样本数", self.app.min_samples_leaf_var,
                  "1~20", "推荐 2~10")

        self.app.subsample_var = ctk.DoubleVar(value=0.8)
        self._row(grid, "子采样比例 (subsample)", self.app.subsample_var,
                  "0.5~1.0", "仅 XGBoost / LightGBM")

        self.app.colsample_var = ctk.DoubleVar(value=0.8)
        self._row(grid, "列采样比例 (colsample)", self.app.colsample_var,
                  "0.3~1.0", "仅 XGBoost / LightGBM")

        self.app.enable_grid_search_var = ctk.BooleanVar(value=False)
        gs_row = ctk.CTkFrame(grid, fg_color=C["bg_tertiary"], corner_radius=8)
        gs_row.pack(fill="x", pady=2)
        make_checkbox(gs_row, text="启用自动超参数优化 (AutoML / GridSearch)", 
                      variable=self.app.enable_grid_search_var).pack(side="left", padx=14, pady=7)
        ctk.CTkLabel(gs_row, text="勾选后将自动寻找最佳参数（会显著增加分析时间）", font=FONTS["tiny"](),
                     text_color=C["warning"]).pack(side="left", padx=10)

    # ── 阈值 ──

    def _build_threshold_card(self, parent):
        card = make_card(parent)
        card.pack(fill="x", padx=12, pady=6)

        make_section_title(card, "良好模型判定阈值", icon="📏").pack(
            anchor="w", padx=18, pady=(16, 4))
        make_hint(card, '同时满足以下条件的模型被判定为 "良好"').pack(
            anchor="w", padx=18, pady=(0, 10))

        grid = make_inner_frame(card)
        grid.pack(fill="x", padx=18, pady=(0, 16))

        self.app.min_r2_var = ctk.DoubleVar(value=0.1)
        self._th_row(grid, "最小 R²", self.app.min_r2_var,
                     ">0.5 优秀  >0.3 良好  >0.1 可接受")
        self.app.min_scc_var = ctk.DoubleVar(value=0.3)
        self._th_row(grid, "最小 SCC (Spearman)", self.app.min_scc_var,
                     ">0.5 强相关  >0.3 中等相关")
        self.app.min_pcc_var = ctk.DoubleVar(value=0.3)
        self._th_row(grid, "最小 PCC (Pearson)", self.app.min_pcc_var,
                     ">0.5 强相关  >0.3 中等相关")

    # ── 行构建辅助 ──

    def _row(self, parent, label, var, rng, hint):
        f = ctk.CTkFrame(parent, fg_color=C["bg_tertiary"], corner_radius=8)
        f.pack(fill="x", pady=2)
        ctk.CTkLabel(f, text=f"  {label}:", width=225, anchor="w",
                     font=FONTS["body_bold"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(12, 4), pady=7)
        make_entry(f, textvariable=var, width=80, height=30,
                   justify="center").pack(side="left", padx=4)
        ctk.CTkLabel(f, text=f"[{rng}]", width=100,
                     font=FONTS["small"](),
                     text_color=C["warning"]).pack(side="left", padx=4)
        ctk.CTkLabel(f, text=hint, font=FONTS["tiny"](),
                     text_color=C["text_muted"], anchor="w"
                     ).pack(side="left", padx=10, fill="x", expand=True)

    def _th_row(self, parent, label, var, hint):
        f = ctk.CTkFrame(parent, fg_color=C["bg_tertiary"], corner_radius=8)
        f.pack(fill="x", pady=2)
        ctk.CTkLabel(f, text=f"  {label}:", width=200, anchor="w",
                     font=FONTS["body_bold"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(12, 4), pady=7)
        make_entry(f, textvariable=var, width=80, height=30,
                   justify="center").pack(side="left", padx=4)
        ctk.CTkLabel(f, text=hint, font=FONTS["small"](),
                     text_color=C["text_secondary"], anchor="w"
                     ).pack(side="left", padx=14, fill="x", expand=True)
