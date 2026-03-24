"""
==========================================================================
多类新污染物室内分布影响因素高级分析程序 - CustomTkinter 现代化UI版
==========================================================================

功能说明:
  - 支持自定义选择影响因素(特征)、污染物(目标变量)和模型参数
  - 支持6种机器学习模型: RandomForest, AdaBoost, XGBoost, LightGBM, CatBoost, GAM
  - Spearman / Pearson 相关性分析
  - 自动导出 Excel 综合结果报告

UI框架: CustomTkinter
  - 暗黑模式 + 圆角组件 + 扁平化设计
  - 原生桌面应用，不依赖浏览器
  - 支持 pyinstaller 打包为 exe

安装依赖命令 (在命令行/终端中逐行执行):
  pip install customtkinter
  pip install pandas openpyxl numpy scipy scikit-learn
  pip install xgboost          # 可选: XGBoost模型
  pip install lightgbm         # 可选: LightGBM模型
  pip install catboost          # 可选: CatBoost模型
  pip install pygam             # 可选: GAM模型
  pip install pyinstaller       # 可选: 用于打包exe

打包exe命令:
  pyinstaller --onefile --windowed --name="污染物分析系统" 2.py

作者: AI Assistant
版本: 2.0
日期: 2024
==========================================================================
"""

# ==================== 标准库导入 ====================
import os
import threading
import traceback
from datetime import datetime

# ==================== 第三方核心库导入 ====================
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import warnings

# ==================== UI框架导入 ====================
# CustomTkinter: 基于Tkinter的现代化UI库，自带暗黑模式、圆角组件
# 官方文档: https://customtkinter.tomschimansky.com/
import customtkinter as ctk
from tkinter import filedialog  # 文件对话框仍使用tkinter原生

# 忽略不必要的警告信息，保持控制台整洁
warnings.filterwarnings('ignore')

# ==================== 可选机器学习库检测 ====================
# 以下库为可选依赖，未安装时对应模型将自动禁用

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from pygam import LinearGAM, s, f

    GAM_AVAILABLE = True
except ImportError:
    GAM_AVAILABLE = False

# ==================== 全局UI配色方案 ====================
# 自定义暗黑极简风格配色，保证视觉统一和高可读性
COLORS = {
    "bg_dark": "#1a1a2e",  # 最深背景色 (主窗口)
    "bg_medium": "#16213e",  # 中等深度背景 (卡片/面板)
    "bg_light": "#0f3460",  # 浅色背景 (高亮区域)
    "accent": "#e94560",  # 强调色/主题色 (按钮/高亮)
    "accent_hover": "#c73e54",  # 强调色悬浮态
    "text_primary": "#eaeaea",  # 主文字颜色
    "text_secondary": "#a0a0b0",  # 次要文字颜色
    "success": "#2ecc71",  # 成功/完成状态色
    "warning": "#f39c12",  # 警告色
    "error": "#e74c3c",  # 错误色
    "border": "#2a2a4a",  # 边框颜色
    "input_bg": "#1e1e3a",  # 输入框背景色
    "card_bg": "#1e1e38",  # 卡片背景色
    "tab_selected": "#e94560",  # 选中标签页颜色
    "tab_unselected": "#2a2a4a",  # 未选中标签页颜色
    "scrollbar": "#3a3a5a",  # 滚动条颜色
    "checkbox_on": "#e94560",  # 复选框选中颜色
    "checkbox_off": "#3a3a5a",  # 复选框未选中颜色
    "progress_bg": "#2a2a4a",  # 进度条背景
    "progress_fill": "#e94560",  # 进度条填充
}


class PollutantAnalysisApp(ctk.CTk):
    """
    污染物影响因素分析系统 - 主应用类
    ==========================================
    使用 CustomTkinter 构建现代化暗黑风格桌面GUI应用。
    包含5个功能页面: 数据加载 → 特征选择 → 目标变量 → 模型参数 → 运行分析
    """

    def __init__(self):
        super().__init__()

        # ============ 窗口基础配置 ============
        self.title("🔬 污染物影响因素分析系统 v2.0")
        self.geometry("1100x750")
        self.minsize(900, 600)  # 最小窗口尺寸，防止界面被压缩

        # ============ CustomTkinter 全局外观设置 ============
        # set_appearance_mode: "Dark"=暗黑模式, "Light"=亮色模式, "System"=跟随系统
        ctk.set_appearance_mode("Dark")
        # set_default_color_theme: 使用内置蓝色主题作为基础(我们会自定义覆盖)
        ctk.set_default_color_theme("blue")

        # ============ 数据存储变量初始化 ============
        self.df = None  # 加载的DataFrame数据
        self.file_path = None  # 数据文件路径
        self.all_columns = []  # 数据所有列名列表
        self.feature_vars = {}  # 特征变量: {列名: {'selected': BoolVar, 'type': StringVar}}
        self.target_vars = {}  # 目标变量: {列名: BoolVar}
        self.category_vars = {}  # 分类标签: {列名: StringVar}
        self.model_vars = {}  # 模型启用状态: {模型名: BoolVar}

        # ============ 构建界面 ============
        self._build_ui()

    # ==================================================================
    #                        界面构建方法
    # ==================================================================

    def _build_ui(self):
        """构建完整UI界面：顶部标题 + 中部标签页"""

        # ---------- 顶部标题栏 ----------
        header_frame = ctk.CTkFrame(
            self,
            height=60,
            corner_radius=0,  # 顶部栏不需要圆角
            fg_color=COLORS["bg_medium"]
        )
        header_frame.pack(fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)  # 固定高度，不随内容变化

        # 应用标题
        ctk.CTkLabel(
            header_frame,
            text="🔬 新污染物影响因素相对重要性分析系统",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLORS["text_primary"]
        ).pack(side="left", padx=20, pady=10)

        # 右侧版本信息标签
        ctk.CTkLabel(
            header_frame,
            text="v2.0 | CustomTkinter",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"]
        ).pack(side="right", padx=20, pady=10)

        # ---------- 中部标签页容器 ----------
        # CTkTabview: CustomTkinter的标签页组件，自带圆角和现代风格
        self.tabview = ctk.CTkTabview(
            self,
            corner_radius=15,
            fg_color=COLORS["bg_dark"],
            segmented_button_fg_color=COLORS["tab_unselected"],
            segmented_button_selected_color=COLORS["accent"],
            segmented_button_selected_hover_color=COLORS["accent_hover"],
            segmented_button_unselected_color=COLORS["tab_unselected"],
            segmented_button_unselected_hover_color=COLORS["bg_light"],
            text_color=COLORS["text_primary"]
        )
        self.tabview.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        # 创建5个标签页
        self.tabview.add("📂 数据加载")
        self.tabview.add("📊 特征选择")
        self.tabview.add("🎯 目标变量")
        self.tabview.add("⚙️ 模型参数")
        self.tabview.add("🚀 运行分析")

        # 分别构建每个标签页的内容
        self._build_data_tab()
        self._build_feature_tab()
        self._build_target_tab()
        self._build_model_tab()
        self._build_run_tab()

    # ------------------------------------------------------------------
    # 标签页1: 数据加载
    # ------------------------------------------------------------------
    def _build_data_tab(self):
        """构建数据加载页面: 文件选择 + 数据预览"""
        tab = self.tabview.tab("📂 数据加载")

        # --- 文件选择区域 ---
        file_frame = ctk.CTkFrame(tab, corner_radius=12, fg_color=COLORS["card_bg"])
        file_frame.pack(fill="x", padx=15, pady=(10, 5))

        ctk.CTkLabel(
            file_frame,
            text="📁 选择Excel数据文件",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["text_primary"]
        ).pack(anchor="w", padx=15, pady=(12, 5))

        # 文件路径输入行
        input_row = ctk.CTkFrame(file_frame, fg_color="transparent")
        input_row.pack(fill="x", padx=15, pady=5)

        # 文件路径输入框
        self.file_entry = ctk.CTkEntry(
            input_row,
            placeholder_text="点击右侧按钮选择 .xlsx / .xls 文件...",
            height=38,
            corner_radius=10,
            border_color=COLORS["border"],
            fg_color=COLORS["input_bg"],
            text_color=COLORS["text_primary"],
            font=ctk.CTkFont(size=13)
        )
        self.file_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        # 浏览按钮
        ctk.CTkButton(
            input_row,
            text="浏览文件",
            width=100,
            height=38,
            corner_radius=10,
            fg_color=COLORS["bg_light"],
            hover_color=COLORS["accent"],
            font=ctk.CTkFont(size=13),
            command=self._browse_file
        ).pack(side="left", padx=(0, 5))

        # 加载数据按钮
        ctk.CTkButton(
            input_row,
            text="✅ 加载数据",
            width=120,
            height=38,
            corner_radius=10,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._load_data
        ).pack(side="left")

        # 底部间距
        ctk.CTkLabel(file_frame, text="", height=5).pack()

        # --- 数据预览区域 ---
        preview_frame = ctk.CTkFrame(tab, corner_radius=12, fg_color=COLORS["card_bg"])
        preview_frame.pack(fill="both", expand=True, padx=15, pady=5)

        ctk.CTkLabel(
            preview_frame,
            text="📋 数据预览",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["text_primary"]
        ).pack(anchor="w", padx=15, pady=(12, 5))

        # CTkTextbox: CustomTkinter的文本框组件，自带圆角和滚动条
        self.preview_textbox = ctk.CTkTextbox(
            preview_frame,
            corner_radius=10,
            fg_color=COLORS["input_bg"],
            text_color=COLORS["text_primary"],
            font=ctk.CTkFont(family="Consolas", size=12),
            wrap="none"  # 不自动换行，保持表格对齐
        )
        self.preview_textbox.pack(fill="both", expand=True, padx=15, pady=(5, 15))

    # ------------------------------------------------------------------
    # 标签页2: 特征选择
    # ------------------------------------------------------------------
    def _build_feature_tab(self):
        """构建特征选择页面: 勾选影响因素并指定类型"""
        tab = self.tabview.tab("📊 特征选择")

        # 提示标签
        ctk.CTkLabel(
            tab,
            text="✏️ 勾选作为影响因素的列，并为每个特征指定数据类型",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["text_primary"]
        ).pack(anchor="w", padx=20, pady=(10, 5))

        ctk.CTkLabel(
            tab,
            text="💡 numeric = 数值型(连续变量)  |  categorical = 分类型(离散变量，将自动编码)",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"]
        ).pack(anchor="w", padx=20, pady=(0, 8))

        # 全选/取消全选 按钮行
        btn_row = ctk.CTkFrame(tab, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(0, 5))

        ctk.CTkButton(
            btn_row, text="全选", width=80, height=30, corner_radius=8,
            fg_color=COLORS["bg_light"], hover_color=COLORS["accent"],
            font=ctk.CTkFont(size=12),
            command=lambda: self._toggle_all_checkboxes(self.feature_vars, True)
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btn_row, text="取消全选", width=80, height=30, corner_radius=8,
            fg_color=COLORS["bg_light"], hover_color=COLORS["accent"],
            font=ctk.CTkFont(size=12),
            command=lambda: self._toggle_all_checkboxes(self.feature_vars, False)
        ).pack(side="left")

        # CTkScrollableFrame: 可滚动的框架，当特征很多时自动出现滚动条
        self.feature_scroll = ctk.CTkScrollableFrame(
            tab,
            corner_radius=12,
            fg_color=COLORS["card_bg"],
            scrollbar_button_color=COLORS["scrollbar"],
            scrollbar_button_hover_color=COLORS["accent"]
        )
        self.feature_scroll.pack(fill="both", expand=True, padx=15, pady=5)

        # 表头
        header = ctk.CTkFrame(self.feature_scroll, fg_color=COLORS["bg_light"], corner_radius=8)
        header.pack(fill="x", padx=5, pady=(5, 8))
        ctk.CTkLabel(header, text="选择", width=60, font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(side="left", padx=10)
        ctk.CTkLabel(header, text="列名", width=300, font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=COLORS["text_primary"], anchor="w").pack(side="left", padx=10)
        ctk.CTkLabel(header, text="数据类型", width=150, font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(side="left", padx=10)

    # ------------------------------------------------------------------
    # 标签页3: 目标变量
    # ------------------------------------------------------------------
    def _build_target_tab(self):
        """构建目标变量页面: 勾选污染物列并指定分类"""
        tab = self.tabview.tab("🎯 目标变量")

        ctk.CTkLabel(
            tab,
            text="🎯 勾选作为目标变量(污染物)的列，并为每个污染物指定类别",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["text_primary"]
        ).pack(anchor="w", padx=20, pady=(10, 5))

        ctk.CTkLabel(
            tab,
            text="💡 类别可自定义输入(如: VOCs, PAHs, 重金属等)，留空则归为'未分类'",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"]
        ).pack(anchor="w", padx=20, pady=(0, 8))

        # 全选/取消全选
        btn_row = ctk.CTkFrame(tab, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(0, 5))

        ctk.CTkButton(
            btn_row, text="全选", width=80, height=30, corner_radius=8,
            fg_color=COLORS["bg_light"], hover_color=COLORS["accent"],
            font=ctk.CTkFont(size=12),
            command=lambda: self._toggle_all_target_checkboxes(True)
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btn_row, text="取消全选", width=80, height=30, corner_radius=8,
            fg_color=COLORS["bg_light"], hover_color=COLORS["accent"],
            font=ctk.CTkFont(size=12),
            command=lambda: self._toggle_all_target_checkboxes(False)
        ).pack(side="left")

        # 可滚动框架
        self.target_scroll = ctk.CTkScrollableFrame(
            tab,
            corner_radius=12,
            fg_color=COLORS["card_bg"],
            scrollbar_button_color=COLORS["scrollbar"],
            scrollbar_button_hover_color=COLORS["accent"]
        )
        self.target_scroll.pack(fill="both", expand=True, padx=15, pady=5)

        # 表头
        header = ctk.CTkFrame(self.target_scroll, fg_color=COLORS["bg_light"], corner_radius=8)
        header.pack(fill="x", padx=5, pady=(5, 8))
        ctk.CTkLabel(header, text="选择", width=60, font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(side="left", padx=10)
        ctk.CTkLabel(header, text="列名", width=300, font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=COLORS["text_primary"], anchor="w").pack(side="left", padx=10)
        ctk.CTkLabel(header, text="污染物类别", width=200, font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(side="left", padx=10)

    # ------------------------------------------------------------------
    # 标签页4: 模型参数
    # ------------------------------------------------------------------
    def _build_model_tab(self):
        """
        构建模型参数页面：模型选择 + 超参数设置 + 性能阈值

        【参数选择理由详细注释】
        所有模型的默认超参数基于以下原则选取:
        1. 对于小~中等数据集(几百到几千条), 过深的树容易过拟合
        2. 学习率较低(0.05~0.1)配合适当的树数量能获得更好的泛化能力
        3. 正则化参数选择中等值，平衡模型复杂度和拟合能力
        """
        tab = self.tabview.tab("⚙️ 模型参数")

        # 整体使用可滚动框架，参数较多时不会溢出
        scroll = ctk.CTkScrollableFrame(
            tab, corner_radius=12, fg_color="transparent",
            scrollbar_button_color=COLORS["scrollbar"],
            scrollbar_button_hover_color=COLORS["accent"]
        )
        scroll.pack(fill="both", expand=True, padx=5, pady=5)

        # ===== 模型选择区域 =====
        model_card = ctk.CTkFrame(scroll, corner_radius=12, fg_color=COLORS["card_bg"])
        model_card.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            model_card,
            text="🤖 选择要使用的模型",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["text_primary"]
        ).pack(anchor="w", padx=15, pady=(12, 8))

        # 模型列表: (内部名, 显示名, 是否可用, 模型说明)
        models = [
            ('RandomForest', '🌲 随机森林', True,
             '经典集成学习方法，对过拟合有较好鲁棒性，适合特征重要性分析'),
            ('AdaBoost', '🚀 AdaBoost', True,
             '自适应提升算法，通过组合多个弱学习器构建强学习器'),
            ('XGBoost', '⚡ XGBoost', XGBOOST_AVAILABLE,
             '极端梯度提升，高效且精度优秀，被广泛用于污染物预测研究'),
            ('LightGBM', '💡 LightGBM', LIGHTGBM_AVAILABLE,
             '微软开源，基于直方图的决策树算法，训练速度极快'),
            ('CatBoost', '🐱 CatBoost', CATBOOST_AVAILABLE,
             'Yandex开源，原生支持分类特征，无需预编码'),
            ('GAM', '📈 GAM广义加性模型', GAM_AVAILABLE,
             '半参数模型，可解释性强，能揭示非线性影响关系'),
        ]

        model_grid = ctk.CTkFrame(model_card, fg_color="transparent")
        model_grid.pack(fill="x", padx=15, pady=(0, 12))

        for i, (key, name, available, desc) in enumerate(models):
            var = ctk.BooleanVar(value=available)  # 默认启用已安装的模型
            self.model_vars[key] = var

            row = ctk.CTkFrame(model_grid, fg_color=COLORS["bg_medium"], corner_radius=8)
            row.pack(fill="x", pady=2)

            cb = ctk.CTkCheckBox(
                row, text=name, variable=var,
                font=ctk.CTkFont(size=13),
                text_color=COLORS["text_primary"] if available else COLORS["text_secondary"],
                fg_color=COLORS["checkbox_on"],
                hover_color=COLORS["accent_hover"],
                border_color=COLORS["border"],
                corner_radius=5
            )
            cb.pack(side="left", padx=12, pady=6)

            # 模型说明文字
            ctk.CTkLabel(
                row, text=f"  {desc}",
                font=ctk.CTkFont(size=11),
                text_color=COLORS["text_secondary"]
            ).pack(side="left", padx=5)

            # 未安装的模型禁用复选框
            if not available:
                cb.configure(state="disabled")
                ctk.CTkLabel(
                    row, text="[未安装]",
                    font=ctk.CTkFont(size=11),
                    text_color=COLORS["error"]
                ).pack(side="right", padx=12)

        # ===== 通用参数区域 =====
        param_card = ctk.CTkFrame(scroll, corner_radius=12, fg_color=COLORS["card_bg"])
        param_card.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            param_card,
            text="🔧 通用超参数",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["text_primary"]
        ).pack(anchor="w", padx=15, pady=(12, 8))

        params_grid = ctk.CTkFrame(param_card, fg_color="transparent")
        params_grid.pack(fill="x", padx=15, pady=(0, 12))

        # --- 测试集比例 ---
        # 【选择理由】0.2~0.3是回归任务常用的测试集比例
        # 数据量较小时用0.2保留更多训练数据；数据量充足时用0.3获得更稳定的评估
        self.test_size_var = ctk.DoubleVar(value=0.3)
        self._add_param_row(
            params_grid, "测试集比例",
            "用于模型评估的数据占比。小数据集建议0.2，中大数据集0.3",
            self.test_size_var, row=0, min_val=0.1, max_val=0.5, step=0.05
        )

        # --- 树数量 (n_estimators) ---
        # 【选择理由】100棵树是经典默认值，在计算成本和性能间取得平衡
        # 对于小数据集(<500条), 100棵通常已足够; 大数据集可增至200~500
        # 过多树数量会增加训练时间但收益递减
        self.n_estimators_var = ctk.IntVar(value=100)
        self._add_param_row(
            params_grid, "树数量 (n_estimators)",
            "集成模型中决策树的数量。100为默认值，小数据集可减至50，大数据集可增至200~500",
            self.n_estimators_var, row=1, min_val=50, max_val=500, step=50
        )

        # --- 最大树深度 (max_depth) ---
        # 【选择理由】max_depth=6 是防止过拟合的关键参数
        # 深度过大(>15)容易在小数据集上过拟合
        # 深度过小(<3)可能欠拟合，无法捕捉复杂交互效应
        # 6~10是环境数据分析中的推荐范围 (参考Nature Scientific Reports相关文献)
        self.max_depth_var = ctk.IntVar(value=6)
        self._add_param_row(
            params_grid, "最大深度 (max_depth)",
            "控制树的最大深度，防止过拟合。推荐范围4~10，小数据集建议4~6",
            self.max_depth_var, row=2, min_val=3, max_val=20, step=1
        )

        # --- 学习率 (learning_rate) ---
        # 【选择理由】0.05是Boosting模型常用的学习率
        # 较小的学习率(0.01~0.05)配合较多树通常效果更好
        # 原代码使用0.1偏大，小数据集容易导致模型波动
        # 0.05在精度和训练速度之间取得良好平衡
        self.learning_rate_var = ctk.DoubleVar(value=0.05)
        self._add_param_row(
            params_grid, "学习率 (learning_rate)",
            "Boosting模型的学习率。较小值(0.01~0.05)更稳定但需更多树，较大值(0.1~0.3)收敛快但易过拟合",
            self.learning_rate_var, row=3, min_val=0.01, max_val=0.3, step=0.01
        )

        # --- 最小分裂样本数 (min_samples_split) ---
        # 【选择理由】5是随机森林中合理的最小分裂阈值
        # 太小(2)会导致叶节点过于稀疏(过拟合)
        # 太大(20+)可能无法捕捉细微的影响模式
        self.min_samples_split_var = ctk.IntVar(value=5)
        self._add_param_row(
            params_grid, "最小分裂样本数",
            "内部节点分裂所需的最少样本数。增大可防止过拟合，推荐5~20",
            self.min_samples_split_var, row=4, min_val=2, max_val=30, step=1
        )

        # --- 最小叶子样本数 (min_samples_leaf) ---
        # 【选择理由】2是保守的最小叶子数
        # 对于小数据集，1可能导致过拟合(单样本叶子节点)
        # 2~5能确保叶节点有足够统计意义
        self.min_samples_leaf_var = ctk.IntVar(value=2)
        self._add_param_row(
            params_grid, "最小叶子样本数",
            "叶节点所需的最少样本数。增大可减少过拟合，推荐2~10",
            self.min_samples_leaf_var, row=5, min_val=1, max_val=20, step=1
        )

        # --- 子采样比例 (subsample) ---
        # 【选择理由】0.8是XGBoost/LightGBM的经典默认值
        # 低于1.0引入随机性，有助于防止过拟合(类似随机森林的bagging思想)
        # 0.7~0.9是环境数据分析文献中的推荐范围
        self.subsample_var = ctk.DoubleVar(value=0.8)
        self._add_param_row(
            params_grid, "子采样比例 (subsample)",
            "每棵树训练时使用的样本比例(仅XGBoost/LightGBM)。0.7~0.9可减少过拟合",
            self.subsample_var, row=6, min_val=0.5, max_val=1.0, step=0.05
        )

        # --- 列采样比例 (colsample_bytree) ---
        # 【选择理由】0.8表示每棵树随机使用80%的特征
        # 引入特征维度的随机性，减少特征之间的相关性对模型的影响
        # 当特征数较少(<10)时建议保持0.8~1.0
        self.colsample_var = ctk.DoubleVar(value=0.8)
        self._add_param_row(
            params_grid, "列采样比例 (colsample)",
            "每棵树使用的特征比例(仅XGBoost/LightGBM)。特征少时建议0.8~1.0",
            self.colsample_var, row=7, min_val=0.3, max_val=1.0, step=0.05
        )

        # ===== 性能阈值区域 =====
        threshold_card = ctk.CTkFrame(scroll, corner_radius=12, fg_color=COLORS["card_bg"])
        threshold_card.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            threshold_card,
            text="📏 良好模型判定阈值",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["text_primary"]
        ).pack(anchor="w", padx=15, pady=(12, 5))

        ctk.CTkLabel(
            threshold_card,
            text="同时满足以下所有条件的模型被认定为'良好模型'",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"]
        ).pack(anchor="w", padx=15, pady=(0, 8))

        thresh_grid = ctk.CTkFrame(threshold_card, fg_color="transparent")
        thresh_grid.pack(fill="x", padx=15, pady=(0, 15))

        # --- 最小R² ---
        # 【选择理由】R²>0.1是非常宽松的阈值
        # 环境数据噪声大，R²通常不会太高
        # 0.1仅排除完全无解释力的模型
        self.min_r2_var = ctk.DoubleVar(value=0.1)
        self._add_threshold_row(thresh_grid, "最小 R²", self.min_r2_var,
                                "决定系数。>0.5优秀，>0.3良好，>0.1可接受", 0)

        # --- 最小SCC ---
        # 【选择理由】SCC(Spearman相关系数)>0.3要求单调性关系
        # 0.3是中等相关的下限(Cohen 1988标准)
        # 对于环境数据，SCC>0.3已表明存在有意义的排序一致性
        self.min_scc_var = ctk.DoubleVar(value=0.3)
        self._add_threshold_row(thresh_grid, "最小 SCC (Spearman)", self.min_scc_var,
                                "Spearman等级相关。>0.5强相关，>0.3中等相关", 1)

        # --- 最小PCC ---
        # 【选择理由】PCC(Pearson相关系数)>0.3要求线性关系
        # 与SCC阈值一致，确保模型预测与真实值有中等以上的线性相关性
        self.min_pcc_var = ctk.DoubleVar(value=0.3)
        self._add_threshold_row(thresh_grid, "最小 PCC (Pearson)", self.min_pcc_var,
                                "Pearson线性相关。>0.5强相关，>0.3中等相关", 2)

    def _add_param_row(self, parent, label, tooltip, variable, row, min_val, max_val, step):
        """
        在参数面板中添加一行参数控件: 标签 + 输入框 + 说明

        Args:
            parent: 父容器
            label: 参数名称
            tooltip: 参数说明文字
            variable: tkinter变量(IntVar/DoubleVar)
            row: 行号
            min_val/max_val: 取值范围
            step: 步进值(仅用于显示参考)
        """
        frame = ctk.CTkFrame(parent, fg_color=COLORS["bg_medium"], corner_radius=8)
        frame.pack(fill="x", pady=2)

        # 参数名称
        ctk.CTkLabel(
            frame, text=f"  {label}:", width=220,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_primary"], anchor="w"
        ).pack(side="left", padx=(10, 5), pady=6)

        # 输入框
        entry = ctk.CTkEntry(
            frame, textvariable=variable, width=80, height=30,
            corner_radius=8,
            fg_color=COLORS["input_bg"],
            border_color=COLORS["border"],
            text_color=COLORS["text_primary"],
            font=ctk.CTkFont(size=12),
            justify="center"
        )
        entry.pack(side="left", padx=5)

        # 范围提示
        ctk.CTkLabel(
            frame, text=f"[{min_val} ~ {max_val}]",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["warning"], width=100
        ).pack(side="left", padx=5)

        # 说明文字
        ctk.CTkLabel(
            frame, text=tooltip,
            font=ctk.CTkFont(size=10),
            text_color=COLORS["text_secondary"],
            anchor="w"
        ).pack(side="left", padx=10, fill="x", expand=True)

    def _add_threshold_row(self, parent, label, variable, tooltip, row):
        """添加阈值参数行"""
        frame = ctk.CTkFrame(parent, fg_color=COLORS["bg_medium"], corner_radius=8)
        frame.pack(fill="x", pady=2)

        ctk.CTkLabel(
            frame, text=f"  {label}:", width=180,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_primary"], anchor="w"
        ).pack(side="left", padx=(10, 5), pady=6)

        entry = ctk.CTkEntry(
            frame, textvariable=variable, width=80, height=30,
            corner_radius=8,
            fg_color=COLORS["input_bg"],
            border_color=COLORS["border"],
            text_color=COLORS["text_primary"],
            font=ctk.CTkFont(size=12),
            justify="center"
        )
        entry.pack(side="left", padx=5)

        ctk.CTkLabel(
            frame, text=tooltip,
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"], anchor="w"
        ).pack(side="left", padx=15, fill="x", expand=True)

    # ------------------------------------------------------------------
    # 标签页5: 运行分析
    # ------------------------------------------------------------------
    def _build_run_tab(self):
        """构建运行分析页面: 输出目录 + 运行按钮 + 进度条 + 日志"""
        tab = self.tabview.tab("🚀 运行分析")

        # --- 输出目录 ---
        dir_card = ctk.CTkFrame(tab, corner_radius=12, fg_color=COLORS["card_bg"])
        dir_card.pack(fill="x", padx=15, pady=(10, 5))

        dir_row = ctk.CTkFrame(dir_card, fg_color="transparent")
        dir_row.pack(fill="x", padx=15, pady=12)

        ctk.CTkLabel(
            dir_row, text="📁 输出目录:",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["text_primary"]
        ).pack(side="left", padx=(0, 10))

        self.output_dir_var = ctk.StringVar(value="./分析结果")
        ctk.CTkEntry(
            dir_row, textvariable=self.output_dir_var,
            height=35, corner_radius=10,
            fg_color=COLORS["input_bg"],
            border_color=COLORS["border"],
            text_color=COLORS["text_primary"],
            font=ctk.CTkFont(size=12)
        ).pack(side="left", fill="x", expand=True, padx=(0, 10))

        ctk.CTkButton(
            dir_row, text="选择目录", width=100, height=35, corner_radius=10,
            fg_color=COLORS["bg_light"], hover_color=COLORS["accent"],
            command=self._browse_output_dir
        ).pack(side="left")

        # --- 运行按钮 (居中大按钮) ---
        btn_frame = ctk.CTkFrame(tab, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=10)

        self.run_button = ctk.CTkButton(
            btn_frame,
            text="🚀  开 始 分 析",
            height=50,
            corner_radius=15,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            font=ctk.CTkFont(size=18, weight="bold"),
            command=self._run_analysis
        )
        self.run_button.pack(fill="x", padx=100)

        # --- 进度条 ---
        progress_card = ctk.CTkFrame(tab, corner_radius=12, fg_color=COLORS["card_bg"])
        progress_card.pack(fill="x", padx=15, pady=5)

        progress_inner = ctk.CTkFrame(progress_card, fg_color="transparent")
        progress_inner.pack(fill="x", padx=15, pady=10)

        self.progress_bar = ctk.CTkProgressBar(
            progress_inner,
            height=18,
            corner_radius=10,
            fg_color=COLORS["progress_bg"],
            progress_color=COLORS["progress_fill"]
        )
        self.progress_bar.pack(fill="x", pady=(0, 5))
        self.progress_bar.set(0)  # 初始进度为0

        self.status_label = ctk.CTkLabel(
            progress_inner,
            text="⏳ 等待开始...",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_secondary"]
        )
        self.status_label.pack(anchor="w")

        # --- 日志输出 ---
        log_card = ctk.CTkFrame(tab, corner_radius=12, fg_color=COLORS["card_bg"])
        log_card.pack(fill="both", expand=True, padx=15, pady=(5, 10))

        ctk.CTkLabel(
            log_card,
            text="📝 运行日志",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["text_primary"]
        ).pack(anchor="w", padx=15, pady=(10, 5))

        self.log_textbox = ctk.CTkTextbox(
            log_card,
            corner_radius=10,
            fg_color=COLORS["input_bg"],
            text_color=COLORS["success"],
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="word"
        )
        self.log_textbox.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    # ==================================================================
    #                        事件处理方法
    # ==================================================================

    def _browse_file(self):
        """打开文件选择对话框"""
        filename = filedialog.askopenfilename(
            title="选择Excel数据文件",
            filetypes=[("Excel文件", "*.xlsx *.xls"), ("所有文件", "*.*")]
        )
        if filename:
            self.file_entry.delete(0, "end")
            self.file_entry.insert(0, filename)
            self.file_path = filename

    def _browse_output_dir(self):
        """打开输出目录选择对话框"""
        dirname = filedialog.askdirectory(title="选择输出目录")
        if dirname:
            self.output_dir_var.set(dirname)

    def _load_data(self):
        """加载Excel数据并更新特征/目标选择界面"""
        path = self.file_entry.get().strip()
        if not path:
            self._show_message("❌ 错误", "请先选择数据文件！", "error")
            return

        self.file_path = path

        try:
            # 读取Excel数据，header=0表示第一行为表头
            self.df = pd.read_excel(self.file_path, header=0)
            self.all_columns = self.df.columns.tolist()

            # 更新数据预览
            preview = f"✅ 数据加载成功!\n\n"
            preview += f"📊 行数: {len(self.df)}    列数: {len(self.all_columns)}\n\n"
            preview += "📋 列名列表:\n"
            for i, col in enumerate(self.all_columns, 1):
                # 显示列的数据类型和非空值数量
                dtype = str(self.df[col].dtype)
                non_null = self.df[col].notna().sum()
                preview += f"  {i:3d}. {col:30s}  [{dtype}]  非空: {non_null}/{len(self.df)}\n"
            preview += f"\n{'=' * 80}\n前5行数据预览:\n{'=' * 80}\n"
            preview += self.df.head().to_string(max_colwidth=20)

            self.preview_textbox.delete("1.0", "end")
            self.preview_textbox.insert("1.0", preview)

            # 更新特征选择和目标选择界面
            self._populate_feature_list()
            self._populate_target_list()

            self._show_message(
                "✅ 加载成功",
                f"数据已成功加载！\n\n共 {len(self.df)} 行, {len(self.all_columns)} 列\n\n"
                f"请切换到'📊 特征选择'和'🎯 目标变量'页面进行设置",
                "info"
            )

        except Exception as e:
            self._show_message("❌ 加载失败", f"读取数据时出错:\n{str(e)}", "error")

    def _populate_feature_list(self):
        """填充特征选择列表"""
        # 清空现有内容(保留表头)
        children = self.feature_scroll.winfo_children()
        for child in children[1:]:  # 跳过表头
            child.destroy()
        self.feature_vars.clear()

        for i, col in enumerate(self.all_columns):
            # 为每列创建一行
            row_frame = ctk.CTkFrame(
                self.feature_scroll,
                fg_color=COLORS["bg_medium"] if i % 2 == 0 else COLORS["card_bg"],
                corner_radius=6
            )
            row_frame.pack(fill="x", padx=5, pady=1)

            # 复选框
            selected_var = ctk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(
                row_frame, text="", variable=selected_var, width=40,
                fg_color=COLORS["checkbox_on"],
                hover_color=COLORS["accent_hover"],
                border_color=COLORS["border"],
                corner_radius=5
            )
            cb.pack(side="left", padx=(15, 5), pady=4)

            # 列名
            ctk.CTkLabel(
                row_frame, text=col, width=300,
                font=ctk.CTkFont(size=12),
                text_color=COLORS["text_primary"], anchor="w"
            ).pack(side="left", padx=10)

            # 类型选择下拉框
            type_var = ctk.StringVar(value="numeric")
            type_menu = ctk.CTkOptionMenu(
                row_frame,
                variable=type_var,
                values=["numeric", "categorical"],
                width=130, height=28,
                corner_radius=8,
                fg_color=COLORS["bg_light"],
                button_color=COLORS["bg_light"],
                button_hover_color=COLORS["accent"],
                dropdown_fg_color=COLORS["bg_medium"],
                dropdown_hover_color=COLORS["accent"],
                dropdown_text_color=COLORS["text_primary"],
                font=ctk.CTkFont(size=11)
            )
            type_menu.pack(side="left", padx=10)

            # 保存变量引用
            self.feature_vars[col] = {'selected': selected_var, 'type': type_var}

    def _populate_target_list(self):
        """填充目标变量选择列表"""
        children = self.target_scroll.winfo_children()
        for child in children[1:]:
            child.destroy()
        self.target_vars.clear()
        self.category_vars.clear()

        for i, col in enumerate(self.all_columns):
            row_frame = ctk.CTkFrame(
                self.target_scroll,
                fg_color=COLORS["bg_medium"] if i % 2 == 0 else COLORS["card_bg"],
                corner_radius=6
            )
            row_frame.pack(fill="x", padx=5, pady=1)

            # 复选框
            selected_var = ctk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(
                row_frame, text="", variable=selected_var, width=40,
                fg_color=COLORS["checkbox_on"],
                hover_color=COLORS["accent_hover"],
                border_color=COLORS["border"],
                corner_radius=5
            )
            cb.pack(side="left", padx=(15, 5), pady=4)

            # 列名
            ctk.CTkLabel(
                row_frame, text=col, width=300,
                font=ctk.CTkFont(size=12),
                text_color=COLORS["text_primary"], anchor="w"
            ).pack(side="left", padx=10)

            # 分类输入框
            cat_var = ctk.StringVar(value="")
            cat_entry = ctk.CTkEntry(
                row_frame, textvariable=cat_var,
                width=180, height=28, corner_radius=8,
                placeholder_text="输入类别(如VOCs)...",
                fg_color=COLORS["input_bg"],
                border_color=COLORS["border"],
                text_color=COLORS["text_primary"],
                font=ctk.CTkFont(size=11)
            )
            cat_entry.pack(side="left", padx=10)

            self.target_vars[col] = selected_var
            self.category_vars[col] = cat_var

    def _toggle_all_checkboxes(self, var_dict, state):
        """全选/取消全选特征变量"""
        for col, info in var_dict.items():
            info['selected'].set(state)

    def _toggle_all_target_checkboxes(self, state):
        """全选/取消全选目标变量"""
        for col, var in self.target_vars.items():
            var.set(state)

    def _log(self, message):
        """向日志文本框追加消息"""
        self.log_textbox.insert("end", message + "\n")
        self.log_textbox.see("end")

    def _update_status(self, message, progress=None):
        """更新进度条和状态文字"""
        self.status_label.configure(text=message)
        if progress is not None:
            self.progress_bar.set(progress / 100.0)  # CTkProgressBar范围是0~1

    def _show_message(self, title, message, msg_type="info"):
        """
        显示自定义弹窗消息

        使用CTkToplevel创建现代风格弹窗，避免原生tk messagebox的丑陋外观
        """
        dialog = ctk.CTkToplevel(self)
        dialog.title(title)
        dialog.geometry("450x220")
        dialog.resizable(False, False)
        dialog.transient(self)  # 设为主窗口的子窗口
        dialog.grab_set()  # 模态弹窗(阻止操作主窗口)

        # 弹窗内容
        content_frame = ctk.CTkFrame(dialog, corner_radius=0, fg_color=COLORS["bg_dark"])
        content_frame.pack(fill="both", expand=True)

        # 图标和标题
        icon_color = {
            "info": COLORS["success"],
            "error": COLORS["error"],
            "warning": COLORS["warning"]
        }.get(msg_type, COLORS["text_primary"])

        ctk.CTkLabel(
            content_frame, text=title,
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=icon_color
        ).pack(pady=(25, 10))

        # 消息内容
        ctk.CTkLabel(
            content_frame, text=message,
            font=ctk.CTkFont(size=13),
            text_color=COLORS["text_primary"],
            wraplength=380,
            justify="center"
        ).pack(pady=10, padx=20)

        # 确定按钮
        ctk.CTkButton(
            content_frame, text="确 定", width=120, height=35,
            corner_radius=10,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            font=ctk.CTkFont(size=13, weight="bold"),
            command=dialog.destroy
        ).pack(pady=(10, 20))

        # 居中显示弹窗
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 450) // 2
        y = self.winfo_y() + (self.winfo_height() - 220) // 2
        dialog.geometry(f"+{x}+{y}")

    # ==================================================================
    #                        核心分析逻辑
    # ==================================================================

    def _run_analysis(self):
        """启动分析(入口方法) - 验证输入后在新线程中执行"""

        # ---- 输入验证 ----
        if self.df is None:
            self._show_message("❌ 错误", "请先在'📂 数据加载'页面加载数据", "error")
            return

        selected_features = [
            (col, info['type'].get())
            for col, info in self.feature_vars.items()
            if info['selected'].get()
        ]
        if not selected_features:
            self._show_message("❌ 错误", "请在'📊 特征选择'页面至少选择一个影响因素", "error")
            return

        selected_targets = [col for col, var in self.target_vars.items() if var.get()]
        if not selected_targets:
            self._show_message("❌ 错误", "请在'🎯 目标变量'页面至少选择一个目标变量", "error")
            return

        # ---- 禁用运行按钮，防止重复点击 ----
        self.run_button.configure(state="disabled", text="⏳ 分析中...")
        self.log_textbox.delete("1.0", "end")  # 清空旧日志

        # ---- 在后台线程运行分析(避免UI卡死) ----
        thread = threading.Thread(
            target=self._analysis_worker,
            args=(selected_features, selected_targets),
            daemon=True  # 守护线程：主窗口关闭时自动终止
        )
        thread.start()

    def _analysis_worker(self, selected_features, selected_targets):
        """
        分析工作线程 - 在后台执行所有计算

        详细步骤:
        1. 数据预处理 (缺失值、编码)
        2. Spearman/Pearson 相关性分析
        3. 依次运行选中的机器学习模型
        4. 综合比较与汇总
        5. 导出Excel报告
        """
        try:
            self._log("=" * 70)
            self._log("🚀 开始分析...")
            self._log("=" * 70)
            self._update_status("🔄 准备数据...", 0)

            # ===== Step 1: 数据预处理 =====
            df_work = self.df.copy()

            feature_names = []  # 特征列名列表
            feature_names_cn = []  # 特征中文名(此处与列名相同)
            X_data = {}  # 处理后的特征数据字典

            for col, ftype in selected_features:
                feature_names.append(col)
                feature_names_cn.append(col)

                if ftype == 'numeric':
                    # 数值型特征: 转为数值，非数值项替换为NaN后填充0
                    X_data[col] = pd.to_numeric(df_work[col], errors='coerce').fillna(0)
                else:
                    # 分类型特征: 使用LabelEncoder将字符串编码为整数
                    # 【注意】LabelEncoder不保证编码的有序性，仅适用于树模型
                    le = LabelEncoder()
                    X_data[col] = le.fit_transform(df_work[col].astype(str))

            X = pd.DataFrame(X_data)

            # 删除含缺失值的行
            valid_idx = X.notna().all(axis=1)
            X = X[valid_idx].reset_index(drop=True)
            df_work = df_work[valid_idx].reset_index(drop=True)

            self._log(f"\n📊 数据概况:")
            self._log(f"   有效样本数: {len(X)}")
            self._log(f"   特征变量({len(feature_names)}个): {', '.join(feature_names_cn)}")
            self._log(f"   目标变量({len(selected_targets)}个)")

            # 获取分类映射
            category_mapping = {}
            for col in selected_targets:
                cat = self.category_vars.get(col, ctk.StringVar(value='')).get().strip()
                category_mapping[col] = cat if cat else '未分类'

            # 准备目标变量(全部转为数值型)
            target_cols = []
            for col in selected_targets:
                if col in df_work.columns:
                    df_work[col] = pd.to_numeric(df_work[col], errors='coerce').fillna(0)
                    target_cols.append(col)

            # ===== Step 2: Spearman相关性分析 =====
            self._log(f"\n{'=' * 50}")
            self._log(f"📈 Step 2: Spearman相关性分析...")
            self._update_status("📈 Spearman相关性分析...", 10)

            spearman_results = []
            for col in target_cols:
                y = df_work[col].values
                for i, feat in enumerate(feature_names):
                    r, p = spearmanr(X[feat], y)
                    spearman_results.append({
                        '目标变量': col,
                        '类别': category_mapping.get(col, '未分类'),
                        '影响因素': feature_names_cn[i],
                        'Spearman_r': round(r, 4),
                        'P值': p,
                        '显著性': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns',
                        '|r|': abs(r)
                    })

            df_spearman = pd.DataFrame(spearman_results)
            self._log("   ✅ Spearman分析完成")

            # ===== Step 3: 机器学习模型 =====
            all_model_results = {}
            active_models = [(k, v) for k, v in self.model_vars.items() if v.get()]
            model_count = len(active_models)
            current_model = 0

            # 读取用户设置的参数
            test_size = self.test_size_var.get()
            n_estimators = self.n_estimators_var.get()
            max_depth = self.max_depth_var.get()
            learning_rate = self.learning_rate_var.get()
            min_samples_split = self.min_samples_split_var.get()
            min_samples_leaf = self.min_samples_leaf_var.get()
            subsample = self.subsample_var.get()
            colsample = self.colsample_var.get()
            min_r2 = self.min_r2_var.get()
            min_scc = self.min_scc_var.get()
            min_pcc = self.min_pcc_var.get()

            # 记录使用的参数到日志
            self._log(f"\n{'=' * 50}")
            self._log(f"⚙️ 模型参数配置:")
            self._log(f"   测试集比例: {test_size}")
            self._log(f"   树数量: {n_estimators}")
            self._log(f"   最大深度: {max_depth}")
            self._log(f"   学习率: {learning_rate}")
            self._log(f"   子采样: {subsample}")
            self._log(f"   列采样: {colsample}")
            self._log(f"   最小分裂样本: {min_samples_split}")
            self._log(f"   最小叶子样本: {min_samples_leaf}")

            # ---------- 随机森林 ----------
            if self.model_vars.get('RandomForest', ctk.BooleanVar(value=False)).get():
                current_model += 1
                progress = 10 + (current_model / max(model_count, 1)) * 60
                self._log(f"\n🌲 [{current_model}/{model_count}] 随机森林...")
                self._update_status(f"🌲 随机森林 ({current_model}/{model_count})...", progress)

                rf_results = []
                for col in target_cols:
                    y = df_work[col].values
                    # 【log10(y+1)变换理由】
                    # 污染物浓度数据通常呈对数正态分布(右偏)
                    # log10变换可以:
                    #   1. 压缩极端值的影响
                    #   2. 使数据更接近正态分布
                    #   3. 提高回归模型的拟合效果
                    #   +1是为了避免log(0)的问题
                    y_log = np.log10(y + 1)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_log, test_size=test_size, random_state=42
                    )

                    # 【随机森林参数说明】
                    # n_estimators: 树的数量，越多越稳定但计算量更大
                    # max_depth: 限制树深度防止过拟合(关键参数)
                    # min_samples_split=5: 内部节点至少需要5个样本才能继续分裂
                    #   - 防止树在噪声数据上过度分裂
                    # min_samples_leaf=2: 叶子节点至少需要2个样本
                    #   - 防止出现单样本叶子(过拟合的典型表现)
                    # random_state=42: 固定随机种子保证结果可复现
                    # n_jobs=-1: 使用全部CPU核心并行计算加速
                    rf = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=42,
                        n_jobs=-1
                    )
                    rf.fit(X_train, y_train)

                    y_pred = rf.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    scc, _ = spearmanr(y_test, y_pred)
                    pcc, _ = pearsonr(y_test, y_pred)

                    # feature_importances_: 基于Gini不纯度的特征重要性
                    # 值越大表示该特征在分裂节点时的贡献越大
                    importances = rf.feature_importances_

                    result = {
                        '目标变量': col,
                        '类别': category_mapping.get(col, '未分类'),
                        'R²': round(r2, 4),
                        'RMSE': round(rmse, 4),
                        'SCC': round(scc, 4),
                        'PCC': round(pcc, 4)
                    }
                    for i, feat in enumerate(feature_names_cn):
                        result[f'{feat}_RI(%)'] = round(importances[i] * 100, 2)

                    rf_results.append(result)

                all_model_results['RandomForest'] = pd.DataFrame(rf_results)
                self._log("   ✅ 随机森林完成")

            # ---------- AdaBoost ----------
            if self.model_vars.get('AdaBoost', ctk.BooleanVar(value=False)).get():
                current_model += 1
                progress = 10 + (current_model / max(model_count, 1)) * 60
                self._log(f"\n🚀 [{current_model}/{model_count}] AdaBoost...")
                self._update_status(f"🚀 AdaBoost ({current_model}/{model_count})...", progress)

                ada_results = []
                for col in target_cols:
                    y = df_work[col].values
                    y_log = np.log10(y + 1)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_log, test_size=test_size, random_state=42
                    )

                    # 【AdaBoost参数说明】
                    # n_estimators: 弱学习器(决策桩)的数量
                    # learning_rate: 收缩系数，控制每个弱学习器的贡献权重
                    #   - 较小的learning_rate需要更多的n_estimators来补偿
                    #   - learning_rate与n_estimators存在权衡关系
                    # 默认基学习器是max_depth=3的决策树
                    # 【注意】AdaBoost的loss默认为'linear'(线性损失)
                    ada = AdaBoostRegressor(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        random_state=42
                    )
                    ada.fit(X_train, y_train)

                    y_pred = ada.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    scc, _ = spearmanr(y_test, y_pred)
                    pcc, _ = pearsonr(y_test, y_pred)

                    importances = ada.feature_importances_

                    result = {
                        '目标变量': col,
                        '类别': category_mapping.get(col, '未分类'),
                        'R²': round(r2, 4),
                        'SCC': round(scc, 4),
                        'PCC': round(pcc, 4)
                    }
                    for i, feat in enumerate(feature_names_cn):
                        result[f'{feat}_RI(%)'] = round(importances[i] * 100, 2)

                    ada_results.append(result)

                all_model_results['AdaBoost'] = pd.DataFrame(ada_results)
                self._log("   ✅ AdaBoost完成")

            # ---------- XGBoost ----------
            if self.model_vars.get('XGBoost', ctk.BooleanVar(value=False)).get() and XGBOOST_AVAILABLE:
                current_model += 1
                progress = 10 + (current_model / max(model_count, 1)) * 60
                self._log(f"\n⚡ [{current_model}/{model_count}] XGBoost...")
                self._update_status(f"⚡ XGBoost ({current_model}/{model_count})...", progress)

                xgb_results = []
                for col in target_cols:
                    y = df_work[col].values
                    y_log = np.log10(y + 1)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_log, test_size=test_size, random_state=42
                    )

                    # 【XGBoost参数说明】
                    # XGBoost是目前环境数据分析中应用最广泛的Boosting模型之一
                    # n_estimators: 提升轮次(树的数量)
                    # max_depth: 每棵树的最大深度
                    #   - XGBoost默认为6，与LightGBM不同(LightGBM默认-1即不限制)
                    #   - 深度6~8在大多数环境数据集上表现良好
                    # learning_rate (eta): 学习率/收缩系数
                    #   - 控制每棵新树的贡献度
                    #   - 0.05较为保守，配合100~200棵树效果好
                    # subsample: 行采样比例(随机选取80%的样本训练每棵树)
                    #   - 引入随机性，防止过拟合
                    # colsample_bytree: 列采样比例(随机选取80%的特征)
                    #   - 类似随机森林的特征bagging思想
                    # reg_alpha=0.1: L1正则化项
                    #   - 有助于稀疏化特征权重，提高泛化能力
                    # reg_lambda=1.0: L2正则化项
                    #   - 平滑权重，防止过拟合(XGBoost默认值)
                    # verbosity=0: 静默模式(不输出训练信息)
                    model = xgb.XGBRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        colsample_bytree=colsample,
                        reg_alpha=0.1,  # L1正则化，帮助特征选择
                        reg_lambda=1.0,  # L2正则化，防止过拟合
                        random_state=42,
                        n_jobs=-1,
                        verbosity=0
                    )
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    scc, _ = spearmanr(y_test, y_pred)
                    pcc, _ = pearsonr(y_test, y_pred)

                    importances = model.feature_importances_

                    result = {
                        '目标变量': col,
                        '类别': category_mapping.get(col, '未分类'),
                        'R²': round(r2, 4),
                        'SCC': round(scc, 4),
                        'PCC': round(pcc, 4)
                    }
                    for i, feat in enumerate(feature_names_cn):
                        result[f'{feat}_RI(%)'] = round(importances[i] * 100, 2)

                    xgb_results.append(result)

                all_model_results['XGBoost'] = pd.DataFrame(xgb_results)
                self._log("   ✅ XGBoost完成")

            # ---------- LightGBM ----------
            if self.model_vars.get('LightGBM', ctk.BooleanVar(value=False)).get() and LIGHTGBM_AVAILABLE:
                current_model += 1
                progress = 10 + (current_model / max(model_count, 1)) * 60
                self._log(f"\n💡 [{current_model}/{model_count}] LightGBM...")
                self._update_status(f"💡 LightGBM ({current_model}/{model_count})...", progress)

                lgb_results = []
                for col in target_cols:
                    y = df_work[col].values
                    y_log = np.log10(y + 1)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_log, test_size=test_size, random_state=42
                    )

                    # 【LightGBM参数说明】
                    # LightGBM使用基于直方图的决策树算法，训练速度快于XGBoost
                    # 特别适合大规模数据集和高维特征
                    # n_estimators: 提升轮次
                    # max_depth: 最大深度
                    #   - LightGBM是leaf-wise生长(与XGBoost的level-wise不同)
                    #   - max_depth用于限制树深度防止过拟合
                    # learning_rate: 学习率
                    # num_leaves=31: 叶子节点数(LightGBM特有参数)
                    #   - 控制树的复杂度，默认31
                    #   - 应小于 2^max_depth 以避免过拟合
                    #   - 对于小数据集，建议减小到15~31
                    # min_child_samples=20: 叶节点最少样本数
                    #   - 防止叶子节点样本过少导致过拟合
                    #   - 小数据集建议设为10~20
                    # subsample/colsample_bytree: 同XGBoost的含义
                    # reg_alpha/reg_lambda: L1/L2正则化
                    # verbosity=-1: 完全静默(不输出任何训练信息)
                    model = lgb.LGBMRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        num_leaves=31,
                        min_child_samples=20,
                        subsample=subsample,
                        colsample_bytree=colsample,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        random_state=42,
                        n_jobs=-1,
                        verbosity=-1
                    )
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    scc, _ = spearmanr(y_test, y_pred)
                    pcc, _ = pearsonr(y_test, y_pred)

                    # LightGBM的feature_importances_返回的是split次数(未归一化)
                    # 需要手动归一化为百分比
                    importances = model.feature_importances_ / max(model.feature_importances_.sum(), 1e-10)

                    result = {
                        '目标变量': col,
                        '类别': category_mapping.get(col, '未分类'),
                        'R²': round(r2, 4),
                        'SCC': round(scc, 4),
                        'PCC': round(pcc, 4)
                    }
                    for i, feat in enumerate(feature_names_cn):
                        result[f'{feat}_RI(%)'] = round(importances[i] * 100, 2)

                    lgb_results.append(result)

                all_model_results['LightGBM'] = pd.DataFrame(lgb_results)
                self._log("   ✅ LightGBM完成")

            # ---------- CatBoost ----------
            if self.model_vars.get('CatBoost', ctk.BooleanVar(value=False)).get() and CATBOOST_AVAILABLE:
                current_model += 1
                progress = 10 + (current_model / max(model_count, 1)) * 60
                self._log(f"\n🐱 [{current_model}/{model_count}] CatBoost...")
                self._update_status(f"🐱 CatBoost ({current_model}/{model_count})...", progress)

                cat_results = []
                # CatBoost原生支持分类特征，需要告知哪些列是分类型
                cat_features_idx = [i for i, (col, ftype) in enumerate(selected_features)
                                    if ftype == 'categorical']

                for col in target_cols:
                    y = df_work[col].values
                    y_log = np.log10(y + 1)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_log, test_size=test_size, random_state=42
                    )

                    # 【CatBoost参数说明】
                    # CatBoost由Yandex开发，最大特色是原生支持分类特征
                    # 不需要预先进行标签编码或独热编码
                    # iterations: 等同于n_estimators(提升轮次)
                    # depth: 树的深度(CatBoost使用对称树结构)
                    #   - 对称树使得CatBoost在推理时非常高效
                    #   - 推荐depth=4~8(对称树不需要很深就能有好效果)
                    # learning_rate: 学习率
                    # l2_leaf_reg=3.0: L2正则化系数
                    #   - 控制叶子节点的正则化强度
                    #   - 默认3.0，增大可防止过拟合
                    # cat_features: 分类特征的列索引
                    #   - CatBoost会自动使用有序目标编码处理分类特征
                    # verbose=False: 不输出训练进度
                    model = CatBoostRegressor(
                        iterations=n_estimators,
                        depth=min(max_depth, 10),  # CatBoost对称树深度一般不超过10
                        learning_rate=learning_rate,
                        l2_leaf_reg=3.0,
                        random_seed=42,
                        verbose=False,
                        cat_features=cat_features_idx if cat_features_idx else None
                    )
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    scc, _ = spearmanr(y_test, y_pred)
                    pcc, _ = pearsonr(y_test, y_pred)

                    # CatBoost的feature_importances_也需归一化
                    importances = model.feature_importances_ / max(model.feature_importances_.sum(), 1e-10)

                    result = {
                        '目标变量': col,
                        '类别': category_mapping.get(col, '未分类'),
                        'R²': round(r2, 4),
                        'SCC': round(scc, 4),
                        'PCC': round(pcc, 4)
                    }
                    for i, feat in enumerate(feature_names_cn):
                        result[f'{feat}_RI(%)'] = round(importances[i] * 100, 2)

                    cat_results.append(result)

                all_model_results['CatBoost'] = pd.DataFrame(cat_results)
                self._log("   ✅ CatBoost完成")

            # ---------- GAM (广义加性模型) ----------
            if self.model_vars.get('GAM', ctk.BooleanVar(value=False)).get() and GAM_AVAILABLE:
                current_model += 1
                progress = 10 + (current_model / max(model_count, 1)) * 60
                self._log(f"\n📈 [{current_model}/{model_count}] GAM分析...")
                self._update_status(f"📈 GAM ({current_model}/{model_count})...", progress)

                gam_results = []
                for col in target_cols:
                    y = df_work[col].values
                    y_log = np.log10(y + 1)
                    X_gam = X.values  # GAM需要numpy数组

                    try:
                        # 【GAM参数说明】
                        # GAM (Generalized Additive Model) 广义加性模型
                        # 核心思想: y = f1(x1) + f2(x2) + ... + fn(xn)
                        # 每个特征有独立的非线性函数f_i，模型可解释性极强
                        #
                        # s(i): 样条平滑项(用于数值型特征)
                        #   - n_splines=10: 样条基函数数量
                        #     较多的splines可以拟合更复杂的非线性关系
                        #     但过多可能过拟合，8~15是合理范围
                        #
                        # f(i): 因子项(用于分类型特征)
                        #   - 将分类变量作为离散因子处理
                        #
                        # gridsearch: 自动搜索最优的正则化参数lambda
                        #   - lambda控制平滑程度，越大越平滑(防过拟合)
                        gam_formula = None
                        for i, (col_name, ftype) in enumerate(selected_features):
                            if ftype == 'numeric':
                                term = s(i, n_splines=10)  # 数值型用样条平滑
                            else:
                                term = f(i)  # 分类型用因子项

                            gam_formula = term if gam_formula is None else gam_formula + term

                        gam = LinearGAM(gam_formula)
                        gam.gridsearch(X_gam, y_log, progress=False)

                        y_pred = gam.predict(X_gam)
                        r2 = r2_score(y_log, y_pred)
                        scc, _ = spearmanr(y_log, y_pred)
                        pcc, _ = pearsonr(y_log, y_pred)

                        # 计算GAM特征重要性:
                        # 通过partial_dependence的标准差来衡量
                        # 标准差越大 = 该特征对预测值的影响范围越大 = 越重要
                        importances = []
                        for i in range(len(selected_features)):
                            try:
                                XX = gam.generate_X_grid(term=i)
                                pdep = gam.partial_dependence(term=i, X=XX)
                                importances.append(np.std(pdep))
                            except Exception:
                                importances.append(0)

                        total = sum(importances) if sum(importances) > 0 else 1
                        ri_values = [(imp / total * 100) for imp in importances]

                        result = {
                            '目标变量': col,
                            '类别': category_mapping.get(col, '未分类'),
                            'R²': round(r2, 4),
                            'SCC': round(scc, 4),
                            'PCC': round(pcc, 4)
                        }
                        for i, feat in enumerate(feature_names_cn):
                            result[f'{feat}_RI(%)'] = round(ri_values[i], 2)

                        gam_results.append(result)

                    except Exception as e:
                        self._log(f"   ⚠️ GAM失败 ({col}): {str(e)}")
                        result = {
                            '目标变量': col,
                            '类别': category_mapping.get(col, '未分类'),
                            'R²': np.nan, 'SCC': np.nan, 'PCC': np.nan
                        }
                        for feat in feature_names_cn:
                            result[f'{feat}_RI(%)'] = np.nan
                        gam_results.append(result)

                all_model_results['GAM'] = pd.DataFrame(gam_results)
                self._log("   ✅ GAM完成")

            # ===== Step 4: 综合分析 =====
            self._log(f"\n{'=' * 50}")
            self._log("📊 Step 4: 综合分析...")
            self._update_status("📊 综合分析...", 80)

            # 汇总各模型的因素重要性(取所有目标变量的平均值)
            all_models_ri = {}
            for model_name, df_results in all_model_results.items():
                ri_values = []
                for feat in feature_names_cn:
                    col_name = f'{feat}_RI(%)'
                    if col_name in df_results.columns:
                        ri_values.append(df_results[col_name].mean())
                    else:
                        ri_values.append(0)
                all_models_ri[model_name] = ri_values

            # 创建综合比较表
            comparison_data = {'影响因素': feature_names_cn}
            for model_name, ri_values in all_models_ri.items():
                comparison_data[f'{model_name}_RI(%)'] = [round(v, 2) for v in ri_values]

            df_comparison = pd.DataFrame(comparison_data)

            # 计算平均RI和排名
            ri_cols = [c for c in df_comparison.columns if 'RI(%)' in c]
            df_comparison['平均RI(%)'] = df_comparison[ri_cols].mean(axis=1).round(2)
            df_comparison['排名'] = df_comparison['平均RI(%)'].rank(ascending=False).astype(int)
            df_comparison = df_comparison.sort_values('排名')

            self._log("   ✅ 综合比较表完成")

            # 模型性能汇总
            model_performance = []
            for model_name, df_results in all_model_results.items():
                df_good = df_results[
                    (df_results['SCC'] > min_scc) &
                    (df_results['PCC'] > min_pcc) &
                    (df_results['R²'] > min_r2)
                    ]
                n_total = len(df_results)
                n_good = len(df_good)
                model_performance.append({
                    '模型': model_name,
                    '目标变量数': n_total,
                    '良好模型数': n_good,
                    '良好比例(%)': round(n_good / max(n_total, 1) * 100, 1),
                    '平均R²': round(df_results['R²'].mean(), 4),
                    '平均SCC': round(df_results['SCC'].mean(), 4),
                    '平均PCC': round(df_results['PCC'].mean(), 4)
                })

            df_performance = pd.DataFrame(model_performance)
            df_performance = df_performance.sort_values('平均SCC', ascending=False)

            self._log("   ✅ 模型性能汇总完成")

            # 按类别汇总
            category_ri_summary = []
            unique_categories = set(category_mapping.values())

            for cat in sorted(unique_categories):
                row = {'类别': cat}
                for model_name, df_results in all_model_results.items():
                    cat_data = df_results[df_results['类别'] == cat]
                    if len(cat_data) > 0:
                        for feat in feature_names_cn:
                            col_name = f'{feat}_RI(%)'
                            if col_name in cat_data.columns:
                                row[f'{model_name}_{feat}_RI均值'] = round(cat_data[col_name].mean(), 2)
                category_ri_summary.append(row)

            df_category_ri = pd.DataFrame(category_ri_summary)

            # ===== Step 5: 导出Excel报告 =====
            self._log(f"\n{'=' * 50}")
            self._log("💾 Step 5: 导出Excel报告...")
            self._update_status("💾 导出结果...", 90)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.output_dir_var.get()
            if not output_dir:
                output_dir = "./分析结果"
            output_dir = f"{output_dir}_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(output_dir, "分析结果.xlsx")

            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Sheet 1-3: Spearman相关性
                df_spearman.to_excel(writer, sheet_name="Spearman相关性_全部", index=False)

                spearman_pivot = df_spearman.pivot(
                    index='目标变量', columns='影响因素', values='Spearman_r')
                spearman_pivot.to_excel(writer, sheet_name="Spearman_矩阵")

                spearman_sig = df_spearman.pivot(
                    index='目标变量', columns='影响因素', values='显著性')
                spearman_sig.to_excel(writer, sheet_name="Spearman_显著性")

                # 各模型详细结果
                for model_name, df_results in all_model_results.items():
                    sheet_name = f"{model_name}_全部"[:31]  # Excel sheet名最多31字符
                    df_results.to_excel(writer, sheet_name=sheet_name, index=False)

                    # 导出良好模型子集
                    df_good = df_results[
                        (df_results['SCC'] > min_scc) &
                        (df_results['PCC'] > min_pcc) &
                        (df_results['R²'] > min_r2)
                        ]
                    if len(df_good) > 0:
                        good_name = f"{model_name}_良好"[:31]
                        df_good.to_excel(writer, sheet_name=good_name, index=False)

                # 综合比较和汇总
                df_comparison.to_excel(writer, sheet_name="因素重要性_综合比较", index=False)
                df_performance.to_excel(writer, sheet_name="模型性能汇总", index=False)

                if len(df_category_ri) > 0:
                    df_category_ri.to_excel(writer, sheet_name="各类别_因素重要性", index=False)

            self._log(f"\n💾 结果已保存: {output_file}")

            # ===== 输出总结 =====
            self._log(f"\n{'=' * 70}")
            self._log("🎉 分析完成!")
            self._log(f"{'=' * 70}")

            self._log("\n📊 【模型性能概况】")
            for _, row in df_performance.iterrows():
                self._log(
                    f"   {row['模型']:15s} | R²={row['平均R²']:.4f} | "
                    f"SCC={row['平均SCC']:.4f} | PCC={row['平均PCC']:.4f} | "
                    f"良好率={row['良好比例(%)']:.1f}%"
                )

            self._log("\n🏆 【因素重要性排序 (多模型平均)】")
            for _, row in df_comparison.iterrows():
                bar = "█" * int(row['平均RI(%)'] / 2)  # 简易柱状图
                self._log(
                    f"   {int(row['排名']):2d}. {row['影响因素']:20s} "
                    f"| RI = {row['平均RI(%)']:6.1f}% {bar}"
                )

            self._update_status("🎉 分析完成!", 100)

            # 通过after在主线程中弹出完成提示
            self.after(100, lambda: self._show_message(
                "🎉 分析完成",
                f"所有分析已完成!\n\n"
                f"📁 结果文件:\n{output_file}\n\n"
                f"共分析 {len(target_cols)} 个目标变量\n"
                f"使用 {len(all_model_results)} 个模型",
                "info"
            ))

        except Exception as e:
            self._log(f"\n❌ 错误: {str(e)}")
            self._log(traceback.format_exc())
            self._update_status("❌ 分析出错", 0)
            self.after(100, lambda: self._show_message(
                "❌ 分析出错", f"分析过程中发生错误:\n\n{str(e)}", "error"
            ))

        finally:
            # 恢复运行按钮
            self.after(100, lambda: self.run_button.configure(
                state="normal", text="🚀  开 始 分 析"
            ))


# ==================================================================
#                        程序入口
# ==================================================================
def main():
    """程序入口函数"""
    app = PollutantAnalysisApp()
    app.mainloop()


if __name__ == "__main__":
    main()