from __future__ import annotations

import logging
import os
import sys
import warnings
from typing import Any
from pathlib import Path

# ============================================================================
# 配置：防止 joblib 在中文路径下的 Unicode 编码问题
# 使用 platformdirs 获取平台特定的缓存目录（ASCII 纯英文路径）
# ============================================================================
def _setup_joblib_cache():
    """设置 joblib 的缓存目录为 ASCII 纯英文路径"""
    try:
        from platformdirs import user_cache_dir
        cache_dir = Path(user_cache_dir("PollutantAnalysis", ensure_exists=True))
    except Exception:
        # 降级方案：使用系统 TEMP 环境变量
        cache_dir = Path(os.environ.get("TEMP", "C:\\temp"))
    
    joblib_cache = cache_dir / "joblib"
    joblib_cache.mkdir(parents=True, exist_ok=True)
    os.environ["JOBLIB_TEMP_FOLDER"] = str(joblib_cache)
    return joblib_cache

_setup_joblib_cache()

from constants import configure_matplotlib, ensure_app_log_dir, get_app_log_path
from event_bus import EventBus
from state import AppState


if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

ensure_app_log_dir()
APP_LOG_PATH = get_app_log_path()

log_handlers: list[logging.Handler] = [
    logging.FileHandler(APP_LOG_PATH, encoding="utf-8"),
]
if not getattr(sys, "frozen", False):
    log_handlers.insert(0, logging.StreamHandler(sys.stdout))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=log_handlers,
)
logger = logging.getLogger("PollutantApp")


def _die_missing_deps(message: str) -> None:
    try:
        logger.error(message)
    except Exception:
        pass

    if getattr(sys, "frozen", False):
        raise RuntimeError(message)

    print(message, file=sys.stderr)
    input("\n按回车键退出...")
    sys.exit(1)


try:
    import customtkinter as ctk
except ModuleNotFoundError:
    _die_missing_deps(
        "未找到模块 customtkinter。\n\n"
        "请在当前 Python 环境中安装依赖，例如：\n"
        "  pip install -r requirements.txt\n\n"
        "或仅安装界面依赖：\n"
        "  pip install customtkinter"
    )

try:
    import matplotlib

    if getattr(sys, "frozen", False):
        matplotlib.use("Agg")
    else:
        matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    _die_missing_deps(f"缺少依赖: {exc.name}\n\n请执行: pip install -r requirements.txt")

configure_matplotlib(plt)
warnings.filterwarnings("ignore")

from modules.theme import C, FONTS, APP_NAME, VERSION, apply_theme, make_btn_secondary, make_tabview, show_message


DEFAULT_WINDOW_SIZE = "1300x820"
MIN_WINDOW_SIZE = (1040, 720)

TAB_LABELS = {
    "data_load": "📂 数据",
    "features": "📋 特征",
    "targets": "🎯 目标",
    "preprocess": "🧹 预处",
    "statistics": "📊 统计",
    "model_params": "⚙️ 模型",
    "analysis": "🚀 分析",
    "cv": "🔄 CV",
    "visualization": "📈 图表",
    "simulation": "🔮 模拟",
}


class _AppStateProxy:
    """Expose app state fields while preserving Tk's callable state() API."""

    def __init__(self, app: "PollutantAnalysisApp", state_store: AppState) -> None:
        object.__setattr__(self, "_app", app)
        object.__setattr__(self, "_state_store", state_store)

    def __call__(self, *args: Any) -> Any:
        return ctk.CTk.state(self._app, *args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._state_store, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._state_store, name, value)


class PollutantAnalysisApp(ctk.CTk):
    def _get_state_attr(self, attr_name: str) -> Any:
        return getattr(self._state_store, attr_name)

    def _set_state_attr(self, attr_name: str, value: Any) -> None:
        setattr(self._state_store, attr_name, value)

    @property
    def app_state(self) -> AppState:
        return self._state_store

    @property
    def df(self) -> Any:
        return self._get_state_attr("df")

    @df.setter
    def df(self, value: Any) -> None:
        self._set_state_attr("df", value)

    @property
    def df_backup(self) -> Any:
        return self._get_state_attr("df_backup")

    @df_backup.setter
    def df_backup(self, value: Any) -> None:
        self._set_state_attr("df_backup", value)

    @property
    def file_path(self) -> Any:
        return self._get_state_attr("file_path")

    @file_path.setter
    def file_path(self, value: Any) -> None:
        self._set_state_attr("file_path", value)

    @property
    def all_columns(self) -> Any:
        return self._get_state_attr("all_columns")

    @all_columns.setter
    def all_columns(self, value: Any) -> None:
        self._set_state_attr("all_columns", value)

    @property
    def analysis_results(self) -> Any:
        return self._get_state_attr("analysis_results")

    @analysis_results.setter
    def analysis_results(self, value: Any) -> None:
        self._set_state_attr("analysis_results", value)

    @property
    def prediction_cache(self) -> Any:
        return self._get_state_attr("prediction_cache")

    @prediction_cache.setter
    def prediction_cache(self, value: Any) -> None:
        self._set_state_attr("prediction_cache", value)

    @property
    def model_cache(self) -> Any:
        return self._get_state_attr("model_cache")

    @model_cache.setter
    def model_cache(self, value: Any) -> None:
        self._set_state_attr("model_cache", value)

    @property
    def X_cache(self) -> Any:
        return self._get_state_attr("X_cache")

    @X_cache.setter
    def X_cache(self, value: Any) -> None:
        self._set_state_attr("X_cache", value)

    @property
    def cv_results(self) -> Any:
        return self._get_state_attr("cv_results")

    @cv_results.setter
    def cv_results(self, value: Any) -> None:
        self._set_state_attr("cv_results", value)

    @property
    def cv_fold_df(self) -> Any:
        return self._get_state_attr("cv_fold_df")

    @cv_fold_df.setter
    def cv_fold_df(self, value: Any) -> None:
        self._set_state_attr("cv_fold_df", value)

    @property
    def cv_detail_df(self) -> Any:
        return self._get_state_attr("cv_detail_df")

    @cv_detail_df.setter
    def cv_detail_df(self, value: Any) -> None:
        self._set_state_attr("cv_detail_df", value)

    @property
    def cv_summary_df(self) -> Any:
        return self._get_state_attr("cv_summary_df")

    @cv_summary_df.setter
    def cv_summary_df(self, value: Any) -> None:
        self._set_state_attr("cv_summary_df", value)

    @property
    def spearman_results_df(self) -> Any:
        return self._get_state_attr("spearman_results_df")

    @spearman_results_df.setter
    def spearman_results_df(self, value: Any) -> None:
        self._set_state_attr("spearman_results_df", value)

    @property
    def comparison_df(self) -> Any:
        return self._get_state_attr("comparison_df")

    @comparison_df.setter
    def comparison_df(self, value: Any) -> None:
        self._set_state_attr("comparison_df", value)

    @property
    def performance_df(self) -> Any:
        return self._get_state_attr("performance_df")

    @performance_df.setter
    def performance_df(self, value: Any) -> None:
        self._set_state_attr("performance_df", value)

    @property
    def label_encoders(self) -> Any:
        return self._get_state_attr("label_encoders")

    @label_encoders.setter
    def label_encoders(self, value: Any) -> None:
        self._set_state_attr("label_encoders", value)

    @property
    def _log_transformed(self) -> Any:
        return self._get_state_attr("log_transformed")

    @_log_transformed.setter
    def _log_transformed(self, value: Any) -> None:
        self._set_state_attr("log_transformed", value)

    @property
    def _preprocessing_applied(self) -> Any:
        return self._get_state_attr("preprocessing_applied")

    @_preprocessing_applied.setter
    def _preprocessing_applied(self, value: Any) -> None:
        self._set_state_attr("preprocessing_applied", value)

    def __init__(self) -> None:
        object.__setattr__(self, "_state_store", AppState())
        super().__init__()
        self.state = _AppStateProxy(self, self._state_store)
        self.event_bus = EventBus()
        self.recent_files: list[str] = []
        self.user_prefs: dict[str, Any] = {"theme": "Dark", "scaling": 100, "font_size": 13}

        self.feature_vars: dict[str, dict[str, Any]] = {}
        self.target_vars: dict[str, Any] = {}
        self.category_vars: dict[str, Any] = {}
        self.model_vars: dict[str, Any] = {}
        self.tabs: dict[str, Any] = {}

        self.header_frame: ctk.CTkFrame | None = None
        self.header_separator: ctk.CTkFrame | None = None
        self.flow_frame: ctk.CTkFrame | None = None
        self.flow_label: ctk.CTkLabel | None = None
        self.flow_hint_label: ctk.CTkLabel | None = None
        self.flow_action_btn: ctk.CTkButton | None = None
        self.menu_bar: Any = None
        self.status_bar: Any = None
        self.tabview: Any = None

        self.title(f"🔬 {APP_NAME} v{VERSION}")
        self.geometry(DEFAULT_WINDOW_SIZE)
        self.minsize(*MIN_WINDOW_SIZE)

        ctk.set_default_color_theme("blue")
        from modules.theme import _read_saved_theme

        initial_theme = _read_saved_theme()
        ctk.set_appearance_mode("Light" if initial_theme == "Light" else "Dark")
        self.configure(fg_color=C["bg_base"])

        self._on_loaded()

    def _on_loaded(self) -> None:
        try:
            self._build_header()
            self._build_flow_banner()
            self._build_tabs()

            from modules.managers import ConfigManager, ExportManager, StatusBar

            self.config_manager = ConfigManager(self)
            self.export_manager = ExportManager(self)
            self.status_bar = StatusBar(self, self)
            self.status_bar.pack(fill="x", side="bottom")
            self.refresh_navigation_state()
        except Exception as exc:
            logger.exception("Failed to initialize UI")
            import tkinter.messagebox

            tkinter.messagebox.showerror("启动错误", f"界面初始化失败\n{exc}")

    def _build_header(self) -> None:
        from modules.dialogs import (
            show_about,
            show_changelog,
            show_param_help,
            show_preferences,
            show_user_guide,
        )
        from modules.managers import CustomMenuBar

        self.header_frame = ctk.CTkFrame(self, height=50, corner_radius=0, fg_color=C["bg_primary"])
        self.header_frame.pack(fill="x")
        self.header_frame.pack_propagate(False)

        self.menu_bar = CustomMenuBar(self.header_frame, self)
        self.menu_bar.pack(side="left", fill="y", padx=(6, 0))

        self.menu_bar.add_menu(
            "文件",
            [
                ("📂  打开数据文件", self._cmd_open_file),
                None,
                ("💾  保存分析配置", lambda: self.config_manager.save_config()),
                ("📥  加载分析配置", lambda: self.config_manager.load_config()),
                None,
                ("📊  导出 CSV", lambda: self.export_manager.export_csv()),
                ("🖨  导出 PDF 报告", lambda: self.export_manager.export_pdf()),
                ("📝  导出日志", lambda: self.export_manager.export_log()),
                None,
                ("🚪  退出", self.on_closing),
            ],
        )
        self.menu_bar.add_menu("编辑", [("⚙️  偏好设置", lambda: show_preferences(self))])
        self.menu_bar.add_menu(
            "工具",
            [
                ("🧹  数据预处理", lambda: self.navigate_to_tab("preprocess")),
                ("📊  数据统计", lambda: self.navigate_to_tab("statistics")),
                ("📈  数据可视化", lambda: self.navigate_to_tab("visualization")),
                ("🔄  交叉验证", lambda: self.navigate_to_tab("cv")),
                ("🔮  情景模拟", lambda: self.navigate_to_tab("simulation")),
            ],
        )
        self.menu_bar.add_menu(
            "帮助",
            [
                ("📖  使用说明", lambda: show_user_guide(self)),
                ("📘  参数说明", lambda: show_param_help(self)),
                None,
                ("📝  更新日志", lambda: show_changelog(self)),
                ("ℹ️  关于", lambda: show_about(self)),
            ],
        )

        ctk.CTkLabel(
            self.header_frame,
            text=f"🔬  {APP_NAME}",
            font=FONTS["h1"](),
            text_color=C["text_primary"],
        ).pack(side="left", padx=30)
        ctk.CTkLabel(
            self.header_frame,
            text=f"v{VERSION}",
            font=FONTS["small"](),
            text_color=C["text_muted"],
        ).pack(side="right", padx=20)

        self.header_separator = ctk.CTkFrame(self, height=2, corner_radius=0, fg_color=C["accent_muted"])
        self.header_separator.pack(fill="x")

    def _build_flow_banner(self) -> None:
        self.flow_frame = ctk.CTkFrame(self, height=54, corner_radius=0, fg_color=C["bg_primary"])
        self.flow_frame.pack(fill="x", pady=(0, 2))
        self.flow_frame.pack_propagate(False)

        text_frame = ctk.CTkFrame(self.flow_frame, fg_color="transparent")
        text_frame.pack(side="left", fill="both", expand=True, padx=16, pady=8)

        self.flow_label = ctk.CTkLabel(
            text_frame,
            text="流程引导",
            font=FONTS["body_bold"](),
            text_color=C["text_primary"],
            anchor="w",
        )
        self.flow_label.pack(fill="x")

        self.flow_hint_label = ctk.CTkLabel(
            text_frame,
            text="请先导入数据，其他步骤会随着进度逐步解锁。",
            font=FONTS["small"](),
            text_color=C["text_secondary"],
            anchor="w",
        )
        self.flow_hint_label.pack(fill="x")

        self.flow_action_btn = make_btn_secondary(
            self.flow_frame,
            text="前往下一步",
            width=120,
            command=lambda: None,
        )
        self.flow_action_btn.pack(side="right", padx=16, pady=9)

    def _build_tabs(self) -> None:
        from modules.tab_analysis import AnalysisTab
        from modules.tab_cv import CrossValidationTab
        from modules.tab_data_load import DataLoadTab
        from modules.tab_features import FeaturesTab
        from modules.tab_model_params import ModelParamsTab
        from modules.tab_preprocess import PreprocessTab
        from modules.tab_simulation import SimulationTab
        from modules.tab_statistics import StatisticsTab
        from modules.tab_targets import TargetsTab
        from modules.tab_visualization import VisualizationTab

        self.tabview = make_tabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=(6, 0))

        for label in TAB_LABELS.values():
            self.tabview.add(label)

        self.tabs = {
            "data_load": DataLoadTab(self.tabview.tab(TAB_LABELS["data_load"]), self),
            "features": FeaturesTab(self.tabview.tab(TAB_LABELS["features"]), self),
            "targets": TargetsTab(self.tabview.tab(TAB_LABELS["targets"]), self),
            "preprocess": PreprocessTab(self.tabview.tab(TAB_LABELS["preprocess"]), self),
            "statistics": StatisticsTab(self.tabview.tab(TAB_LABELS["statistics"]), self),
            "model_params": ModelParamsTab(self.tabview.tab(TAB_LABELS["model_params"]), self),
            "analysis": AnalysisTab(self.tabview.tab(TAB_LABELS["analysis"]), self),
            "cv": CrossValidationTab(self.tabview.tab(TAB_LABELS["cv"]), self),
            "visualization": VisualizationTab(self.tabview.tab(TAB_LABELS["visualization"]), self),
            "simulation": SimulationTab(self.tabview.tab(TAB_LABELS["simulation"]), self),
        }
        self.tabs["viz"] = self.tabs["visualization"]
        if hasattr(self.tabview, "_segmented_button"):
            self.tabview._segmented_button.configure(command=self._handle_tab_click)

    def _resolve_tab_label(self, tab_key_or_label: str) -> str:
        return TAB_LABELS.get(tab_key_or_label, tab_key_or_label)

    def _tab_key_from_label(self, label: str) -> str | None:
        for key, tab_label in TAB_LABELS.items():
            if tab_label == label:
                return key
        return None

    def _tab_unlock_state(self) -> dict[str, tuple[bool, str]]:
        has_data = self.df is not None
        feature_count = len(self.get_selected_features()) if self.feature_vars else 0
        target_count = len(self.get_selected_targets()) if self.target_vars else 0
        has_analysis = bool(self.analysis_results)
        has_models = bool(self.model_cache)

        return {
            "data_load": (True, ""),
            "features": (has_data, "请先在【📂 数据】页面导入数据。"),
            "targets": (has_data, "请先在【📂 数据】页面导入数据。"),
            "preprocess": (has_data, "请先在【📂 数据】页面导入数据。"),
            "statistics": (has_data, "请先在【📂 数据】页面导入数据。"),
            "model_params": (has_data, "请先在【📂 数据】页面导入数据。"),
            "analysis": (
                has_data and feature_count > 0 and target_count > 0,
                "请先完成【📋 特征】和【🎯 目标】的勾选。"
                if has_data else "请先在【📂 数据】页面导入数据。",
            ),
            "cv": (
                has_data and feature_count > 0 and target_count > 0,
                "请先完成【📋 特征】和【🎯 目标】的勾选。"
                if has_data else "请先在【📂 数据】页面导入数据。",
            ),
            "visualization": (
                has_analysis,
                "请先在【🚀 分析】页面运行一次完整分析。",
            ),
            "simulation": (
                has_models,
                "请先在【🚀 分析】页面生成可用模型缓存。",
            ),
        }

    def _next_step_descriptor(self) -> tuple[str, str, str]:
        has_data = self.df is not None
        feature_count = len(self.get_selected_features()) if self.feature_vars else 0
        target_count = len(self.get_selected_targets()) if self.target_vars else 0
        has_analysis = bool(self.analysis_results)
        has_models = bool(self.model_cache)

        if not has_data:
            return ("data_load", "流程 1/5：导入数据", "下一步：在【📂 数据】页面选择文件并加载，其他页面会逐步解锁。")
        if feature_count == 0:
            return ("features", "流程 2/5：选择特征", "下一步：在【📋 特征】页面勾选用于建模的影响因素。")
        if target_count == 0:
            return ("targets", "流程 3/5：选择目标", "下一步：在【🎯 目标】页面勾选待分析的目标变量。")
        if not has_analysis:
            return ("analysis", "流程 4/5：运行分析", "下一步：确认【⚙️ 模型】参数后，前往【🚀 分析】生成结果。")
        if not has_models:
            return ("visualization", "流程 5/5：查看结果", "分析结果已生成，接下来可以在【📈 图表】查看图形。")
        return ("visualization", "流程完成", "已完成主要流程，可前往【📈 图表】查看结果，或在【🔮 模拟】中进行情景模拟。")

    def _update_flow_banner(self) -> None:
        if self.flow_label is None or self.flow_hint_label is None or self.flow_action_btn is None:
            return
        next_key, title, hint = self._next_step_descriptor()
        self.flow_label.configure(text=title)
        self.flow_hint_label.configure(text=hint)
        self.flow_action_btn.configure(command=lambda key=next_key: self.navigate_to_tab(key, force=True))
        if next_key == "analysis":
            self.flow_action_btn.configure(text="开始分析")
        elif next_key == "visualization" and bool(self.analysis_results):
            self.flow_action_btn.configure(text="查看结果")
        else:
            self.flow_action_btn.configure(text="前往下一步")

    def _show_locked_tab_hint(self, selected_label: str, reason: str) -> None:
        show_message(self, "🔒 当前步骤未解锁", f"{selected_label}\n\n{reason}", "info")

    def _handle_tab_click(self, selected_name: str) -> None:
        rules = self._tab_unlock_state()
        tab_key = self._tab_key_from_label(selected_name)
        current_name = getattr(self.tabview, "_current_name", TAB_LABELS["data_load"])

        if tab_key is None:
            self.tabview._segmented_button_callback(selected_name)
            self._update_flow_banner()
            return

        enabled, reason = rules.get(tab_key, (True, ""))
        if enabled:
            self.tabview._segmented_button_callback(selected_name)
            self._update_flow_banner()
            return

        if hasattr(self.tabview, "_segmented_button"):
            self.tabview._segmented_button.set(current_name)
        self._show_locked_tab_hint(selected_name, reason)

    def navigate_to_tab(self, tab_key_or_label: str, force: bool = False) -> bool:
        if self.tabview is None:
            return False
        target_label = self._resolve_tab_label(tab_key_or_label)
        tab_key = self._tab_key_from_label(target_label)
        rules = self._tab_unlock_state()
        if not force and tab_key is not None:
            enabled, reason = rules.get(tab_key, (True, ""))
            if not enabled:
                self._show_locked_tab_hint(target_label, reason)
                return False
        self.tabview.set(target_label)
        self._update_flow_banner()
        return True

    def refresh_navigation_state(self) -> None:
        rules = self._tab_unlock_state()
        segmented = getattr(self.tabview, "_segmented_button", None)
        buttons = getattr(segmented, "_buttons_dict", {}) if segmented is not None else {}

        for key, label in TAB_LABELS.items():
            button = buttons.get(label)
            if button is None:
                continue
            enabled, _reason = rules.get(key, (True, ""))
            button.configure(state="normal" if enabled else "disabled")

        self._update_flow_banner()

        for tab in dict.fromkeys(self.tabs.values()):
            if hasattr(tab, "refresh_empty_state"):
                try:
                    tab.refresh_empty_state()
                except Exception:
                    logger.debug("Failed to refresh empty state for %s", tab, exc_info=True)

    def _collect_tk_vars(self) -> dict[str, Any]:
        saved: dict[str, Any] = {}
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            obj = getattr(self, attr_name, None)
            if isinstance(obj, (ctk.StringVar, ctk.IntVar, ctk.DoubleVar, ctk.BooleanVar)):
                try:
                    saved[attr_name] = obj.get()
                except Exception:
                    logger.debug("Failed to collect Tk var %s", attr_name, exc_info=True)
        return saved

    def _restore_tk_vars(self, saved: dict[str, Any]) -> None:
        for attr_name, value in saved.items():
            obj = getattr(self, attr_name, None)
            if obj is not None:
                try:
                    obj.set(value)
                except Exception:
                    logger.debug("Failed to restore Tk var %s", attr_name, exc_info=True)

    def _map_color(self, current: Any, old_palette: dict[str, str]) -> Any:
        if isinstance(current, (list, tuple)):
            return type(current)(self._map_color(item, old_palette) for item in current)
        if not isinstance(current, str):
            return current
        if current == "transparent":
            return current
        for key, old_value in old_palette.items():
            if current == old_value:
                return C.get(key, current)
        return current

    def _apply_theme_to_widget_tree(self, widget: Any, old_palette: dict[str, str]) -> None:
        option_names = (
            "fg_color",
            "bg_color",
            "text_color",
            "border_color",
            "hover_color",
            "button_color",
            "button_hover_color",
            "progress_color",
            "segmented_button_fg_color",
            "segmented_button_selected_color",
            "segmented_button_selected_hover_color",
            "segmented_button_unselected_color",
            "segmented_button_unselected_hover_color",
            "dropdown_fg_color",
            "dropdown_hover_color",
            "dropdown_text_color",
            "scrollbar_button_color",
            "scrollbar_button_hover_color",
        )
        for option_name in option_names:
            try:
                current_value = widget.cget(option_name)
            except Exception:
                continue
            try:
                widget.configure(**{option_name: self._map_color(current_value, old_palette)})
            except Exception:
                logger.debug("Failed to update %s on %s", option_name, widget, exc_info=True)

        # CTkTabview hides its internal segmented button from winfo_children(),
        # so the tab strip can keep stale colors after theme switching.
        if hasattr(widget, "_segmented_button"):
            try:
                widget.configure(
                    segmented_button_fg_color=C["tab_bg"],
                    segmented_button_selected_color=C["tab_selected"],
                    segmented_button_selected_hover_color=C["accent_hover"],
                    segmented_button_unselected_color=C["tab_bg"],
                    segmented_button_unselected_hover_color=C["tab_hover"],
                    text_color=C["text_primary"],
                )
            except Exception:
                logger.debug("Failed to refresh tabview segmented button colors", exc_info=True)

        children = list(widget.winfo_children())
        for internal_child_name in ("_segmented_button", "_canvas"):
            internal_child = getattr(widget, internal_child_name, None)
            if internal_child is not None and internal_child not in children:
                children.append(internal_child)

        for child in children:
            self._apply_theme_to_widget_tree(child, old_palette)

    def switch_theme(self, mode: str, scale: int | None = None) -> None:
        saved_vars = self._collect_tk_vars()
        current_tab = None
        if self.tabview is not None:
            try:
                current_tab = self.tabview.get()
            except Exception:
                logger.debug("Failed to read current tab", exc_info=True)

        old_palette = dict(C)
        apply_theme(mode)
        ctk.set_appearance_mode("Light" if mode == "Light" else "Dark")
        self.configure(fg_color=C["bg_base"])

        if scale is not None:
            try:
                ctk.set_widget_scaling(scale / 100.0)
            except Exception:
                logger.debug("Failed to set widget scaling", exc_info=True)

        self._apply_theme_to_widget_tree(self, old_palette)
        self._restore_tk_vars(saved_vars)
        self._refresh_data_driven_views()
        self.refresh_navigation_state()
        self.update_status_bar()

        if current_tab:
            try:
                self.tabview.set(current_tab)
            except Exception:
                logger.debug("Failed to restore current tab", exc_info=True)

    def _refresh_data_driven_views(self) -> None:
        try:
            if self._state_store.df is not None:
                self.tabs["features"].populate()
                self.tabs["targets"].populate()
                self.tabs["preprocess"].refresh_columns()
                self.tabs["preprocess"].check_unlock()
                self.tabs["statistics"].enable_buttons()
            self.tabs["visualization"].refresh_targets()
            self.tabs["simulation"].populate_targets()
            self.refresh_navigation_state()
        except Exception:
            logger.exception("Failed to refresh views after theme/data change")

    def get_selected_features(self) -> list[str]:
        return [
            col
            for col, info in self.feature_vars.items()
            if isinstance(info, dict) and info.get("selected", ctk.BooleanVar()).get()
        ]

    def get_selected_targets(self) -> list[str]:
        return [col for col, var in self.target_vars.items() if var.get()]

    def get_feature_types(self) -> list[tuple[str, str]]:
        return [
            (col, info["type"].get())
            for col, info in self.feature_vars.items()
            if info["selected"].get()
        ]

    def update_status_bar(self) -> None:
        if self.status_bar is not None:
            self.status_bar.refresh()

    def add_recent_file(self, file_path: str) -> None:
        self.config_manager.add_recent(file_path)

    def _cmd_open_file(self) -> None:
        from tkinter import filedialog

        file_name = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[("Excel", "*.xlsx *.xls"), ("所有文件", "*.*")],
        )
        if file_name:
            tab = self.tabs["data_load"]
            tab.file_entry.delete(0, "end")
            tab.file_entry.insert(0, file_name)
            self.file_path = file_name
            tab.load_data()
            self.navigate_to_tab("data_load", force=True)

    def on_closing(self) -> None:
        if "analysis" in self.tabs:
            analysis_tab = self.tabs["analysis"]
            if hasattr(analysis_tab, "_cancel"):
                analysis_tab._cancel()
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except Exception:
            logger.debug("Failed to close matplotlib figures", exc_info=True)
        try:
            self.quit()
            self.destroy()
        except Exception:
            logger.debug("Failed to destroy main window", exc_info=True)
        os._exit(0)


def main() -> PollutantAnalysisApp:
    import multiprocessing

    multiprocessing.freeze_support()
    app = PollutantAnalysisApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    return app


if __name__ == "__main__":
    main().mainloop()
