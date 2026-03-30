"""
高级可视化标签页模块 (📈 可视图)
---------------------------------------------------------------------
生成专业出版级别的图表，主要用于解释复杂机器学习模型。

支持的图表类型:
1. SHAP 摘要图 (Summary Plot): 全局解释性，显示特征对模型输出的非线性与交互影响。
2. 特征重要性条形图 (Bar Chart): 基于排列重要性计算的特征相对重要性贡献百分比。
3. 学习曲线 (Learning Curve): 诊断模型是否陷入高偏差(欠拟合)或高方差(过拟合)。
4. 模型性能雷达图 (Radar Chart): 多维评估指标对比。
5. 散点图与残差图: 直观观察预测值与真实值的贴合程度及残差分布。

安全机制:
- 对计算复杂度极高的模型（如 Stacking）禁用学习曲线生成。
- 为 AdaBoost 的 KernelExplainer 静音进度条，防止 PyInstaller 打包后的控制台奔溃。
"""

import contextlib
import io
import logging
import threading

import numpy as np
import pandas as pd
import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from constants import get_app_log_path
from .theme import (C, FONTS, show_message, make_card, make_inner_frame,
                    make_btn_primary, make_btn_secondary, make_optionmenu,
                    make_empty_state)
from .analysis_engine import prepare_feature_frame


logger = logging.getLogger(__name__)


class VisualizationTab:

    def __init__(self, parent, app):
        self.app = app
        self.parent = parent
        self._canvas = None
        self._learning_curve_running = False
        self._shap_running = False
        self._build()

    def _build(self):
        card = make_card(self.parent)
        card.pack(fill="x", padx=18, pady=(14, 6))
        row = make_inner_frame(card)
        row.pack(fill="x", padx=18, pady=12)

        ctk.CTkLabel(row, text="图表类型:", font=FONTS["body_bold"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(0, 8))
        self.viz_type = ctk.StringVar(value="相关性热力图")
        self.type_menu = make_optionmenu(
            row, variable=self.viz_type, width=200,
            values=["相关性热力图", "特征重要性柱状图", "模型性能对比", "预测值vs真实值", "模型性能雷达图", "SHAP 摘要图", "学习曲线 (Learning Curve)"]
        )
        self.type_menu.pack(side="left", padx=4)

        ctk.CTkLabel(row, text="模型:", font=FONTS["body"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(18, 4))
        self.viz_model = ctk.StringVar(value="RandomForest")
        self.model_menu = make_optionmenu(
            row, variable=self.viz_model, width=160,
            values=["RandomForest", "AdaBoost", "XGBoost", "LightGBM", "CatBoost", "GAM", "Stacking"]
        )
        self.model_menu.pack(side="left", padx=4)

        ctk.CTkLabel(row, text="目标:", font=FONTS["body"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(18, 4))
        self.viz_target = ctk.StringVar(value="(所有目标平均)")
        self.target_menu = make_optionmenu(
            row, variable=self.viz_target, width=180,
            values=["(所有目标平均)"]
        )
        self.target_menu.pack(side="left", padx=4)

        ctk.CTkLabel(row, text="SHAP:", font=FONTS["body"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(18, 4))
        self.shap_mode = ctk.StringVar(value="整体特征")
        self.shap_mode_menu = make_optionmenu(
            row, variable=self.shap_mode, width=140,
            values=["整体特征", "类别展开"]
        )
        self.shap_mode_menu.pack(side="left", padx=4)

        self.generate_btn = make_btn_primary(
            row, text="📈  生成图表", width=120, command=self._generate
        )
        self.generate_btn.pack(side="left", padx=16)
                         
        make_btn_secondary(row, text="💾 保存图片", width=100,
                           command=self._save_image).pack(side="left")

        self.chart_frame = make_card(self.parent)
        self.chart_frame.pack(fill="both", expand=True, padx=18, pady=(6, 14))
        self.empty_state = make_empty_state(
            self.chart_frame,
            "📈",
            "暂无可视化结果",
            "请先在【🚀 分析】页面运行一次完整分析，之后这里会解锁模型图表、散点图和 SHAP 结果。",
            button_text="前往分析页",
            command=lambda: self.app.navigate_to_tab("analysis", force=True) if hasattr(self.app, 'navigate_to_tab') else None,
        )
        self.empty_state.pack(fill="both", expand=True, padx=6, pady=6)

        self.context_label = ctk.CTkLabel(
            self.parent,
            text="热力图默认展示数值型特征的相关性；模型相关图表会在分析完成后联动可选模型与目标。",
            font=FONTS["small"](),
            text_color=C["text_muted"],
            anchor="w"
        )
        self.context_label.pack(fill="x", padx=24, pady=(0, 10))

        for var in (self.viz_type, self.viz_model, self.viz_target, self.shap_mode):
            var.trace_add("write", lambda *_: self._refresh_context())
        self.refresh_targets()
        self._refresh_context()
        self.refresh_empty_state()

    def _clear(self):
        if hasattr(self, '_canvas') and self._canvas:
            import matplotlib.pyplot as plt
            fig = self._canvas.figure
            self._canvas.get_tk_widget().destroy()
            plt.close(fig)
            self._canvas = None
        self._show_empty_state()

    def _embed(self, fig):
        self._hide_empty_state()
        self._canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    def _show_empty_state(self):
        if self.empty_state is not None and not self.empty_state.winfo_manager():
            self.empty_state.pack(fill="both", expand=True, padx=6, pady=6)

    def _hide_empty_state(self):
        if self.empty_state is not None and self.empty_state.winfo_manager():
            self.empty_state.pack_forget()

    def refresh_empty_state(self):
        if not self.app.analysis_results:
            self.empty_state.title_label.configure(text="暂无可视化结果")
            self.empty_state.message_label.configure(
                text="请先在【🚀 分析】页面运行一次完整分析，之后这里会解锁模型图表、散点图和 SHAP 结果。"
            )
            if self.empty_state.action_button is not None:
                self.empty_state.action_button.configure(
                    text="前往分析页",
                    command=lambda: self.app.navigate_to_tab("analysis", force=True) if hasattr(self.app, 'navigate_to_tab') else None,
                )
                if not self.empty_state.action_button.winfo_manager():
                    self.empty_state.action_button.pack(pady=(0, 24))
        else:
            self.empty_state.title_label.configure(text="选择图表并生成")
            self.empty_state.message_label.configure(
                text="分析结果已可用。选择图表类型、模型和目标变量后，点击上方“生成图表”。"
            )
            if self.empty_state.action_button is not None and self.empty_state.action_button.winfo_manager():
                self.empty_state.action_button.pack_forget()

    def refresh_targets(self):
        self.refresh_models()
        targets = ["(所有目标平均)"]
        if self.app.analysis_results:
            first_df = next(iter(self.app.analysis_results.values()))
            if '目标变量' in first_df.columns:
                targets.extend(first_df['目标变量'].dropna().astype(str).unique().tolist())
        self.target_menu.configure(values=targets)
        if self.viz_target.get() not in targets:
            self.viz_target.set(targets[0])
        self.refresh_empty_state()

    def refresh_models(self):
        if self.app.analysis_results:
            models = list(self.app.analysis_results.keys())
        else:
            models = ["RandomForest", "AdaBoost", "XGBoost", "LightGBM", "CatBoost", "GAM", "Stacking"]
        self.model_menu.configure(values=models)
        if self.viz_model.get() not in models:
            self.viz_model.set(models[0])

    def _metric_df(self):
        selected_target = self._selected_target()
        rows = []
        for model_name, df in self.app.analysis_results.items():
            dfr = df.copy()
            if selected_target is not None and '目标变量' in dfr.columns:
                dfr = dfr[dfr['目标变量'] == selected_target]
            if dfr.empty:
                continue
            rows.append({
                '模型': model_name,
                '平均R²': round(dfr['R²'].mean(), 4),
                '平均SCC': round(dfr['SCC'].mean(), 4),
                '平均PCC': round(dfr['PCC'].mean(), 4),
            })
        return pd.DataFrame(rows)

    def _refresh_context(self):
        chart_type = self.viz_type.get()
        model_needed = chart_type in ["特征重要性柱状图", "预测值vs真实值", "SHAP 摘要图", "学习曲线 (Learning Curve)"]
        target_optional = chart_type in ["特征重要性柱状图", "模型性能对比", "模型性能雷达图", "预测值vs真实值", "SHAP 摘要图", "学习曲线 (Learning Curve)"]
        shap_needed = chart_type == "SHAP 摘要图"
        self.model_menu.configure(state="normal" if model_needed else "disabled")
        self.target_menu.configure(state="normal" if target_optional else "disabled")
        self.shap_mode_menu.configure(state="normal" if shap_needed else "disabled")

        if chart_type == "相关性热力图":
            selected = self.app.get_selected_features() if hasattr(self.app, 'get_selected_features') else []
            msg = f"当前热力图优先展示已选数值特征，共 {len(selected)} 个候选。"
        elif chart_type in ["模型性能对比", "模型性能雷达图"]:
            target = self._selected_target()
            msg = "当前展示整体模型表现。"
            if target is not None:
                msg = f"当前按目标变量 {target} 对比各模型表现。"
        else:
            target = self._selected_target()
            msg = f"当前模型：{self.viz_model.get()}"
            if target is not None:
                msg += f"｜目标：{target}"
            if chart_type == "SHAP 摘要图":
                msg += f"｜展示：{self.shap_mode.get()}"
        self.context_label.configure(text=msg)

    def _split_encoded_feature_name(self, encoded_name, original_columns):
        clean_name = encoded_name.split('__', 1)[-1]
        sorted_columns = sorted(original_columns, key=len, reverse=True)
        for col in sorted_columns:
            prefix = f"{col}_"
            if clean_name == col:
                return col, col
            if clean_name.startswith(prefix):
                suffix = clean_name[len(prefix):]
                return col, f"{col} ({suffix})"
        return clean_name, clean_name

    def _prepare_shap_display_data(self, shap_values, X_transformed, feature_names, original_X):
        return self._prepare_shap_display_data_for_mode(
            shap_values,
            X_transformed,
            feature_names,
            original_X,
            self.shap_mode.get(),
        )

    def _prepare_shap_display_data_for_mode(self, shap_values, X_transformed, feature_names, original_X, mode):
        sv_df = pd.DataFrame(shap_values, columns=feature_names)
        original_columns = list(original_X.columns)
        grouped = {}
        expanded_labels = []
        for name in feature_names:
            base_name, expanded_name = self._split_encoded_feature_name(name, original_columns)
            grouped[name] = base_name
            expanded_labels.append(expanded_name)

        if mode == "类别展开":
            display_sv = sv_df.copy()
            display_sv.columns = expanded_labels
            display_X = pd.DataFrame(X_transformed, columns=expanded_labels)
            return display_sv.values, display_X

        grouped_columns = {}
        for encoded_name, base_name in grouped.items():
            grouped_columns.setdefault(base_name, []).append(encoded_name)

        display_sv = pd.DataFrame(index=sv_df.index)
        for base_name, encoded_cols in grouped_columns.items():
            display_sv[base_name] = sv_df[encoded_cols].sum(axis=1)

        display_X = original_X.copy()
        for col in display_X.columns:
            if not pd.api.types.is_numeric_dtype(display_X[col]):
                display_X[col] = pd.factorize(display_X[col].astype(str))[0]
        display_X = display_X[display_sv.columns]
        return display_sv.values, display_X

    def _selected_target(self):
        target = self.viz_target.get()
        return None if target == "(所有目标平均)" else target

    @contextlib.contextmanager
    def _suppress_external_output(self):
        """Mute noisy third-party progress/log output in packaged GUI builds."""
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield

    def _normalize_shap_values(self, shap_values):
        if isinstance(shap_values, list):
            return shap_values[0]
        if hasattr(shap_values, 'values'):
            return shap_values.values
        return shap_values

    def _runtime_error_message(self, summary: str, detail: str) -> str:
        return f"{summary}\n{detail}\n\n详细日志已写入:\n{get_app_log_path()}"

    def _render_shap_summary(self, mn, col, sv, X_plot):
        from matplotlib.figure import Figure
        import matplotlib.pyplot as plt
        import shap

        self._shap_running = False
        self._set_generate_button_state(True)

        fig = Figure(figsize=(10, 6), dpi=90, facecolor=C["bg_primary"])
        ax = fig.add_subplot(111)

        old_gca = plt.gca
        old_gcf = plt.gcf
        plt.gca = lambda **kwargs: ax
        plt.gcf = lambda: fig

        try:
            shap.summary_plot(sv, X_plot, show=False, color_bar=True, max_display=15)
        finally:
            plt.gca = old_gca
            plt.gcf = old_gcf

        ax.set_title(f"{mn} ({col}) - SHAP 摘要图", fontsize=14, color=C["text_primary"], pad=15)
        ax.set_facecolor(C["bg_primary"])
        ax.tick_params(colors=C["text_primary"])
        ax.xaxis.label.set_color(C["text_primary"])
        ax.yaxis.label.set_color(C["text_primary"])

        cb = fig.axes[-1] if len(fig.axes) > 1 else None
        if cb:
            cb.tick_params(colors=C["text_primary"])
            cb.set_ylabel("特征值 (Feature Value)", color=C["text_primary"])

        fig.tight_layout()
        self._embed(fig)

        if hasattr(self.app, 'status_bar'):
            self.app.status_bar.set("SHAP 图已生成", None)

    def _shap_failed(self, message: str):
        self._shap_running = False
        self._set_generate_button_state(True)
        show_message(self.app, "❌", message, "error")
        if hasattr(self.app, 'status_bar'):
            self.app.status_bar.set("SHAP 图生成失败", None)

    def _adaboost_shap_worker(
        self,
        actual_model,
        X_transformed_df,
        X_sample,
        X_sample_transformed,
        feature_names,
        shap_mode,
        mn,
        col,
    ):
        try:
            import shap

            with self._suppress_external_output():
                background = shap.kmeans(X_transformed_df, 10)
                explainer = shap.KernelExplainer(actual_model.predict, background)
                shap_values = explainer.shap_values(X_sample_transformed, silent=True)

            sv = self._normalize_shap_values(shap_values)
            sv, X_plot = self._prepare_shap_display_data_for_mode(
                sv,
                X_sample_transformed.values,
                feature_names,
                X_sample,
                shap_mode,
            )
        except Exception as exc:
            import traceback

            err_msg = traceback.format_exc()
            print("SHAP Error:\n", err_msg)
            self._safe_after(lambda: self._shap_failed(f"生成 SHAP 图失败:\n{exc}\n\n请查看终端获取详细信息"))
            return

        self._safe_after(lambda: self._render_shap_summary(mn, col, sv, X_plot))

    def _adaboost_shap_worker_safe(
        self,
        actual_model,
        X_transformed_df,
        X_sample,
        X_sample_transformed,
        feature_names,
        shap_mode,
        mn,
        col,
    ):
        try:
            import shap

            with self._suppress_external_output():
                background = shap.kmeans(X_transformed_df, 10)
                explainer = shap.KernelExplainer(actual_model.predict, background)
                shap_values = explainer.shap_values(X_sample_transformed, silent=True)

            sv = self._normalize_shap_values(shap_values)
            sv, X_plot = self._prepare_shap_display_data_for_mode(
                sv,
                X_sample_transformed.values,
                feature_names,
                X_sample,
                shap_mode,
            )
        except Exception as exc:
            logger.exception("Failed to generate AdaBoost SHAP summary")
            message = self._runtime_error_message("生成 SHAP 图失败:", str(exc))
            self._safe_after(lambda m=message: self._shap_failed(m))
            return

        self._safe_after(lambda: self._render_shap_summary(mn, col, sv, X_plot))

    def _select_model_key(self, model_name):
        selected_target = self._selected_target()
        
        # 兼容两种 key 格式: 字符串 "RandomForest" 或 元组 ("RandomForest", "TargetCol")
        matching_keys = []
        for k in self.app.model_cache.keys():
            if isinstance(k, tuple) and k[0] == model_name:
                matching_keys.append(k)
            elif k == model_name:
                matching_keys.append(k)
                
        if selected_target is not None:
            # 过滤出匹配目标变量的 key
            matching_keys = [k for k in matching_keys if (isinstance(k, tuple) and k[1] == selected_target)]
            
        return matching_keys[0] if matching_keys else None

    def _generate(self):
        t = self.viz_type.get()
        if t == "相关性热力图":
            self._heatmap()
        elif t == "特征重要性柱状图":
            self._importance()
        elif t == "模型性能对比":
            self._comparison()
        elif t == "预测值vs真实值":
            self._scatter()
        elif t == "模型性能雷达图":
            self._radar()
        elif t == "SHAP 摘要图":
            self._shap_summary()
        elif t == "学习曲线 (Learning Curve)":
            self._start_learning_curve()

    def _save_image(self):
        if self._canvas is None:
            show_message(self.app, "ℹ️ 提示", "请先生成一张图表", "info")
            return
            
        from tkinter import filedialog
        import os
        
        fp = filedialog.asksaveasfilename(
            title="保存图表图片", 
            defaultextension=".png",
            filetypes=[("PNG 图片", "*.png"), ("JPEG 图片", "*.jpg"), ("PDF 文档", "*.pdf")],
            initialfile=f"{self.viz_type.get().split(' ')[0]}.png"
        )
        if not fp:
            return
            
        try:
            # 拿到 matplotlib 的 Figure 对象并保存
            fig = self._canvas.figure
            fig.savefig(fp, dpi=300, bbox_inches='tight')
            show_message(self.app, "✅ 保存成功", f"图片已保存至:\n{fp}", "info")
        except Exception as e:
            show_message(self.app, "❌ 保存失败", str(e), "error")

    # ── 相关性热力图 ──

    def _heatmap(self):
        if self.app.df is None:
            show_message(self.app, "❌", "请先加载数据", "error")
            return
            show_message(self.app, "❌", "请先加载数据", "error")
            return
        nc_all = self.app.df.select_dtypes(include=[np.number]).columns.tolist()
        selected = self.app.get_selected_features() if hasattr(self.app, 'get_selected_features') else []
        nc = [c for c in selected if c in nc_all][:20] if selected else nc_all[:20]
        if len(nc) < 2:
            show_message(self.app, "ℹ️", "数值列不足", "info")
            return
        corr = self.app.df[nc].corr()
        self._clear()
        fig = Figure(figsize=(10, 8), dpi=90, facecolor=C["bg_primary"])
        ax = fig.add_subplot(111)
        im = ax.imshow(corr.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(len(nc)))
        ax.set_yticks(range(len(nc)))
        ax.set_xticklabels(nc, rotation=45, ha='right', fontsize=8, color=C["text_primary"])
        ax.set_yticklabels(nc, fontsize=8, color=C["text_primary"])
        title = "特征相关性热力图"
        if selected:
            title = "已选特征相关性热力图"
        ax.set_title(title, fontsize=14, color=C["text_primary"], pad=12)
        ax.set_facecolor(C["bg_primary"])
        n = len(nc)
        for i in range(n):
            for j in range(n):
                v = corr.values[i, j]
                tc = 'white' if abs(v) > 0.5 else C["text_primary"]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7, color=tc)
        fig.tight_layout()
        self._embed(fig)

    # ── SHAP 摘要图 ──

    def _shap_summary(self):
        if self._shap_running:
            show_message(self.app, "提示", "SHAP 图正在后台生成，请稍候。", "info")
            return

        try:
            import shap
        except ImportError:
            show_message(self.app, "❌", "未安装 shap 库\n请在终端运行: pip install shap", "error")
            return
            
        mn = self.viz_model.get()
        if not hasattr(self.app, 'model_cache') or not hasattr(self.app, 'X_cache'):
            show_message(self.app, "ℹ️", "请先运行分析以生成模型缓存", "info")
            return
            
        key = self._select_model_key(mn)
        if key is None:
            show_message(self.app, "ℹ️", f"模型 {mn} 没有缓存，请确认分析已运行", "info")
            return
        model = self.app.model_cache[key]
        col = key[1]
        cached_X = self.app.X_cache.get((mn, col), self.app.X_cache.get(col))
        if cached_X is None:
            show_message(self.app, "ℹ️", "缺少该模型的特征缓存，请重新运行分析", "info")
            return
        X = cached_X.copy()
        
        # 部分模型可能不支持 SHAP (例如 GAM)
        if mn in ['GAM', 'Stacking']:
            show_message(self.app, "ℹ️", f"{mn} 模型当前不支持 SHAP 分析", "info")
            return
            
        self._clear()
        
        # SHAP 绘图会直接使用 matplotlib 的 pyplot，我们需要捕获它
        import matplotlib.pyplot as plt
        
        # 确保在使用前清理旧的 plt 状态，避免干扰
        plt.clf()
        plt.close('all')
        
        # 设置字体以免中文乱码
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        try:
            import traceback
            
            # 如果模型是 Pipeline，我们需要提取出预处理器和实际模型
            if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps and 'model' in model.named_steps:
                preprocessor = model.named_steps['preprocessor']
                actual_model = model.named_steps['model']
                
                # SHAP 需要对预处理后的数据进行解释
                X_transformed = preprocessor.transform(X)
                
                # 尝试获取特征名称
                try:
                    feature_names = list(preprocessor.get_feature_names_out())
                except Exception:
                    feature_names = [f"Feature {i}" for i in range(X_transformed.shape[1])]
            else:
                actual_model = model
                X_transformed = X.values
                feature_names = list(X.columns)
                
            # TreeExplainer 不支持 AdaBoost，所以要判断模型类型
            if mn == 'AdaBoost':
                # AdaBoost 需要用 KernelExplainer 或 PermutationExplainer
                # 为了速度，我们用 KMeans 对背景数据做个简要汇总
                X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
                # 计算 SHAP 值 (KernelExplainer 较慢，所以只取部分样本展示以防卡死，这里取前100个)
                sample_size = min(len(X), 100)
                X_sample = X.iloc[:sample_size].copy()
                X_sample_transformed = X_transformed_df.iloc[:sample_size].copy()
                shap_mode = self.shap_mode.get()

                self._shap_running = True
                self._set_generate_button_state(False)
                if hasattr(self.app, 'status_bar'):
                    self.app.status_bar.set("正在后台生成 AdaBoost 的 SHAP 图...", None)

                threading.Thread(
                    target=self._adaboost_shap_worker_safe,
                    args=(
                        actual_model,
                        X_transformed_df,
                        X_sample,
                        X_sample_transformed,
                        feature_names,
                        shap_mode,
                        mn,
                        col,
                    ),
                    daemon=True,
                ).start()
                return
            else:
                explainer = shap.TreeExplainer(actual_model)
                shap_values = explainer.shap_values(X_transformed)
            
            # 提取shap_values（不同模型返回的结构不同）
            if isinstance(shap_values, list):
                # 有些模型（如多分类）返回list
                sv = shap_values[0]
            elif hasattr(shap_values, 'values'):
                # 较新的 shap 版本返回 Explanation 对象
                sv = shap_values.values
            else:
                sv = shap_values

            if mn == 'AdaBoost':
                sv, X_plot = self._prepare_shap_display_data(
                    sv,
                    X_sample_transformed.values,
                    feature_names,
                    X_sample
                )
            else:
                sv, X_plot = self._prepare_shap_display_data(
                    sv,
                    X_transformed,
                    feature_names,
                    X
                )
                
            # 完全抛弃隐式的 plt 状态，使用 OO API
            from matplotlib.figure import Figure
            fig = Figure(figsize=(10, 6), dpi=90, facecolor=C["bg_primary"])
            ax = fig.add_subplot(111)
            
            # 为了让不支持 ax 参数的 SHAP 库画在我们的 ax 上，临时挂载当前上下文
            import matplotlib.pyplot as plt
            old_gca = plt.gca
            old_gcf = plt.gcf
            plt.gca = lambda **kwargs: ax
            plt.gcf = lambda: fig
            
            try:
                shap.summary_plot(sv, X_plot, show=False, color_bar=True, max_display=15)
            finally:
                plt.gca = old_gca
                plt.gcf = old_gcf
            
            # 定制化调整
            ax.set_title(f"{mn} ({col}) - SHAP 摘要图", fontsize=14, color=C["text_primary"], pad=15)
            ax.set_facecolor(C["bg_primary"])
            ax.tick_params(colors=C["text_primary"])
            ax.xaxis.label.set_color(C["text_primary"])
            ax.yaxis.label.set_color(C["text_primary"])
            
            # 调整 colorbar
            cb = fig.axes[-1] if len(fig.axes) > 1 else None
            if cb:
                cb.tick_params(colors=C["text_primary"])
                cb.set_ylabel("特征值 (Feature Value)", color=C["text_primary"])
                
            fig.tight_layout()
            self._embed(fig)
        except Exception as e:
            err_msg = traceback.format_exc()
            show_message(self.app, "❌", f"生成 SHAP 图失败:\n{e}\n\n请查看终端获取详细信息", "error")
            print("SHAP Error:\n", err_msg)

    # ── 学习曲线 (Learning Curve) ──
    
    def _safe_after(self, callback):
        try:
            self.app.after(0, callback)
            return True
        except Exception:
            return False

    def _set_generate_button_state(self, enabled: bool) -> None:
        if hasattr(self, 'generate_btn'):
            if enabled and not self._learning_curve_running and not self._shap_running:
                self.generate_btn.configure(state="normal")
            else:
                self.generate_btn.configure(state="disabled")

    def _resolve_cached_feature_specs(self, cached_X: pd.DataFrame) -> list[tuple[str, str]]:
        feature_specs = []
        for feature_name in cached_X.columns.tolist():
            feature_info = getattr(self.app, 'feature_vars', {}).get(feature_name, {})
            feature_type_var = feature_info.get('type')
            if feature_type_var is not None:
                feature_type = feature_type_var.get()
            elif pd.api.types.is_numeric_dtype(cached_X[feature_name]):
                feature_type = 'numeric'
            else:
                feature_type = 'categorical'
            feature_specs.append((feature_name, feature_type))
        return feature_specs

    def _render_learning_curve(self, mn, col, train_sizes, train_scores, test_scores):
        self._learning_curve_running = False
        self._set_generate_button_state(True)

        fig = Figure(figsize=(10, 6), dpi=90, facecolor=C["bg_primary"])
        ax = fig.add_subplot(111)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        ax.grid(True, linestyle='--', alpha=0.6, color=C["border"])
        ax.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color=C["error"],
        )
        ax.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color=C["success"],
        )

        ax.plot(train_sizes, train_scores_mean, 'o-', color=C["error"], label="训练集得分 (R²)")
        ax.plot(train_sizes, test_scores_mean, 'o-', color=C["success"], label="交叉验证得分 (R²)")

        ax.set_title(f"{mn} ({col}) - 学习曲线", fontsize=14, color=C["text_primary"], pad=15)
        ax.set_xlabel("训练样本数量", fontsize=12, color=C["text_primary"])
        ax.set_ylabel("得分 (R²)", fontsize=12, color=C["text_primary"])
        ax.set_facecolor(C["bg_primary"])
        ax.tick_params(colors=C["text_muted"])

        for sp in ax.spines.values():
            sp.set_color(C["border"])

        ax.legend(loc="best", facecolor=C["bg_primary"], edgecolor=C["border"], labelcolor=C["text_primary"])
        fig.tight_layout()
        self._embed(fig)

        if hasattr(self.app, 'status_bar'):
            self.app.status_bar.set("学习曲线已生成", None)

    def _learning_curve_failed(self, message: str):
        self._learning_curve_running = False
        self._set_generate_button_state(True)
        show_message(self.app, "❌", message, "error")
        if hasattr(self.app, 'status_bar'):
            self.app.status_bar.set("学习曲线生成失败", None)

    def _learning_curve_worker(self, model, X, y, mn, col):
        try:
            from sklearn.model_selection import learning_curve
            from joblib import parallel_backend

            with parallel_backend('threading'):
                train_sizes, train_scores, test_scores = learning_curve(
                    model,
                    X,
                    y,
                    cv=3,
                    n_jobs=1,
                    train_sizes=np.linspace(0.2, 1.0, 5),
                    scoring='r2',
                )
        except Exception as exc:
            import traceback

            err_msg = traceback.format_exc()
            print("Learning Curve Error:\n", err_msg)
            self._safe_after(lambda: self._learning_curve_failed(f"生成学习曲线失败:\n{exc}"))
            return

        self._safe_after(
            lambda: self._render_learning_curve(mn, col, train_sizes, train_scores, test_scores)
        )

    def _learning_curve_worker_safe(self, model, X, y, mn, col):
        try:
            from sklearn.model_selection import learning_curve
            from joblib import parallel_backend

            with parallel_backend('threading'):
                train_sizes, train_scores, test_scores = learning_curve(
                    model,
                    X,
                    y,
                    cv=3,
                    n_jobs=1,
                    train_sizes=np.linspace(0.2, 1.0, 5),
                    scoring='r2',
                )
        except Exception as exc:
            logger.exception("Failed to generate learning curve")
            message = self._runtime_error_message("生成学习曲线失败:", str(exc))
            self._safe_after(lambda m=message: self._learning_curve_failed(m))
            return

        self._safe_after(
            lambda: self._render_learning_curve(mn, col, train_sizes, train_scores, test_scores)
        )

    def _start_learning_curve(self):
        if self._learning_curve_running:
            show_message(self.app, "ℹ️", "学习曲线正在后台生成，请稍候。", "info")
            return

        mn = self.viz_model.get()
        if not hasattr(self.app, 'model_cache') or not hasattr(self.app, 'X_cache'):
            show_message(self.app, "ℹ️", "请先运行分析以生成模型缓存", "info")
            return

        if mn == 'GAM':
            show_message(self.app, "ℹ️", "GAM 模型不支持标准学习曲线绘制", "info")
            return

        if mn == 'Stacking':
            show_message(self.app, "ℹ️", "Stacking 融合模型计算复杂度极高，不支持生成学习曲线。", "warning")
            return

        key = self._select_model_key(mn)
        if key is None:
            show_message(self.app, "ℹ️", f"模型 {mn} 没有缓存，请确认分析已运行", "info")
            return

        model = self.app.model_cache[key]
        col = key[1]
        cached_X = self.app.X_cache.get((mn, col), self.app.X_cache.get(col))
        if cached_X is None:
            show_message(self.app, "ℹ️", "缺少该模型的特征缓存，请重新运行分析", "info")
            return

        if self.app.df is None:
            show_message(self.app, "❌", "请先加载数据", "error")
            return

        feature_specs = self._resolve_cached_feature_specs(cached_X)
        try:
            df_work, X_all, _, _, _ = prepare_feature_frame(self.app.df, feature_specs)
        except KeyError as exc:
            show_message(self.app, "❌", f"学习曲线所需的特征列不存在:\n{exc}", "error")
            return

        if col not in df_work.columns:
            show_message(self.app, "❌", "找不到对应的目标变量列", "error")
            return

        valid_idx = df_work[col].notna()
        X = X_all.loc[valid_idx].reset_index(drop=True)
        y_valid = pd.to_numeric(df_work.loc[valid_idx, col], errors='coerce').to_numpy()

        if hasattr(self.app, '_log_transformed') and self.app._log_transformed:
            y = y_valid
        else:
            y = np.log10(y_valid + 1)

        if len(X) != len(y) or len(X) == 0:
            show_message(self.app, "ℹ️", "当前目标变量在有效样本过滤后无法生成学习曲线。", "info")
            return

        self._learning_curve_running = True
        self._set_generate_button_state(False)
        self._clear()

        if hasattr(self.app, 'status_bar'):
            self.app.status_bar.set(f"正在后台生成 {mn} 的学习曲线...", None)

        threading.Thread(
            target=self._learning_curve_worker_safe,
            args=(model, X, y, mn, col),
            daemon=True,
        ).start()

    def _learning_curve(self):
        return self._start_learning_curve()

        mn = self.viz_model.get()
        if not hasattr(self.app, 'model_cache') or not hasattr(self.app, 'X_cache'):
            show_message(self.app, "ℹ️", "请先运行分析以生成模型缓存", "info")
            return
            
        key = self._select_model_key(mn)
        if key is None:
            show_message(self.app, "ℹ️", f"模型 {mn} 没有缓存，请确认分析已运行", "info")
            return
        model = self.app.model_cache[key]
        col = key[1]
        cached_X = self.app.X_cache.get((mn, col), self.app.X_cache.get(col))
        if cached_X is None:
            show_message(self.app, "ℹ️", "缺少该模型的特征缓存，请重新运行分析", "info")
            return
        
        # 重新获取 y 数据 (经过 log10 处理)
        if self.app.df is None:
            show_message(self.app, "鉂?", "璇峰厛鍔犺浇鏁版嵁", "error")
            return

        feature_specs = []
        for feature_name in cached_X.columns.tolist():
            feature_info = getattr(self.app, 'feature_vars', {}).get(feature_name, {})
            feature_type_var = feature_info.get('type')
            if feature_type_var is not None:
                feature_type = feature_type_var.get()
            elif pd.api.types.is_numeric_dtype(cached_X[feature_name]):
                feature_type = 'numeric'
            else:
                feature_type = 'categorical'
            feature_specs.append((feature_name, feature_type))

        try:
            df_work, X_all, _, _, _ = prepare_feature_frame(self.app.df, feature_specs)
        except KeyError as exc:
            show_message(self.app, "❌", f"学习曲线所需的特征列不存在:\n{exc}", "error")
            return
            show_message(self.app, "鉂?", f"瀛︿範鏇茬嚎鎵€闇€鐨勭壒寰佸垪涓嶅瓨鍦?:\n{exc}", "error")
            return
        if col not in df_work.columns:
            show_message(self.app, "❌", "找不到对应的目标变量列", "error")
            return
            
        # 这里需要用 X 对应的 index，因为 X 可能由于 dropna 被过滤过了
        valid_idx = df_work[col].notna()
        X = X_all.loc[valid_idx].reset_index(drop=True)
        y_valid = pd.to_numeric(df_work.loc[valid_idx, col], errors='coerce').to_numpy()
        
        if hasattr(self.app, '_log_transformed') and self.app._log_transformed:
            y = y_valid
        else:
            y = np.log10(y_valid + 1)
        
        # 对齐样本数 (以防 X 和 y 不一致)
        if len(X) != len(y) or len(X) == 0:
            # 简化处理：这部分在分析时是做了 notna() 过滤的
            show_message(self.app, "ℹ️", "由于缺失值过滤，X和y维度不一致，请在预处理阶段先处理缺失值", "info")
            return

        self._clear()
        import matplotlib.pyplot as plt
        from sklearn.model_selection import learning_curve
        
        fig = Figure(figsize=(10, 6), dpi=90, facecolor=C["bg_primary"])
        ax = fig.add_subplot(111)
        
        try:
            # GAM 不支持 scikit-learn 的 clone() 机制，无法画学习曲线
            if mn == 'GAM':
                show_message(self.app, "ℹ️", "GAM 模型不支持标准学习曲线绘制", "info")
                return
                
            # Stacking 绘制学习曲线极其耗时，通常会导致界面卡死，给出提示或简化处理
            if mn == 'Stacking':
                show_message(self.app, "ℹ️", "Stacking 融合模型计算复杂度极高，不支持生成学习曲线。", "warning")
                return
                
            # 由于使用了 Pipeline，如果有分类特征字符串，需要保证 cv 切分时各类都在
            # learning_curve 直接传 Pipeline 进去是兼容的
            
            # 计算学习曲线 (为了速度，设置 cv=3, 5个训练点)
            # 因为我们在 tab_analysis 里面使用了独立进程防止死锁
            # learning_curve 如果 n_jobs=-1 可能会再次触发多进程，所以我们使用多线程 backend
            from joblib import parallel_backend
            with parallel_backend('threading'):
                train_sizes, train_scores, test_scores = learning_curve(
                    model, X, y, cv=3, n_jobs=1, # 强制单进程/多线程，避免 loky 嵌套崩溃
                    train_sizes=np.linspace(0.2, 1.0, 5),
                    scoring='r2'
                )
            
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            
            ax.grid(True, linestyle='--', alpha=0.6, color=C["border"])
            ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1, color=C["error"])
            ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1, color=C["success"])
                            
            ax.plot(train_sizes, train_scores_mean, 'o-', color=C["error"], label="训练集得分 (R²)")
            ax.plot(train_sizes, test_scores_mean, 'o-', color=C["success"], label="交叉验证得分 (R²)")
            
            ax.set_title(f"{mn} ({col}) - 学习曲线", fontsize=14, color=C["text_primary"], pad=15)
            ax.set_xlabel("训练样本数量", fontsize=12, color=C["text_primary"])
            ax.set_ylabel("得分 (R²)", fontsize=12, color=C["text_primary"])
            ax.set_facecolor(C["bg_primary"])
            ax.tick_params(colors=C["text_muted"])
            
            for sp in ax.spines.values():
                sp.set_color(C["border"])
                
            ax.legend(loc="best", facecolor=C["bg_primary"], edgecolor=C["border"], labelcolor=C["text_primary"])
            fig.tight_layout()
            self._embed(fig)
            
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            show_message(self.app, "❌", f"生成学习曲线失败:\n{e}", "error")
            print("Learning Curve Error:\n", err_msg)

    # ── 特征重要性 ──

    def _importance(self):
        ar = self.app.analysis_results
        if not ar:
            show_message(self.app, "ℹ️", "请先运行分析", "info")
            return
        mn = self.viz_model.get()
        if mn not in ar:
            show_message(self.app, "ℹ️", f"模型 {mn} 无结果\n可用: {', '.join(ar)}", "info")
            return
        df = ar[mn]
        selected_target = self._selected_target()
        if selected_target is not None:
            df = df[df['目标变量'] == selected_target]
            if df.empty:
                show_message(self.app, "ℹ️", f"{mn} 没有目标变量 {selected_target} 的结果", "info")
                return
        ri_c = [c for c in df.columns if c.endswith('_RI(%)')]
        if not ri_c:
            return
        names = [c.replace('_RI(%)', '') for c in ri_c]
        vals = [df[c].mean() for c in ri_c]
        pairs = sorted(zip(names, vals), key=lambda p: p[1])
        self._clear()
        fig = Figure(figsize=(10, max(5, len(pairs) * 0.4)), dpi=90,
                     facecolor=C["bg_primary"])
        ax = fig.add_subplot(111)
        bars = ax.barh(range(len(pairs)), [p[1] for p in pairs],
                       color=C["accent"], edgecolor=C["bg_primary"], height=0.55)
        for b, v in zip(bars, [p[1] for p in pairs]):
            ax.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2,
                    f'{v:.1f}%', va='center', fontsize=9, color=C["text_primary"])
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels([p[0] for p in pairs], fontsize=9, color=C["text_primary"])
        ax.set_xlabel('相对重要性 RI(%)', fontsize=11, color=C["text_primary"])
        title = f'{mn} — 特征重要性排序'
        if selected_target is not None:
            title = f'{mn} — {selected_target} 特征重要性排序'
        ax.set_title(title, fontsize=13, color=C["text_primary"], pad=10)
        ax.set_facecolor(C["bg_primary"])
        ax.tick_params(colors=C["text_muted"])
        for sp in ax.spines.values():
            sp.set_color(C["border"])
        fig.tight_layout()
        self._embed(fig)

    # ── 模型性能对比 ──

    def _comparison(self):
        if self.app.performance_df is None and not self.app.analysis_results:
            show_message(self.app, "ℹ️", "请先运行分析", "info")
            return
        selected_target = self._selected_target()
        dp = self._metric_df() if selected_target is not None else self.app.performance_df
        if dp is None or dp.empty:
            show_message(self.app, "ℹ️", "当前筛选条件下没有可展示的性能结果", "info")
            return
        models = dp['模型'].tolist()
        self._clear()
        fig = Figure(figsize=(10, 6), dpi=90, facecolor=C["bg_primary"])
        ax = fig.add_subplot(111)
        x = np.arange(len(models))
        w = 0.22
        colors = [C["accent"], C["success"], C["info"]]
        for i, met in enumerate(['平均R²', '平均SCC', '平均PCC']):
            if met in dp.columns:
                vs = dp[met].tolist()
                brs = ax.bar(x + i * w, vs, w, label=met, color=colors[i],
                             edgecolor=C["bg_primary"])
                for b, v in zip(brs, vs):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                            f'{v:.3f}', ha='center', va='bottom', fontsize=8,
                            color=C["text_primary"])
        ax.set_xticks(x + w)
        ax.set_xticklabels(models, fontsize=10, color=C["text_primary"])
        ax.set_ylabel('Score', fontsize=11, color=C["text_primary"])
        title = '模型性能对比'
        if selected_target is not None:
            title = f'{selected_target} - 模型性能对比'
        ax.set_title(title, fontsize=14, color=C["text_primary"], pad=10)
        ax.legend(fontsize=9, facecolor=C["bg_primary"],
                  edgecolor=C["border"], labelcolor=C["text_primary"])
        ax.set_facecolor(C["bg_primary"])
        ax.tick_params(colors=C["text_muted"])
        for sp in ax.spines.values():
            sp.set_color(C["border"])
        fig.tight_layout()
        self._embed(fig)

    # ── 预测散点图 ──

    def _scatter(self):
        pc = self.app.prediction_cache
        if not pc:
            show_message(self.app, "ℹ️", "请先运行分析", "info")
            return
        mn = self.viz_model.get()
        selected_target = self._selected_target()
        matching = [(k, v) for k, v in pc.items() if k[0] == mn]
        if selected_target is not None:
            matching = [(k, v) for k, v in matching if k[1] == selected_target]
        if not matching:
            avail = set(k[0] for k in pc)
            show_message(self.app, "ℹ️",
                         f"{mn} 无缓存\n可用: {', '.join(avail)}", "info")
            return
        n = min(len(matching), 6)
        nc = min(3, n)
        nr = (n + nc - 1) // nc
        self._clear()
        fig = Figure(figsize=(4.8 * nc, 4.2 * nr), dpi=90, facecolor=C["bg_primary"])
        for idx, ((_, tgt), (yt, yp)) in enumerate(matching[:n]):
            ax = fig.add_subplot(nr, nc, idx + 1)
            ax.scatter(yt, yp, c=C["accent"], alpha=0.6, s=18, edgecolors=C["bg_primary"])
            lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
            mg = (hi - lo) * 0.05
            ax.plot([lo - mg, hi + mg], [lo - mg, hi + mg],
                    '--', color=C["success"], lw=1.5, alpha=0.8)
            if len(yt) > 1:
                corr = np.corrcoef(yt, yp)[0, 1]
                ax.text(0.05, 0.92, f'R={corr:.3f}', transform=ax.transAxes,
                        fontsize=8, color=C["text_primary"],
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=C["bg_secondary"], edgecolor=C["border"]))
            ax.set_xlabel('真实值', fontsize=9, color=C["text_primary"])
            ax.set_ylabel('预测值', fontsize=9, color=C["text_primary"])
            ax.set_title(tgt, fontsize=10, color=C["text_primary"], pad=6)
            ax.set_facecolor(C["bg_primary"])
            ax.tick_params(colors=C["text_muted"], labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(C["border"])
        fig.tight_layout(pad=2.0)
        self._embed(fig)

    # ── 雷达图 ──

    def _radar(self):
        if self.app.performance_df is None and not self.app.analysis_results:
            show_message(self.app, "ℹ️", "请先运行分析", "info")
            return
        selected_target = self._selected_target()
        dp = self._metric_df() if selected_target is not None else self.app.performance_df
        if dp is None or dp.empty:
            show_message(self.app, "ℹ️", "当前筛选条件下没有可展示的性能结果", "info")
            return
        models = dp['模型'].tolist()
        metrics = ['平均R²', '平均SCC', '平均PCC']
        
        # 确保所有指标都存在
        for m in metrics:
            if m not in dp.columns:
                show_message(self.app, "❌", f"缺少性能指标: {m}", "error")
                return

        self._clear()
        fig = Figure(figsize=(8, 8), dpi=90, facecolor=C["bg_primary"])
        ax = fig.add_subplot(111, polar=True)
        
        N = len(metrics)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11, color=C["text_primary"])
        ax.set_rlabel_position(0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color=C["text_muted"], size=8)
        ax.set_ylim(0, 1.05)
        
        # 预设几个颜色
        colors = ['#6366f1', '#34d399', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']
        
        for i, model in enumerate(models):
            values = dp.loc[dp['模型'] == model, metrics].values.flatten().tolist()
            values += values[:1]
            c = colors[i % len(colors)]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=c)
            ax.fill(angles, values, color=c, alpha=0.1)
            
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9, 
                  facecolor=C["bg_primary"], edgecolor=C["border"], labelcolor=C["text_primary"])
        title = "各模型性能指标雷达图"
        if selected_target is not None:
            title = f"{selected_target} - 模型性能指标雷达图"
        ax.set_title(title, size=14, color=C["text_primary"], pad=20)
        ax.set_facecolor(C["bg_primary"])
        
        fig.tight_layout()
        self._embed(fig)
