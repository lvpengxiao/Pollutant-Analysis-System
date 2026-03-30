"""
管理器模块
包含自定义菜单栏、状态栏、配置管理器和导出管理器。
"""

import logging
import os
import sys
import json
from datetime import datetime

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import customtkinter as ctk
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from .theme import (C, FONTS, APP_NAME, VERSION, show_message,
                    make_btn_primary, make_btn_secondary, make_checkbox)
from .reporting import reproducibility_dataframe


# ═══════════════════════════════════════════════════════════════
#  自定义菜单栏（暗色风格，不使用系统原生 Menu）
# ═══════════════════════════════════════════════════════════════

class CustomMenuBar(ctk.CTkFrame):

    def __init__(self, parent, app, **kw):
        super().__init__(parent, height=34, corner_radius=0,
                         fg_color=C["bg_primary"], **kw)
        self.pack_propagate(False)
        self.app = app
        self._dropdown = None
        self._active_key = None
        self._menus = {}

    def add_menu(self, label, items):
        btn = ctk.CTkButton(
            self, text=label, width=58, height=28,
            corner_radius=6, fg_color="transparent",
            hover_color=C["bg_hover"],
            text_color=C["text_secondary"],
            font=FONTS["body"](),
            command=lambda l=label: self._toggle(l))
        btn.pack(side="left", padx=2, pady=3)
        self._menus[label] = {'button': btn, 'items': items}

    def _toggle(self, key):
        if self._dropdown:
            try:
                self._dropdown.destroy()
            except Exception as e:
                logger.debug(f"销毁下拉菜单失败: {e}")
            self._dropdown = None
            if self._active_key == key:
                self._active_key = None
                return
        self._active_key = key
        info = self._menus[key]
        btn = info['button']
        items = info['items']

        dd = ctk.CTkToplevel(self.app)
        dd.overrideredirect(True)
        dd.attributes('-topmost', True)
        dd.configure(fg_color=C["bg_tertiary"])

        frame = ctk.CTkFrame(dd, corner_radius=10, fg_color=C["bg_tertiary"],
                              border_width=1, border_color=C["border_light"])
        frame.pack(padx=0, pady=0)

        for item in items:
            if item is None:
                ctk.CTkFrame(frame, height=1, fg_color=C["border"]).pack(
                    fill="x", padx=10, pady=4)
            else:
                lbl, cmd = item
                ctk.CTkButton(
                    frame, text=lbl, anchor="w", height=32,
                    fg_color="transparent", hover_color=C["bg_hover"],
                    text_color=C["text_primary"], font=FONTS["body"](),
                    corner_radius=6,
                    command=lambda c=cmd: self._exec(c)
                ).pack(fill="x", padx=4, pady=1)

        dd.update_idletasks()
        x = btn.winfo_rootx()
        y = btn.winfo_rooty() + btn.winfo_height() + 2
        w = max(frame.winfo_reqwidth(), 200)
        h = frame.winfo_reqheight()
        dd.geometry(f"{w}x{h}+{x}+{y}")
        self._dropdown = dd
        dd.bind('<Leave>', lambda e: self.app.after(400, self._auto_close))

    def _auto_close(self):
        if self._dropdown is None:
            return
        try:
            if not self._dropdown.winfo_exists():
                return
            px, py = self._dropdown.winfo_pointerx(), self._dropdown.winfo_pointery()
            wx, wy = self._dropdown.winfo_rootx(), self._dropdown.winfo_rooty()
            ww, wh = self._dropdown.winfo_width(), self._dropdown.winfo_height()
            if not (wx <= px <= wx + ww and wy <= py <= wy + wh):
                self._dropdown.destroy()
                self._dropdown = None
                self._active_key = None
        except Exception as e:
            logger.debug(f"下拉菜单位置检查失败: {e}")
            self._dropdown = None

    def _exec(self, cmd):
        if self._dropdown:
            try:
                self._dropdown.destroy()
            except Exception as e:
                logger.debug(f"销毁下拉菜单失败: {e}")
            self._dropdown = None
            self._active_key = None
        cmd()


# ═══════════════════════════════════════════════════════════════
#  状态栏
# ═══════════════════════════════════════════════════════════════

class StatusBar(ctk.CTkFrame):

    def __init__(self, parent, app, **kw):
        super().__init__(parent, height=28, corner_radius=0,
                         fg_color=C["bg_primary"], **kw)
        self.pack_propagate(False)
        self.app = app

        self.file_lbl = ctk.CTkLabel(
            self, text="未加载数据", font=FONTS["small"](),
            text_color=C["text_muted"])
        self.file_lbl.pack(side="left", padx=16)

        self.shape_lbl = ctk.CTkLabel(
            self, text="", font=FONTS["small"](),
            text_color=C["text_muted"])
        self.shape_lbl.pack(side="left", padx=16)

        self.clock_lbl = ctk.CTkLabel(
            self, text="", font=FONTS["small"](),
            text_color=C["text_muted"])
        self.clock_lbl.pack(side="right", padx=16)
        self._tick()

    def _tick(self):
        self.clock_lbl.configure(text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.after(1000, self._tick)

    def set(self, text, progress=None):
        """由分析线程调用，在底部状态栏显示实时信息"""
        try:
            self.file_lbl.configure(text=f"⏳ {text}", text_color=C["warning"])
        except Exception as e:
            logger.debug(f"状态栏更新失败: {e}")

    def refresh(self):
        if self.app.df is not None:
            fn = os.path.basename(self.app.file_path) if self.app.file_path else "未知"
            self.file_lbl.configure(text=f"📁 {fn}", text_color=C["text_secondary"])
            self.shape_lbl.configure(
                text=f"行: {self.app.df.shape[0]}   列: {self.app.df.shape[1]}")
        else:
            self.file_lbl.configure(text="未加载数据", text_color=C["text_muted"])
            self.shape_lbl.configure(text="")


# ═══════════════════════════════════════════════════════════════
#  配置管理器
# ═══════════════════════════════════════════════════════════════

class ConfigManager:

    def __init__(self, app):
        self.app = app
        self.path = os.path.join(os.path.expanduser('~'),
                                 '.pollutant_analysis_config.json')
        app.recent_files = []
        app.user_prefs = {'theme': 'Dark', 'scaling': 100, 'font_size': 13}
        self._load_prefs()

    def _load_prefs(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.app.recent_files = data.get('recent_files', [])
                p = data.get('preferences', {})
                if p:
                    self.app.user_prefs.update(p)
        except Exception as e:
            logger.warning(f"无法加载用户偏好设置: {e}")

    def _save_prefs(self):
        try:
            data = {'recent_files': self.app.recent_files[:10],
                    'preferences': self.app.user_prefs}
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"无法保存用户偏好设置: {e}")

    def add_recent(self, fp):
        fp = os.path.abspath(fp)
        if fp in self.app.recent_files:
            self.app.recent_files.remove(fp)
        self.app.recent_files.insert(0, fp)
        self.app.recent_files = self.app.recent_files[:10]
        self._save_prefs()

    def save_config(self):
        fp = filedialog.asksaveasfilename(
            title="保存分析配置", defaultextension=".json",
            filetypes=[("JSON", "*.json")])
        if not fp:
            return
        app = self.app
        cfg = {
            'version': VERSION,
            'timestamp': datetime.now().isoformat(),
            'data_file': app.file_path,
            'features': {c: {'selected': i['selected'].get(), 'type': i['type'].get()}
                         for c, i in app.feature_vars.items()},
            'targets': {c: {'selected': v.get(),
                            'category': app.category_vars.get(c, ctk.StringVar()).get()}
                        for c, v in app.target_vars.items()},
            'parameters': {
                'test_size': app.test_size_var.get(),
                'n_estimators': app.n_estimators_var.get(),
                'max_depth': app.max_depth_var.get(),
                'learning_rate': app.learning_rate_var.get(),
                'min_samples_split': app.min_samples_split_var.get(),
                'min_samples_leaf': app.min_samples_leaf_var.get(),
                'subsample': app.subsample_var.get(),
                'colsample': app.colsample_var.get(),
                'min_r2': app.min_r2_var.get(),
                'min_scc': app.min_scc_var.get(),
                'min_pcc': app.min_pcc_var.get(),
            },
            'models': {n: v.get() for n, v in app.model_vars.items()},
            'output_dir': app.output_dir_var.get(),
        }
        try:
            with open(fp, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            show_message(app, "✅ 已保存", f"配置已保存至:\n{fp}", "info")
        except Exception as e:
            show_message(app, "❌ 失败", str(e), "error")

    def load_config(self):
        fp = filedialog.askopenfilename(
            title="加载分析配置", filetypes=[("JSON", "*.json")])
        if not fp:
            return
        app = self.app
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            p = cfg.get('parameters', {})
            for k, var in [('test_size', 'test_size_var'), ('n_estimators', 'n_estimators_var'),
                           ('max_depth', 'max_depth_var'), ('learning_rate', 'learning_rate_var'),
                           ('min_samples_split', 'min_samples_split_var'),
                           ('min_samples_leaf', 'min_samples_leaf_var'),
                           ('subsample', 'subsample_var'), ('colsample', 'colsample_var'),
                           ('min_r2', 'min_r2_var'), ('min_scc', 'min_scc_var'),
                           ('min_pcc', 'min_pcc_var')]:
                if k in p and hasattr(app, var):
                    getattr(app, var).set(p[k])
            for n, en in cfg.get('models', {}).items():
                if n in app.model_vars:
                    app.model_vars[n].set(en)
            if cfg.get('output_dir'):
                app.output_dir_var.set(cfg['output_dir'])
            for c, i in cfg.get('features', {}).items():
                if c in app.feature_vars:
                    app.feature_vars[c]['selected'].set(i.get('selected', False))
                    app.feature_vars[c]['type'].set(i.get('type', 'numeric'))
            for c, i in cfg.get('targets', {}).items():
                if c in app.target_vars:
                    app.target_vars[c].set(i.get('selected', False))
                if c in app.category_vars:
                    app.category_vars[c].set(i.get('category', ''))
            show_message(app, "✅ 已加载", f"配置已加载:\n{fp}", "info")
        except Exception as e:
            show_message(app, "❌ 失败", str(e), "error")


# ═══════════════════════════════════════════════════════════════
#  导出管理器
# ═══════════════════════════════════════════════════════════════

class ExportManager:

    def __init__(self, app):
        self.app = app

    def _selected_features(self):
        return [
            col for col, info in self.app.feature_vars.items()
            if isinstance(info, dict) and info.get('selected') is not None and info['selected'].get()
        ]

    def _selected_targets(self):
        return [col for col, var in self.app.target_vars.items() if var.get()]

    def _enabled_models(self):
        return {name: var.get() for name, var in self.app.model_vars.items()}

    def _reproducibility_df(self, extra_rows=None):
        return reproducibility_dataframe(
            file_path=self.app.file_path,
            test_size=self.app.test_size_var.get() if hasattr(self.app, 'test_size_var') else None,
            preprocessing_applied=self.app._preprocessing_applied,
            log_transformed=self.app._log_transformed,
            selected_features=self._selected_features(),
            selected_targets=self._selected_targets(),
            enabled_models=self._enabled_models(),
            output_dir=self.app.output_dir_var.get() if hasattr(self.app, 'output_dir_var') else None,
            extra_rows=extra_rows,
        )

    def _report_sections_dialog(self):
        dialog = ctk.CTkToplevel(self.app)
        dialog.title("选择报告章节")
        dialog.geometry("460x470")
        dialog.transient(self.app)
        dialog.grab_set()

        ctk.CTkLabel(
            dialog,
            text="自定义 PDF 报告模板",
            font=FONTS["h2"](),
            text_color=C["text_primary"],
        ).pack(anchor="w", padx=20, pady=(18, 8))

        ctk.CTkLabel(
            dialog,
            text="勾选要包含在本次报告中的章节。",
            font=FONTS["body"](),
            text_color=C["text_secondary"],
        ).pack(anchor="w", padx=20, pady=(0, 12))

        section_defs = [
            ("cover", "封面"),
            ("reproducibility", "实验配置"),
            ("summary", "执行摘要"),
            ("performance_chart", "模型性能图表"),
            ("performance_table", "模型性能表"),
            ("target_pages", "目标变量摘要"),
            ("importance_pages", "模型重要性"),
            ("heatmap", "相关性热力图"),
        ]
        section_vars = {key: ctk.BooleanVar(value=True) for key, _ in section_defs}

        checkbox_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        checkbox_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        for key, label in section_defs:
            make_checkbox(checkbox_frame, text=label, variable=section_vars[key]).pack(anchor="w", pady=6)

        hint = ctk.CTkLabel(
            dialog,
            text="",
            font=FONTS["small"](),
            text_color=C["warning"],
        )
        hint.pack(anchor="w", padx=20, pady=(0, 8))

        result = {}

        def confirm():
            sections = {key: var.get() for key, var in section_vars.items()}
            if not any(sections.values()):
                hint.configure(text="至少选择一个章节。")
                return
            result["sections"] = sections
            dialog.destroy()

        button_row = ctk.CTkFrame(dialog, fg_color="transparent")
        button_row.pack(fill="x", padx=20, pady=(0, 18))
        make_btn_secondary(button_row, text="取消", width=110, command=dialog.destroy).pack(side="right", padx=(8, 0))
        make_btn_primary(button_row, text="继续导出", width=120, command=confirm).pack(side="right")

        self.app.wait_window(dialog)
        return result.get("sections")

    def _save_pdf_page(self, pdf, fig, page_no, section_title):
        fig.add_artist(Line2D([0.04, 0.96], [0.955, 0.955], transform=fig.transFigure, color="#d9d9d9", lw=0.8))
        fig.add_artist(Line2D([0.04, 0.96], [0.045, 0.045], transform=fig.transFigure, color="#d9d9d9", lw=0.8))
        fig.text(0.04, 0.972, APP_NAME, ha="left", va="top", fontsize=9, color="#6b7280")
        fig.text(0.96, 0.972, section_title, ha="right", va="top", fontsize=9, color="#6b7280")
        fig.text(0.5, 0.018, f"第 {page_no} 页", ha="center", va="bottom", fontsize=9, color="#6b7280")
        pdf.savefig(fig)
        plt.close(fig)
        return page_no + 1

    def _collect_target_summaries(self):
        summaries = []
        target_names = sorted({
            str(target)
            for df in self.app.analysis_results.values()
            if '目标变量' in df.columns
            for target in df['目标变量'].dropna().astype(str).tolist()
        })

        for target in target_names:
            candidates = []
            for model_name, df in self.app.analysis_results.items():
                if '目标变量' not in df.columns:
                    continue
                matched = df[df['目标变量'].astype(str) == target]
                if matched.empty:
                    continue
                candidates.append((model_name, matched.iloc[0]))

            if not candidates:
                continue

            def metric_value(row, key):
                value = row.get(key, np.nan)
                return float(value) if pd.notna(value) else float("-inf")

            best_model, best_row = max(
                candidates,
                key=lambda item: (
                    metric_value(item[1], 'SCC'),
                    metric_value(item[1], 'PCC'),
                    metric_value(item[1], 'R²'),
                ),
            )

            importance_pairs = []
            for column_name, value in best_row.items():
                if isinstance(column_name, str) and column_name.endswith('_RI(%)') and pd.notna(value):
                    importance_pairs.append((column_name.replace('_RI(%)', ''), float(value)))
            importance_pairs = sorted(importance_pairs, key=lambda item: item[1], reverse=True)[:5]

            summaries.append({
                "target": target,
                "category": best_row.get('类别', '未分类'),
                "best_model": best_model,
                "metrics": {
                    "R²": best_row.get('R²', np.nan),
                    "RMSE": best_row.get('RMSE', np.nan),
                    "SCC": best_row.get('SCC', np.nan),
                    "PCC": best_row.get('PCC', np.nan),
                },
                "importance_pairs": importance_pairs,
                "predictions": self.app.prediction_cache.get((best_model, target)),
                "candidate_count": len(candidates),
            })

        return summaries

    def export_csv(self):
        app = self.app
        if not app.analysis_results:
            show_message(app, "ℹ️", "请先运行分析", "info")
            return
        d = filedialog.askdirectory(title="选择 CSV 导出目录")
        if not d:
            return
        try:
            out = os.path.join(d, f"CSV_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(out, exist_ok=True)
            if app.spearman_results_df is not None:
                app.spearman_results_df.to_csv(
                    os.path.join(out, "01_Spearman相关性.csv"), index=False, encoding='utf-8-sig')
            for mn, df in app.analysis_results.items():
                df.to_csv(os.path.join(out, f"02_模型结果_{mn}.csv"), index=False, encoding='utf-8-sig')
            if app.comparison_df is not None:
                app.comparison_df.to_csv(
                    os.path.join(out, "03_综合比较.csv"), index=False, encoding='utf-8-sig')
            if app.performance_df is not None:
                app.performance_df.to_csv(
                    os.path.join(out, "04_模型性能汇总.csv"), index=False, encoding='utf-8-sig')
            summary_lines = [
                f"项目: {APP_NAME} v{VERSION}",
                f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"数据文件: {app.file_path or '未记录'}",
                f"模型数量: {len(app.analysis_results)}",
            ]
            if app.performance_df is not None and not app.performance_df.empty:
                best = app.performance_df.sort_values('平均SCC', ascending=False).iloc[0]
                summary_lines.append(
                    f"推荐关注模型: {best['模型']} (平均R²={best['平均R²']:.4f}, 平均SCC={best['平均SCC']:.4f}, 平均PCC={best['平均PCC']:.4f})"
                )
            with open(os.path.join(out, "00_导出说明.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(summary_lines))
            show_message(app, "✅ 导出成功", f"CSV 已导出至:\n{out}", "info")
        except Exception as e:
            show_message(app, "❌ 失败", str(e), "error")

    def export_pdf(self):
        app = self.app
        if not app.analysis_results:
            show_message(app, "ℹ️", "请先运行分析", "info")
            return
        selected_sections = self._report_sections_dialog()
        if not selected_sections:
            return
        fp = filedialog.asksaveasfilename(
            title="导出 PDF 报告", defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile=f"分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        if not fp:
            return
        try:
            section_labels = {
                "cover": "封面",
                "reproducibility": "实验配置",
                "summary": "执行摘要",
                "performance_chart": "模型性能图表",
                "performance_table": "模型性能表",
                "target_pages": "目标变量摘要",
                "importance_pages": "模型重要性",
                "heatmap": "相关性热力图",
            }
            chosen_sections = [
                section_labels[key]
                for key, enabled in selected_sections.items()
                if enabled and key in section_labels
            ]
            target_names = sorted({
                str(target)
                for df in app.analysis_results.values()
                if '目标变量' in df.columns
                for target in df['目标变量'].dropna().astype(str).tolist()
            })
            reproducibility_df = self._reproducibility_df(
                extra_rows=[
                    ("导出类型", "PDF 报告"),
                    ("报告章节", "、".join(chosen_sections) if chosen_sections else "未选择"),
                ]
            )
            target_summaries = self._collect_target_summaries() if selected_sections.get("target_pages") else []
            page_no = 1

            with PdfPages(fp) as pdf:
                if selected_sections.get("cover"):
                    fig = Figure(figsize=(11, 8.5), facecolor='white')
                    ax = fig.add_subplot(111)
                    ax.axis('off')
                    ax.text(0.5, 0.74, APP_NAME, fontsize=24, ha='center', fontweight='bold')
                    ax.text(0.5, 0.63, '分析报告', fontsize=18, ha='center')
                    ax.text(0.5, 0.52, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            fontsize=12, ha='center', color='#6b7280')
                    ax.text(
                        0.5, 0.46,
                        f"数据文件: {os.path.basename(app.file_path) if app.file_path else '未记录'}",
                        fontsize=11, ha='center', color='#6b7280'
                    )
                    ax.text(0.5, 0.40, f"报告章节: {len(chosen_sections)}", fontsize=11, ha='center', color='#6b7280')
                    ax.text(0.5, 0.24, f'v{VERSION}', fontsize=10, ha='center', color='#9ca3af')
                    page_no = self._save_pdf_page(pdf, fig, page_no, section_labels["cover"])

                if selected_sections.get("reproducibility"):
                    fig = Figure(figsize=(11, 8.5), facecolor='white')
                    ax = fig.add_subplot(111)
                    ax.axis('off')
                    ax.text(0.05, 0.92, "实验配置与可复现性记录", fontsize=18, fontweight='bold', va='top')
                    table = ax.table(
                        cellText=reproducibility_df.values.tolist(),
                        colLabels=reproducibility_df.columns.tolist(),
                        cellLoc='left',
                        colLoc='left',
                        bbox=[0.05, 0.08, 0.90, 0.78],
                    )
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1, 1.4)
                    page_no = self._save_pdf_page(pdf, fig, page_no, section_labels["reproducibility"])

                if selected_sections.get("summary") and app.performance_df is not None and not app.performance_df.empty:
                    perf = app.performance_df.sort_values('平均SCC', ascending=False).reset_index(drop=True)
                    best = perf.iloc[0]
                    fig = Figure(figsize=(11, 8.5), facecolor='white')
                    ax = fig.add_subplot(111)
                    ax.axis('off')
                    summary_lines = [
                        "执行摘要",
                        "",
                        f"数据文件: {os.path.basename(app.file_path) if app.file_path else '未记录'}",
                        f"参与比较模型数: {len(perf)}",
                        f"目标变量数: {len(target_names)}",
                        f"推荐优先关注模型: {best['模型']}",
                        f"平均R²: {best['平均R²']:.4f}",
                        f"平均SCC: {best['平均SCC']:.4f}",
                        f"平均PCC: {best['平均PCC']:.4f}",
                        "",
                        "解读建议:",
                        "1. 优先比较平均SCC与平均PCC更高的模型。",
                        "2. 若多个模型结果接近，可结合R²与解释性共同判断。",
                        "3. 目标变量摘要页可用于单个指标的汇报与审阅。"
                    ]
                    ax.text(0.08, 0.92, "\n".join(summary_lines), fontsize=15, va='top')
                    page_no = self._save_pdf_page(pdf, fig, page_no, section_labels["summary"])

                if selected_sections.get("performance_chart") and app.performance_df is not None and not app.performance_df.empty:
                    fig = Figure(figsize=(11, 6.8), facecolor='white')
                    ax = fig.add_subplot(111)
                    dp = app.performance_df.sort_values('平均SCC', ascending=False).reset_index(drop=True)
                    model_names = dp['模型'].tolist()
                    x = np.arange(len(model_names))
                    width = 0.22
                    colors = ['#2563eb', '#10b981', '#f59e0b']
                    for idx, metric in enumerate(['平均R²', '平均SCC', '平均PCC']):
                        if metric in dp.columns:
                            ax.bar(
                                x + idx * width - width,
                                dp[metric].tolist(),
                                width=width,
                                label=metric,
                                color=colors[idx]
                            )
                    ax.set_xticks(x)
                    ax.set_xticklabels(model_names)
                    ax.set_ylabel('Score')
                    ax.set_title('模型性能对比')
                    ax.grid(axis='y', alpha=0.25)
                    ax.legend(loc='upper right')
                    fig.tight_layout(rect=[0.03, 0.05, 0.97, 0.92])
                    page_no = self._save_pdf_page(pdf, fig, page_no, section_labels["performance_chart"])

                if selected_sections.get("performance_table") and app.performance_df is not None and not app.performance_df.empty:
                    fig = Figure(figsize=(11, 8.5), facecolor='white')
                    ax = fig.add_subplot(111)
                    ax.axis('off')
                    preferred_cols = ['模型', '目标变量数', '良好模型数', '良好比例(%)', '平均R²', '平均SCC', '平均PCC']
                    table_df = app.performance_df.copy()
                    table_df = table_df[[col for col in preferred_cols if col in table_df.columns]]
                    table = ax.table(
                        cellText=table_df.round(4).values.tolist(),
                        colLabels=table_df.columns.tolist(),
                        cellLoc='center',
                        colLoc='center',
                        bbox=[0.04, 0.10, 0.92, 0.76],
                    )
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1, 1.4)
                    ax.text(0.04, 0.90, "模型性能汇总表", fontsize=17, fontweight='bold', va='top')
                    page_no = self._save_pdf_page(pdf, fig, page_no, section_labels["performance_table"])

                if selected_sections.get("target_pages"):
                    for summary in target_summaries:
                        fig = Figure(figsize=(11, 8.5), facecolor='white')
                        grid = fig.add_gridspec(
                            2, 2,
                            height_ratios=[2.2, 1.0],
                            width_ratios=[1.7, 1.0],
                            left=0.06, right=0.96, top=0.88, bottom=0.10,
                            hspace=0.28, wspace=0.18,
                        )
                        ax_scatter = fig.add_subplot(grid[0, 0])
                        ax_importance = fig.add_subplot(grid[0, 1])
                        ax_text = fig.add_subplot(grid[1, :])
                        ax_text.axis('off')

                        predictions = summary.get("predictions")
                        if predictions is not None and len(predictions) == 2:
                            y_true = np.asarray(predictions[0], dtype=float)
                            y_pred = np.asarray(predictions[1], dtype=float)
                            valid = ~(np.isnan(y_true) | np.isnan(y_pred))
                            y_true = y_true[valid]
                            y_pred = y_pred[valid]
                            if len(y_true) > 0:
                                ax_scatter.scatter(
                                    y_true, y_pred,
                                    alpha=0.75, s=28, color='#2563eb', edgecolors='none'
                                )
                                data_min = float(min(np.min(y_true), np.min(y_pred)))
                                data_max = float(max(np.max(y_true), np.max(y_pred)))
                                ax_scatter.plot(
                                    [data_min, data_max], [data_min, data_max],
                                    '--', color='#ef4444', lw=1.4
                                )
                                ax_scatter.set_xlabel('Observed')
                                ax_scatter.set_ylabel('Predicted')
                                ax_scatter.set_title(f"{summary['target']} 预测散点图")
                                ax_scatter.grid(alpha=0.20)
                            else:
                                ax_scatter.text(0.5, 0.5, "无可用预测数据", ha='center', va='center')
                                ax_scatter.set_axis_off()
                        else:
                            ax_scatter.text(0.5, 0.5, "未找到预测输出", ha='center', va='center')
                            ax_scatter.set_axis_off()

                        importance_pairs = summary.get("importance_pairs") or []
                        if importance_pairs:
                            labels = [name for name, _ in importance_pairs][::-1]
                            values = [value for _, value in importance_pairs][::-1]
                            ax_importance.barh(labels, values, color='#10b981')
                            ax_importance.set_xlabel('RI(%)')
                            ax_importance.set_title('Top 5 重要性')
                            ax_importance.grid(axis='x', alpha=0.20)
                        else:
                            ax_importance.text(0.5, 0.5, "无可用重要性数据", ha='center', va='center')
                            ax_importance.set_axis_off()

                        metrics = summary.get("metrics", {})
                        metric_lines = [
                            f"目标变量: {summary['target']}",
                            f"类别: {summary.get('category', '未分类')}",
                            f"最佳模型: {summary.get('best_model', '未记录')}",
                            f"参与比较模型数: {summary.get('candidate_count', 0)}",
                            f"R²: {metrics.get('R²', np.nan):.4f}" if pd.notna(metrics.get('R²', np.nan)) else "R²: N/A",
                            f"RMSE: {metrics.get('RMSE', np.nan):.4f}" if pd.notna(metrics.get('RMSE', np.nan)) else "RMSE: N/A",
                            f"SCC: {metrics.get('SCC', np.nan):.4f}" if pd.notna(metrics.get('SCC', np.nan)) else "SCC: N/A",
                            f"PCC: {metrics.get('PCC', np.nan):.4f}" if pd.notna(metrics.get('PCC', np.nan)) else "PCC: N/A",
                        ]
                        ax_text.text(0.0, 0.95, "\n".join(metric_lines), va='top', fontsize=12)
                        page_no = self._save_pdf_page(pdf, fig, page_no, f"{section_labels['target_pages']} - {summary['target']}")

                if selected_sections.get("importance_pages"):
                    for model_name, df in app.analysis_results.items():
                        ri_columns = [col for col in df.columns if isinstance(col, str) and col.endswith('_RI(%)')]
                        if not ri_columns:
                            continue
                        names = [col.replace('_RI(%)', '') for col in ri_columns]
                        values = [float(df[col].mean()) if pd.notna(df[col].mean()) else 0.0 for col in ri_columns]
                        pairs = sorted(zip(names, values), key=lambda item: item[1])
                        fig = Figure(figsize=(11, max(5.5, len(pairs) * 0.38)), facecolor='white')
                        ax = fig.add_subplot(111)
                        ax.barh(range(len(pairs)), [item[1] for item in pairs], color='#2563eb', height=0.65)
                        ax.set_yticks(range(len(pairs)))
                        ax.set_yticklabels([item[0] for item in pairs], fontsize=9)
                        ax.set_xlabel('RI(%)')
                        ax.set_title(f'{model_name} 特征重要性汇总')
                        ax.grid(axis='x', alpha=0.25)
                        fig.tight_layout(rect=[0.03, 0.05, 0.97, 0.92])
                        page_no = self._save_pdf_page(pdf, fig, page_no, f"{section_labels['importance_pages']} - {model_name}")

                if selected_sections.get("heatmap") and app.df is not None:
                    numeric_columns = app.df.select_dtypes(include=[np.number]).columns[:15]
                    if len(numeric_columns) >= 2:
                        corr = app.df[numeric_columns].corr()
                        fig = Figure(figsize=(10.8, 8.2), facecolor='white')
                        ax = fig.add_subplot(111)
                        image = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
                        fig.colorbar(image, ax=ax, shrink=0.82)
                        ax.set_xticks(range(len(numeric_columns)))
                        ax.set_yticks(range(len(numeric_columns)))
                        ax.set_xticklabels(numeric_columns, rotation=45, ha='right', fontsize=8)
                        ax.set_yticklabels(numeric_columns, fontsize=8)
                        ax.set_title('数值列相关性热力图')
                        fig.tight_layout(rect=[0.03, 0.05, 0.97, 0.92])
                        page_no = self._save_pdf_page(pdf, fig, page_no, section_labels["heatmap"])

            show_message(app, "✅ 导出成功", f"PDF 已导出至:\n{fp}", "info")
        except Exception as e:
            show_message(app, "❌ 失败", str(e), "error")

    def export_log(self):
        app = self.app
        fp = filedialog.asksaveasfilename(
            title="导出日志", defaultextension=".log",
            filetypes=[("日志", "*.log"), ("文本", "*.txt")])
        if not fp:
            return
        try:
            tab = app.tabs.get('analysis')
            content = tab.log.get("1.0", "end").strip() if tab else ""
            if not content:
                show_message(app, "ℹ️", "日志为空", "info")
                return
            header = f"{'═' * 70}\n  {APP_NAME} v{VERSION}\n"
            header += f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            if app.file_path:
                header += f"  数据: {app.file_path}\n"
            header += f"{'═' * 70}\n\n"
            with open(fp, 'w', encoding='utf-8') as f:
                f.write(header + content)
            show_message(app, "✅ 导出成功", f"日志已导出至:\n{fp}", "info")
        except Exception as e:
            show_message(app, "❌ 失败", str(e), "error")
