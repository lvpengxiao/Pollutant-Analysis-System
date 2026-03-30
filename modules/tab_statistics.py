"""
数据统计概览标签页模块
提供描述性统计、数据质量报告和分布直方图可视化。
"""

import logging
import numpy as np
import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .theme import (C, FONTS, show_message, make_btn_primary,
                    make_btn_secondary, make_textbox, make_scrollframe,
                    make_empty_state)

logger = logging.getLogger(__name__)


class StatisticsTab:

    def __init__(self, parent, app):
        self.app = app
        self.parent = parent
        self._canvas = None
        self._build()
        if hasattr(self.app, 'event_bus'):
            self.app.event_bus.subscribe('data_shape_changed', self.enable_buttons)

    def _build(self):
        bar = ctk.CTkFrame(self.parent, fg_color="transparent")
        bar.pack(fill="x", padx=18, pady=(14, 6))

        self.stat_btn = make_btn_primary(bar, text="📊  描述性统计", width=130,
                                         command=self._gen_stats)
        self.stat_btn.pack(side="left", padx=(0, 10))
        self.quality_btn = make_btn_secondary(bar, text="📋  数据质量报告", width=130,
                                              command=self._gen_quality)
        self.quality_btn.pack(side="left", padx=(0, 10))
        self.dist_btn = make_btn_secondary(bar, text="📈  分布直方图", width=120,
                                           command=self._plot_dist)
        self.dist_btn.pack(side="left", padx=(0, 10))
        self.kde_btn = make_btn_secondary(bar, text="🌊  KDE密度图", width=120,
                                          command=self._plot_kde)
        self.kde_btn.pack(side="left", padx=(0, 10))
        self.box_btn = make_btn_secondary(bar, text="📦  分布箱线图", width=120,
                                          command=self._plot_boxplot)
        self.box_btn.pack(side="left", padx=(0, 10))
        self.vif_btn = make_btn_secondary(bar, text="🔍  VIF 共线性诊断", width=140,
                                          command=self._calc_vif)
        self.vif_btn.pack(side="left", padx=(0, 10))
        self.export_btn = make_btn_secondary(bar, text="💾  导出基础数据", width=120,
                                             command=self._export_basic_stats)
        self.export_btn.pack(side="left")
        for btn in (self.stat_btn, self.quality_btn, self.dist_btn, self.kde_btn,
                    self.box_btn, self.vif_btn, self.export_btn):
            btn.configure(state="disabled")

        self.textbox = make_textbox(self.parent, wrap="none")
        self.textbox.pack(fill="both", expand=True, padx=18, pady=(6, 4))
        self.textbox.insert("1.0", "请点击上方按钮生成统计信息\n")

        # 将图表区域改为可滚动，防止特征过多时被压扁
        self.chart_frame = make_scrollframe(self.parent)
        self.chart_frame.pack(fill="both", expand=True, padx=18, pady=(4, 14))
        
        # 为了让图表能够在 ScrollFrame 内正确定位，再嵌套一个普通的 frame
        self.inner_chart = ctk.CTkFrame(self.chart_frame, fg_color="transparent")
        self.inner_chart.pack(fill="both", expand=True)
        self.empty_state = make_empty_state(
            self.inner_chart,
            "📊",
            "暂无统计图表",
            "请先在【📂 数据】页面导入数据，然后点击上方按钮生成描述统计、数据质量或分布图。",
            button_text="打开数据页",
            command=lambda: self.app.navigate_to_tab("data_load", force=True) if hasattr(self.app, 'navigate_to_tab') else None,
        )
        self.empty_state.pack(fill="both", expand=True, padx=6, pady=6)
        self.refresh_empty_state()

    def _require(self):
        if self.app.df is None:
            show_message(self.app, "❌ 错误", "请先加载数据", "error")
            return False
        return True

    def enable_buttons(self):
        for btn in (self.stat_btn, self.quality_btn, self.dist_btn, self.kde_btn,
                    self.box_btn, self.vif_btn, self.export_btn):
            btn.configure(state="normal")
        self.refresh_empty_state()

    def _show_empty_state(self):
        if self.empty_state is not None and not self.empty_state.winfo_manager():
            self.empty_state.pack(fill="both", expand=True, padx=6, pady=6)

    def _hide_empty_state(self):
        if self.empty_state is not None and self.empty_state.winfo_manager():
            self.empty_state.pack_forget()

    def refresh_empty_state(self):
        if self.app.df is None:
            self.empty_state.title_label.configure(text="暂无统计图表")
            self.empty_state.message_label.configure(
                text="请先在【📂 数据】页面导入数据，然后点击上方按钮生成描述统计、数据质量或分布图。"
            )
            if self.empty_state.action_button is not None:
                self.empty_state.action_button.configure(
                    text="打开数据页",
                    command=lambda: self.app.navigate_to_tab("data_load", force=True) if hasattr(self.app, 'navigate_to_tab') else None,
                )
                if not self.empty_state.action_button.winfo_manager():
                    self.empty_state.action_button.pack(pady=(0, 24))
        else:
            self.empty_state.title_label.configure(text="选择一种统计视图")
            self.empty_state.message_label.configure(
                text="数据已加载。点击上方按钮查看描述性统计、分布图或 VIF 共线性诊断。"
            )
            if self.empty_state.action_button is not None and self.empty_state.action_button.winfo_manager():
                self.empty_state.action_button.pack_forget()

    def _clear_chart(self):
        if self._canvas:
            import matplotlib.pyplot as plt
            fig = self._canvas.figure
            self._canvas.get_tk_widget().destroy()
            plt.close(fig)
            self._canvas = None
        for widget in self.inner_chart.winfo_children():
            if widget is not self.empty_state:
                widget.destroy()
        self._show_empty_state()

    def _export_basic_stats(self):
        if not self._require():
            return
            
        import pandas as pd
        from tkinter import filedialog
        
        df = self.app.df
        nc = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not nc:
            show_message(self.app, "ℹ️ 提示", "数据中没有数值型列，无法计算统计数据", "info")
            return
            
        fp = filedialog.asksaveasfilename(
            title="导出基础统计数据", 
            defaultextension=".xlsx",
            filetypes=[("Excel 文件", "*.xlsx")],
            initialfile="基础统计数据.xlsx"
        )
        if not fp:
            return
            
        selected_features = self.app.get_selected_features()
        target_cols = self.app.get_selected_targets()
            
        # 整体统计数据表仅针对用户勾选的特征和目标变量
        cols_to_stat = []
        if selected_features or target_cols:
            cols_to_stat = [c for c in nc if c in selected_features or c in target_cols]
        else:
            cols_to_stat = nc
            
        try:
            with pd.ExcelWriter(fp, engine='openpyxl') as writer:
                # 1. 整体基础统计表
                stats_list = []
                for col in cols_to_stat:
                    s = df[col].dropna()
                    if len(s) == 0: continue
                    stats_list.append({
                        '变量名': col,
                        '样本数': len(s),
                        '均值 (Mean)': round(s.mean(), 4),
                        '中位数 (Median)': round(s.median(), 4),
                        '标准差 (Std)': round(s.std(), 4),
                        '最小值 (Min)': round(s.min(), 4),
                        '最大值 (Max)': round(s.max(), 4),
                        '25%分位数 (Q1)': round(s.quantile(0.25), 4),
                        '75%分位数 (Q3)': round(s.quantile(0.75), 4),
                        '变异系数 (CV)': round(s.std() / s.mean() if s.mean() != 0 else 0, 4)
                    })
                df_overall = pd.DataFrame(stats_list)
                df_overall.to_excel(writer, sheet_name="整体统计数据", index=False)
                
                # 2. 如果用户在“目标”页定义了分类，生成分类汇总表
                if hasattr(self.app, 'category_vars') and self.app.category_vars:
                    cat_map = {}
                    for col, var in self.app.category_vars.items():
                        cat = var.get().strip()
                        if cat and col in nc:  # 只统计有分类标签的数值列
                            if cat not in cat_map:
                                cat_map[cat] = []
                            cat_map[cat].append(col)
                            
                    if cat_map:
                        cat_stats = []
                        for cat, cols in cat_map.items():
                            # 将该类别下所有物质的数值展平计算（或计算均值的均值等）
                            # 这里采用先对每个物质算均值，再求该类别的总体平均水平
                            means = [df[c].mean() for c in cols if not df[c].dropna().empty]
                            medians = [df[c].median() for c in cols if not df[c].dropna().empty]
                            
                            if means:
                                cat_stats.append({
                                    '物质类别': cat,
                                    '包含物质数量': len(cols),
                                    '类别平均浓度 (各类均值的均值)': round(np.mean(means), 4),
                                    '类别中位浓度 (各类中位的中位)': round(np.median(medians), 4),
                                    '包含的物质列表': ", ".join(cols)
                                })
                        if cat_stats:
                            df_cat = pd.DataFrame(cat_stats)
                            df_cat.to_excel(writer, sheet_name="分类物质汇总", index=False)
                
                # 3. 如果数据集中有离散型特征（如房间类型、季节），进行分组统计
                # 自动寻找可能的分类列（非数值型，且唯一值少于 15 个）
                cat_cols = [c for c in df.columns if c not in nc and 1 < df[c].nunique() < 15]
                for cat_col in cat_cols:
                    # 取前5个重要数值列进行分组展示
                    group_stat_cols = nc[:5]
                    grouped = df.groupby(cat_col)[group_stat_cols].mean().round(4).reset_index()
                    # 表名不能超过31字符
                    sheet_name = f"按_{cat_col}_分组均值"[:31]
                    grouped.to_excel(writer, sheet_name=sheet_name, index=False)
                    
            show_message(self.app, "✅ 导出成功", f"基础统计数据已成功导出至:\n{fp}", "info")
        except Exception as e:
            show_message(self.app, "❌ 导出失败", str(e), "error")

    def _calc_vif(self):
        if not self._require():
            return
            
        import pandas as pd
        import numpy as np
        
        df = self.app.df
        nc = df.select_dtypes(include=[np.number]).columns.tolist()
        
        selected_features = self.app.get_selected_features()
        target_cols = self.app.get_selected_targets()
            
        # 如果用户选了特征，就只分析选中的；如果没选，默认使用所有非目标的数值列
        if selected_features:
            feat_cols = [c for c in selected_features if c in nc and c not in target_cols]
        else:
            feat_cols = [c for c in nc if c not in target_cols]
        
        if len(feat_cols) < 2:
            show_message(self.app, "ℹ️ 提示", "特征数量少于2个，无法计算VIF", "info")
            return
            
        # 检查是否有 statsmodels
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            from statsmodels.tools.tools import add_constant
        except ImportError:
            show_message(self.app, "❌ 缺少依赖", "请先在终端运行: pip install statsmodels", "error")
            return
            
        self.textbox.delete("1.0", "end")
        self.textbox.insert("end", f"🔍 正在计算 {len(feat_cols)} 个特征的方差膨胀因子 (VIF)...\n\n")
        self.textbox.update()
        
        X = df[feat_cols].dropna()
        if len(X) == 0:
            self.textbox.insert("end", "❌ 错误: 选定特征中存在缺失值导致有效样本为0，请先在【预处理】页处理缺失值。")
            return
            
        # VIF 计算需要截距项
        X_with_const = add_constant(X)
        
        vif_data = []
        for i, col in enumerate(X_with_const.columns):
            if col == 'const': continue
            try:
                vif = variance_inflation_factor(X_with_const.values, i)
            except Exception as e:
                logger.warning(f"特征 {col} 的 VIF 计算失败: {e}，设为 inf")
                vif = np.inf
            vif_data.append((col, vif))
            
        vif_df = pd.DataFrame(vif_data, columns=['特征', 'VIF']).sort_values('VIF', ascending=False)
        
        self.textbox.insert("end", f"{'特征名称':<30} | {'VIF值':<15} | 诊断结果\n")
        self.textbox.insert("end", "-" * 70 + "\n")
        
        for _, row in vif_df.iterrows():
            feat = row['特征']
            vif = row['VIF']
            if np.isinf(vif):
                vif_str = "Inf"
                status = "🔴 极度共线"
            else:
                vif_str = f"{vif:.2f}"
                if vif > 10: status = "🔴 严重共线 (建议剔除)"
                elif vif > 5: status = "🟡 中度共线 (需关注)"
                else: status = "🟢 正常"
                
            self.textbox.insert("end", f"{feat:<30} | {vif_str:<15} | {status}\n")
            
        self.textbox.insert("end", "\n💡 提示: 在环境数据分析中，VIF > 10 通常被认为存在严重的多重共线性，建议在【预处理】页面剔除或降维。")

    def _gen_stats(self):
        if not self._require():
            return
        df = self.app.df
        self.textbox.delete("1.0", "end")
        desc = df.describe(include='all').round(4)
        info = "📊  描述性统计分析\n" + "═" * 80 + "\n\n"
        info += desc.to_string() + "\n\n"
        nc = df.select_dtypes(include=[np.number]).columns
        if len(nc) > 0:
            info += "═" * 80 + "\n📈  各数值列详细统计\n" + "─" * 80 + "\n"
            for col in nc:
                s = df[col].dropna()
                if len(s) == 0:
                    continue
                info += f"\n▶ {col}:\n"
                info += f"   均值={s.mean():.4f}   中位数={s.median():.4f}   "
                info += f"标准差={s.std():.4f}\n"
                info += f"   最小={s.min():.4f}   最大={s.max():.4f}   "
                info += f"偏度={s.skew():.4f}   峰度={s.kurtosis():.4f}\n"
                info += f"   Q1={s.quantile(.25):.4f}   "
                info += f"Q3={s.quantile(.75):.4f}   "
                info += f"IQR={s.quantile(.75)-s.quantile(.25):.4f}\n"
        self.textbox.insert("1.0", info)

    def _gen_quality(self):
        if not self._require():
            return
        df = self.app.df
        total = len(df)
        self.textbox.delete("1.0", "end")
        info = "📋  数据质量报告\n" + "═" * 92 + "\n\n"
        info += f"{'列名':25s} {'类型':10s} {'缺失':>7s} {'缺失率':>7s} "
        info += f"{'唯一值':>7s} {'最常见值':>20s}\n"
        info += "─" * 92 + "\n"
        for col in df.columns:
            dt = str(df[col].dtype)
            nm = df[col].isnull().sum()
            mp = f"{nm/total*100:.1f}%"
            nu = df[col].nunique()
            mode = df[col].mode()
            tv = str(mode.iloc[0]) if len(mode) > 0 else "N/A"
            if len(tv) > 18:
                tv = tv[:15] + "…"
            info += f"{col:25s} {dt:10s} {nm:7d} {mp:>7s} {nu:7d} {tv:>20s}\n"
        info += "─" * 92 + "\n"
        info += f"总行数: {total}   总列数: {len(df.columns)}   "
        comp = (1 - df.isnull().sum().sum() / max(total * len(df.columns), 1)) * 100
        info += f"缺失总计: {df.isnull().sum().sum()}   完整度: {comp:.2f}%\n"
        self.textbox.insert("1.0", info)

    def _plot_dist(self):
        if not self._require():
            return
        
        # 仅针对选中的变量绘图
        df = self.app.df
        nc = df.select_dtypes(include=[np.number]).columns.tolist()
        
        target_cols = self.app.get_selected_targets()
        selected_features = self.app.get_selected_features()
                    
        cols = [c for c in nc if c in target_cols or c in selected_features]
        if not cols:
            cols = nc
            
        if not cols:
            show_message(self.app, "ℹ️ 提示", "没有数值型列", "info")
            return

        self._clear_chart()

        # 如果变量太多，最多只画前 24 个，防止卡死
        cols = cols[:24]
        n = len(cols)
        ncols_g = min(4, n)
        nrows = (n + ncols_g - 1) // ncols_g

        fig = Figure(figsize=(12, 2.8 * nrows), dpi=90, facecolor=C["bg_primary"])
        for i, col in enumerate(cols):
            ax = fig.add_subplot(nrows, ncols_g, i + 1)
            data = self.app.df[col].dropna().values
            ax.hist(data, bins=20, color=C["accent"], alpha=0.85,
                    edgecolor=C["bg_primary"])
            ax.set_title(col, fontsize=9, color=C["text_primary"], pad=4)
            ax.tick_params(colors=C["text_muted"], labelsize=7)
            ax.set_facecolor(C["bg_primary"])
            for sp in ax.spines.values():
                sp.set_color(C["border"])
        fig.tight_layout(pad=2.0)

        # 根据图表高度动态设置 inner_chart 的高度以支持滚动
        fig_height = max(400, int(2.8 * nrows * 90))
        self.inner_chart.configure(height=fig_height)
        self._hide_empty_state()
        self._canvas = FigureCanvasTkAgg(fig, master=self.inner_chart)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    def _plot_kde(self):
        if not self._require():
            return
            
        df = self.app.df
        nc = df.select_dtypes(include=[np.number]).columns.tolist()
        
        target_cols = self.app.get_selected_targets()
        selected_features = self.app.get_selected_features()
                    
        cols = [c for c in nc if c in target_cols or c in selected_features]
        if not cols:
            cols = nc
            
        if not cols:
            show_message(self.app, "ℹ️ 提示", "没有数值型列", "info")
            return

        self._clear_chart()

        cols = cols[:24]
        n = len(cols)
        ncols_g = min(4, n)
        nrows = (n + ncols_g - 1) // ncols_g

        fig = Figure(figsize=(12, 2.8 * nrows), dpi=90, facecolor=C["bg_primary"])
        from scipy.stats import gaussian_kde
        for i, col in enumerate(cols):
            ax = fig.add_subplot(nrows, ncols_g, i + 1)
            data = self.app.df[col].dropna().values
            if len(data) > 1 and np.std(data) > 0:
                kde = gaussian_kde(data)
                x_vals = np.linspace(min(data), max(data), 200)
                y_vals = kde(x_vals)
                ax.plot(x_vals, y_vals, color=C["accent"], lw=2)
                ax.fill_between(x_vals, y_vals, alpha=0.3, color=C["accent"])
            else:
                ax.text(0.5, 0.5, "数据无效", ha='center', va='center')
                
            ax.set_title(col, fontsize=9, color=C["text_primary"], pad=4)
            ax.tick_params(colors=C["text_muted"], labelsize=7)
            ax.set_facecolor(C["bg_primary"])
            for sp in ax.spines.values():
                sp.set_color(C["border"])
        fig.tight_layout(pad=2.0)

        fig_height = max(400, int(2.8 * nrows * 90))
        self.inner_chart.configure(height=fig_height)
        self._hide_empty_state()
        self._canvas = FigureCanvasTkAgg(fig, master=self.inner_chart)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    def _plot_boxplot(self):
        if not self._require():
            return
        nc = self.app.df.select_dtypes(include=[np.number]).columns.tolist()
        if not nc:
            show_message(self.app, "ℹ️ 提示", "没有数值型列", "info")
            return

        self._clear_chart()

        cols = nc[:12]
        n = len(cols)
        ncols_g = min(4, n)
        nrows = (n + ncols_g - 1) // ncols_g

        fig = Figure(figsize=(12, 2.8 * nrows), dpi=90, facecolor=C["bg_primary"])
        for i, col in enumerate(cols):
            ax = fig.add_subplot(nrows, ncols_g, i + 1)
            data = self.app.df[col].dropna().values
            box = ax.boxplot(data, vert=False, patch_artist=True)
            for patch in box['boxes']:
                patch.set_facecolor(C["accent"])
                patch.set_alpha(0.7)
            for median in box['medians']:
                median.set_color(C["error"])
                median.set_linewidth(2)
            ax.set_title(col, fontsize=9, color=C["text_primary"], pad=4)
            ax.tick_params(colors=C["text_muted"], labelsize=7)
            ax.set_facecolor(C["bg_primary"])
            for sp in ax.spines.values():
                sp.set_color(C["border"])
        fig.tight_layout(pad=2.0)

        fig_height = max(400, int(2.8 * nrows * 90))
        self.inner_chart.configure(height=fig_height)
        self._hide_empty_state()
        self._canvas = FigureCanvasTkAgg(fig, master=self.inner_chart)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
