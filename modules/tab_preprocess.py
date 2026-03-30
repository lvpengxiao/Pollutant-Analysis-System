"""
数据预处理标签页模块 (🧹 清洗)
---------------------------------------------------------------------
对原始数据集进行清洗和数学变换，为后续的机器学习任务准备高质量的数据。

提供工具:
1. 缺失值处理: 支持均值、中位数、众数、常数填补或直接删除。
2. 异常值处理: 基于 IQR / Z-Score 截断或移除极端值。
3. 数据缩放: 标准化 (Z-Score) 和 归一化 (MinMax)。
4. 正态性变换: Log10, Ln, Box-Cox 变换，可有效缓解偏态分布带来的影响。
   注: 变换仅针对用户选中的特征/目标执行，防止污染无关 ID 列。
5. 数据恢复: 提供一键恢复到初始状态的功能。
"""

import numpy as np
import pandas as pd
import customtkinter as ctk
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

from .theme import (C, FONTS, show_message, make_card, make_inner_frame,
                    make_section_title, make_hint, make_btn_primary,
                    make_btn_secondary, make_btn_warning, make_btn_danger,
                    make_entry, make_optionmenu, make_textbox)


class PreprocessTab:

    def __init__(self, parent, app):
        self.app = app
        self.parent = parent
        self.advanced_buttons = []
        self._build()
        if hasattr(self.app, 'event_bus'):
            self.app.event_bus.subscribe('data_shape_changed', self._on_data_shape_changed)

    def _on_data_shape_changed(self):
        self.refresh_columns()
        self.check_unlock()

    def refresh_columns(self):
        cols = self.app.all_columns if self.app.all_columns else ["(请先加载数据)"]
        self.filter_col_menu.configure(values=cols)
        if cols:
            self.filter_col_var.set(cols[0])

    # ── UI ────────────────────────────────────────────────

    def _build(self):
        self.scroll = ctk.CTkScrollableFrame(
            self.parent, fg_color="transparent",
            scrollbar_button_color=C["scrollbar"],
            scrollbar_button_hover_color=C["scrollbar_hover"])
        self.scroll.pack(fill="both", expand=True, padx=6, pady=6)

        ctk.CTkLabel(
            self.scroll,
            text="━━━  基础数据清洗（无需先选特征）  ━━━",
            font=FONTS["h3"](),
            text_color=C["accent_light"]
        ).pack(fill="x", padx=12, pady=(10, 4))

        self._build_backup_card(self.scroll)
        self._build_missing_card(self.scroll)
        self._build_outlier_card(self.scroll)
        self._build_normality_card(self.scroll)
        self._build_norm_card(self.scroll)
        self._build_filter_card(self.scroll)

        self.advanced_header = ctk.CTkLabel(
            self.scroll,
            text="━━━  高级特征工程（需先在 📋特征 页勾选变量）  ━━━",
            font=FONTS["h3"](),
            text_color=C["warning"]
        )
        self.advanced_header.pack(fill="x", padx=12, pady=(20, 4))

        self.lock_frame = make_card(self.scroll)
        self.lock_frame.pack(fill="x", padx=12, pady=6)
        self.lock_label = ctk.CTkLabel(
            self.lock_frame,
            text="🔒 请先在【📋 特征】页面勾选至少 2 个特征变量后，此区域自动解锁",
            font=FONTS["body"](),
            text_color=C["text_muted"]
        )
        self.lock_label.pack(padx=18, pady=20)

        self.advanced_container = ctk.CTkFrame(self.scroll, fg_color="transparent")
        self.advanced_container.pack(fill="x")
        self._build_dim_reduction_card(self.advanced_container)
        self.refresh_columns()
        self.check_unlock()

    def check_unlock(self):
        feats = self.app.get_selected_features() if hasattr(self.app, 'get_selected_features') else []
        targets = self.app.get_selected_targets() if hasattr(self.app, 'get_selected_targets') else []

        if len(feats) >= 2 and targets:
            self.advanced_header.configure(
                text=f"━━━  高级特征工程（已选 {len(feats)} 个特征，{len(targets)} 个目标）  ━━━",
                text_color=C["success"]
            )
            self.lock_label.configure(
                text="✅ 已满足高级特征工程条件，可执行共线性过滤、PCA 和交互特征生成。",
                text_color=C["success"]
            )
            for btn in self.advanced_buttons:
                btn.configure(state="normal")
            return

        if len(feats) >= 2:
            self.advanced_header.configure(
                text=f"━━━  高级特征工程（已选 {len(feats)} 个特征，建议补充目标变量）  ━━━",
                text_color=C["warning"]
            )
            self.lock_label.configure(
                text="⚠️ 已选足够特征，可继续执行高级特征工程；建议先在【🎯 目标】页勾选目标变量。",
                text_color=C["warning"]
            )
            for btn in self.advanced_buttons:
                btn.configure(state="normal")
            return

        self.advanced_header.configure(
            text="━━━  高级特征工程（需先在 📋特征 页勾选变量）  ━━━",
            text_color=C["warning"]
        )
        self.lock_label.configure(
            text="🔒 请先在【📋 特征】页面勾选至少 2 个特征变量后，此区域自动解锁",
            text_color=C["text_muted"]
        )
        for btn in self.advanced_buttons:
            btn.configure(state="disabled")

    # ── 备份/恢复 ──

    def _build_backup_card(self, parent):
        card = make_card(parent)
        card.pack(fill="x", padx=12, pady=6)
        row = make_inner_frame(card)
        row.pack(fill="x", padx=18, pady=14)

        make_section_title(row, "数据保护", icon="🔒").pack(side="left", padx=(0, 24))
        make_btn_secondary(row, text="创建备份", width=110,
                           command=self._backup).pack(side="left", padx=4)
        make_btn_warning(row, text="恢复备份", width=110,
                         command=self._restore).pack(side="left", padx=4)
        self.backup_label = ctk.CTkLabel(
            row, text="  尚未创建备份", font=FONTS["small"](),
            text_color=C["text_muted"])
        self.backup_label.pack(side="left", padx=16)

    # ── 缺失值 ──

    def _build_missing_card(self, parent):
        card = make_card(parent)
        card.pack(fill="x", padx=12, pady=6)

        make_section_title(card, "缺失值检测与处理", icon="🔍").pack(
            anchor="w", padx=18, pady=(16, 8))

        # 添加提示信息
        ctk.CTkLabel(card, text="⚠️ 注意：此处的处理仅为方便数据探索。分析引擎已内置 Pipeline 避免数据泄露。", 
                     font=FONTS["small"](), text_color=C["warning"]).pack(anchor="w", padx=18, pady=(0, 8))

        row = make_inner_frame(card)
        row.pack(fill="x", padx=18, pady=(0, 6))

        make_btn_secondary(row, text="检测缺失值", width=120,
                           command=self._detect_missing).pack(side="left", padx=(0, 12))

        ctk.CTkLabel(row, text="策略:", font=FONTS["body"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(0, 4))
        self.miss_strategy = ctk.StringVar(value="均值填充")
        make_optionmenu(
            row, variable=self.miss_strategy, width=150,
            values=["均值填充", "中位数填充", "众数填充", "删除缺失行", "线性插值"]
        ).pack(side="left", padx=4)

        make_btn_primary(row, text="执行处理", width=110,
                         command=self._handle_missing).pack(side="left", padx=12)

        self.miss_text = make_textbox(card, height=110)
        self.miss_text.pack(fill="x", padx=18, pady=(4, 16))

    # ── 异常值 ──

    def _build_outlier_card(self, parent):
        card = make_card(parent)
        card.pack(fill="x", padx=12, pady=6)

        make_section_title(card, "异常值检测", icon="📊").pack(
            anchor="w", padx=18, pady=(16, 8))

        row = make_inner_frame(card)
        row.pack(fill="x", padx=18, pady=(0, 6))

        ctk.CTkLabel(row, text="方法:", font=FONTS["body"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(0, 4))
        self.outlier_method = ctk.StringVar(value="IQR箱线图法")
        make_optionmenu(
            row, variable=self.outlier_method, width=150,
            values=["IQR箱线图法", "Z-score法"]
        ).pack(side="left", padx=4)

        ctk.CTkLabel(row, text="Z 阈值:", font=FONTS["body"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(16, 4))
        self.zscore_th = ctk.DoubleVar(value=3.0)
        make_entry(row, textvariable=self.zscore_th, width=60,
                   justify="center").pack(side="left", padx=4)

        make_btn_secondary(row, text="检测", width=80,
                           command=self._detect_outliers).pack(side="left", padx=8)
        make_btn_danger(row, text="移除异常值", width=120,
                        command=self._remove_outliers).pack(side="left", padx=4)

        self.outlier_text = make_textbox(card, height=110)
        self.outlier_text.pack(fill="x", padx=18, pady=(4, 16))

    # ── 正态性变换 ──

    def _build_normality_card(self, parent):
        card = make_card(parent)
        card.pack(fill="x", padx=12, pady=6)
        
        make_section_title(card, "正态性诊断与变换", icon="📈").pack(
            anchor="w", padx=18, pady=(16, 8))
            
        row = make_inner_frame(card)
        row.pack(fill="x", padx=18, pady=(0, 6))
        
        make_btn_secondary(row, text="Shapiro-Wilk 正态性检验", width=180,
                           command=self._test_normality).pack(side="left", padx=(0, 10))
                           
        ctk.CTkLabel(row, text="选择变换方法:", font=FONTS["body"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(10, 4))
                     
        self.transform_method = ctk.StringVar(value="Log10 (x+1)")
        make_optionmenu(row, variable=self.transform_method, width=140,
                        values=["Log10 (x+1)", "Ln (x+1)", "Box-Cox (要求x>0)"]).pack(side="left", padx=4)
                        
        make_btn_warning(row, text="执行变换 (仅数值列)", width=160,
                         command=self._apply_transform).pack(side="left", padx=10)
                         
        self.normality_text = make_textbox(card, height=120)
        self.normality_text.pack(fill="x", padx=18, pady=(4, 16))

    def _test_normality(self):
        if not self._require_data():
            return
            
        from scipy.stats import shapiro, skew
        df = self.app.df
        nc = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 获取用户勾选的目标变量和特征
        target_cols = self.app.get_selected_targets()
        selected_features = self.app.get_selected_features()
                    
        # 正态性检验主要关注连续型的数值特征和目标变量
        cols_to_test = []
        if target_cols or selected_features:
            cols_to_test = [c for c in nc if c in target_cols or c in selected_features]
        else:
            cols_to_test = nc
            
        if not cols_to_test:
            self.normality_text.insert("1.0", "没有选中的数值型列可检验\n")
            return
            
        self.normality_text.delete("1.0", "end")
        self.normality_text.insert("end", f"📌 正在对 {len(cols_to_test)} 个选中的变量进行检验\n")
        self.normality_text.insert("end", f"{'变量名':<25} | {'偏度(Skew)':<10} | {'Shapiro p值':<12} | 结论\n")
        self.normality_text.insert("end", "-" * 70 + "\n")
        
        for col in cols_to_test:
            data = df[col].dropna()
            if len(data) < 3: continue
            
            s_val = skew(data)
            # Shapiro 检验对样本量敏感，通常最大限制 5000
            stat, p = shapiro(data[:5000])
            
            is_normal = p > 0.05
            rec = "🟢 近似正态" if is_normal else ("🔴 严重右偏 (建议Log/BoxCox)" if s_val > 1 else "🟡 非正态")
            
            self.normality_text.insert("end", f"{col[:24]:<25} | {s_val:>10.2f} | {p:>12.2e} | {rec}\n")
            
        self.normality_text.insert("end", "\n💡 提示: 环境浓度数据通常呈现严重的右偏（偏度>1），进行对数变换或 Box-Cox 变换可显著提升线性/加性模型的拟合效果。")
        self.normality_text.see("1.0")

    def _apply_transform(self):
        if not self._require_data():
            return
            
        from scipy.stats import boxcox
        df = self.app.df
        nc = df.select_dtypes(include=[np.number]).columns.tolist()
        method = self.transform_method.get()

        # ── 修复：仅变换用户选中的特征和目标变量 ──
        selected_features = self.app.get_selected_features()
        target_cols = self.app.get_selected_targets()
        cols_to_transform = [c for c in nc if c in selected_features or c in target_cols]

        if not cols_to_transform:
            # 如果用户没有选任何特征/目标，给出明确提示
            show_message(self.app, "⚠️ 提示",
                         "请先在【📋 特征】或【🎯 目标】页面勾选要变换的变量。\n\n"
                         "为避免误变换 ID、编号等无关列，变换仅作用于已选中的变量。",
                         "warning")
            return

        self.normality_text.delete("1.0", "end")
        transformed_cols = []
        errors = []

        for col in cols_to_transform:
            data = df[col]
            if data.isnull().any():
                errors.append(f"{col}: 存在缺失值，请先处理")
                continue

            try:
                if method == "Log10 (x+1)":
                    if (data < -1).any():
                        errors.append(f"{col}: 存在 < -1 的值，log10(x+1) 无意义")
                        continue
                    df[col] = np.log10(data + 1)
                    transformed_cols.append(col)
                elif method == "Ln (x+1)":
                    if (data < -1).any():
                        errors.append(f"{col}: 存在 < -1 的值")
                        continue
                    df[col] = np.log1p(data)
                    transformed_cols.append(col)
                elif method == "Box-Cox (要求x>0)":
                    if (data <= 0).any():
                        errors.append(f"{col}: Box-Cox 要求所有数据严格大于0")
                    else:
                        transformed_data, lmbda = boxcox(data)
                        df[col] = transformed_data
                        transformed_cols.append(f"{col} (λ={lmbda:.2f})")
            except Exception as e:
                errors.append(f"{col}: {str(e)}")

        self.app.df = df
        # 记录变换标志，供分析引擎参考
        self.app._log_transformed = True
        self.app._preprocessing_applied = {
            'type': 'transform',
            'method': method,
            'columns': transformed_cols
        }
        self.app.all_columns = self.app.df.columns.tolist()
        if hasattr(self.app, 'tabs') and 'statistics' in self.app.tabs:
            # 清空统计图表缓存
            pass
            
        msg = f"✅ 变换完成: {method}\n"
        msg += f"⚠️ 仅对已选中的 {len(cols_to_transform)} 个变量执行变换\n\n"
        if transformed_cols:
            msg += f"成功变换了 {len(transformed_cols)} 个变量:\n" + ", ".join(transformed_cols) + "\n\n"
        if errors:
            msg += f"⚠️ 以下变量未能变换:\n" + "\n".join(errors)
            
        self.normality_text.insert("1.0", msg)

    # ── 标准化 ──

    def _build_norm_card(self, parent):
        card = make_card(parent)
        card.pack(fill="x", padx=12, pady=6)

        make_section_title(card, "数据标准化 / 归一化", icon="📐").pack(
            anchor="w", padx=18, pady=(16, 8))

        ctk.CTkLabel(card, text="⚠️ 警告：全局标准化会导致数据泄露！建议直接进入【分析】页，系统会自动使用 Pipeline 处理。", 
                     font=FONTS["small"](), text_color=C["warning"]).pack(anchor="w", padx=18, pady=(0, 8))

        row = make_inner_frame(card)
        row.pack(fill="x", padx=18, pady=(0, 6))

        ctk.CTkLabel(row, text="方法:", font=FONTS["body"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(0, 4))
        self.norm_method = ctk.StringVar(value="StandardScaler (Z标准化)")
        make_optionmenu(
            row, variable=self.norm_method, width=260,
            values=["StandardScaler (Z标准化)", "MinMaxScaler (0-1归一化)"]
        ).pack(side="left", padx=4)

        make_btn_warning(row, text="应用到数值列", width=130,
                         command=self._apply_norm).pack(side="left", padx=12)

        self.norm_text = make_textbox(card, height=80)
        self.norm_text.pack(fill="x", padx=18, pady=(4, 16))

    # ── 特征降维 ──

    def _build_dim_reduction_card(self, parent):
        card = make_card(parent)
        card.pack(fill="x", padx=12, pady=6)

        make_section_title(card, "特征降维与过滤", icon="✂️").pack(
            anchor="w", padx=18, pady=(16, 8))

        row = make_inner_frame(card)
        row.pack(fill="x", padx=18, pady=(0, 6))

        # 相关性过滤
        ctk.CTkLabel(row, text="高共线性阈值:", font=FONTS["body"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(0, 4))
        self.corr_th = ctk.DoubleVar(value=0.9)
        make_entry(row, textvariable=self.corr_th, width=60, justify="center").pack(side="left", padx=4)
        self.corr_btn = make_btn_warning(row, text="移除高相关特征", width=120,
                                         command=self._remove_collinear)
        self.corr_btn.pack(side="left", padx=(8, 24))

        # PCA降维
        ctk.CTkLabel(row, text="PCA 保留方差:", font=FONTS["body"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(0, 4))
        self.pca_var = ctk.DoubleVar(value=0.95)
        make_entry(row, textvariable=self.pca_var, width=60, justify="center").pack(side="left", padx=4)
        self.pca_btn = make_btn_primary(row, text="执行 PCA 降维", width=120,
                                        command=self._apply_pca)
        self.pca_btn.pack(side="left", padx=8)

        # 多项式特征
        ctk.CTkLabel(row, text="多项式阶数:", font=FONTS["body"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(16, 4))
        self.poly_degree = ctk.IntVar(value=2)
        make_entry(row, textvariable=self.poly_degree, width=50, justify="center").pack(side="left", padx=4)
        self.poly_btn = make_btn_primary(row, text="生成交互特征", width=120,
                                         command=self._apply_poly)
        self.poly_btn.pack(side="left", padx=8)

        self.dim_text = make_textbox(card, height=80)
        self.dim_text.pack(fill="x", padx=18, pady=(4, 16))
        self.advanced_buttons = [self.corr_btn, self.pca_btn, self.poly_btn]
        return card

    # ── 筛选 ──

    def _build_filter_card(self, parent):
        card = make_card(parent)
        card.pack(fill="x", padx=12, pady=6)

        make_section_title(card, "数据筛选与过滤", icon="🔎").pack(
            anchor="w", padx=18, pady=(16, 8))

        row = make_inner_frame(card)
        row.pack(fill="x", padx=18, pady=(0, 12))

        ctk.CTkLabel(row, text="列:", font=FONTS["body"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(0, 4))
        self.filter_col_var = ctk.StringVar(value="")
        self.filter_col_menu = make_optionmenu(
            row, variable=self.filter_col_var, width=170,
            values=["(请先加载数据)"])
        self.filter_col_menu.pack(side="left", padx=4)

        ctk.CTkLabel(row, text="条件:", font=FONTS["body"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(12, 4))
        self.filter_op = ctk.StringVar(value=">")
        make_optionmenu(
            row, variable=self.filter_op, width=70,
            values=[">", ">=", "<", "<=", "==", "!="]
        ).pack(side="left", padx=4)

        self.filter_val = ctk.StringVar(value="0")
        make_entry(row, textvariable=self.filter_val, width=100,
                   placeholder_text="数值…").pack(side="left", padx=4)

        make_btn_primary(row, text="筛选", width=80,
                         command=self._apply_filter).pack(side="left", padx=8)
        make_btn_warning(row, text="重置数据", width=100,
                         command=self._restore).pack(side="left", padx=4)

        self.filter_label = ctk.CTkLabel(
            card, text="", font=FONTS["small"](),
            text_color=C["text_secondary"])
        self.filter_label.pack(anchor="w", padx=18, pady=(0, 12))

    # ── 业务逻辑 ─────────────────────────────────────────

    def _df(self):
        return self.app.df

    def _require_data(self):
        if self.app.df is None:
            show_message(self.app, "❌ 错误", "请先加载数据", "error")
            return False
        return True

    def _backup(self):
        if not self._require_data():
            return
        self.app.df_backup = self.app.df.copy()
        self.backup_label.configure(
            text=f"  ✅ 已备份 ({len(self.app.df_backup)} 行)",
            text_color=C["success"])

    def _restore(self):
        if self.app.df_backup is None:
            show_message(self.app, "❌ 错误", "没有可用的备份数据", "error")
            return
        self.app.df = self.app.df_backup.copy()
        self.app.all_columns = self.app.df.columns.tolist()
        if hasattr(self.app, 'tabs'):
            if 'features' in self.app.tabs:
                self.app.tabs['features'].populate()
            if 'targets' in self.app.tabs:
                self.app.tabs['targets'].populate()
        self.refresh_columns()
        self.check_unlock()
        self.app.update_status_bar()
        self.app._log_transformed = False
        self.app._preprocessing_applied = None
        show_message(self.app, "✅ 恢复成功",
                     f"数据已恢复 ({len(self.app.df)} 行)", "info")

    def _detect_missing(self):
        if not self._require_data():
            return
        df = self.app.df
        self.miss_text.delete("1.0", "end")
        missing = df.isnull().sum()
        total = len(df)
        info = "📊  缺失值检测报告\n" + "━" * 52 + "\n"
        info += f"{'列名':28s}  {'缺失数':>8s}  {'缺失率':>8s}\n"
        info += "─" * 52 + "\n"
        found = False
        for col in df.columns:
            n = missing[col]
            if n > 0:
                found = True
                info += f"{col:28s}  {n:8d}  {n/total*100:7.2f}%\n"
        if not found:
            info += "  ✅ 未发现缺失值\n"
        info += "─" * 52 + f"\n总计: {missing.sum()} 个缺失值\n"
        self.miss_text.insert("1.0", info)

    def _handle_missing(self):
        if not self._require_data():
            return
        df = self.app.df
        before = df.isnull().sum().sum()
        if before == 0:
            show_message(self.app, "ℹ️ 提示", "没有缺失值", "info")
            return
        strat = self.miss_strategy.get()
        nc = df.select_dtypes(include=[np.number]).columns
        if strat == "均值填充":
            df[nc] = df[nc].fillna(df[nc].mean())
            self.app.df = df
        elif strat == "中位数填充":
            df[nc] = df[nc].fillna(df[nc].median())
            self.app.df = df
        elif strat == "众数填充":
            for c in df.columns:
                m = df[c].mode()
                if len(m) > 0:
                    df[c] = df[c].fillna(m[0])
            self.app.df = df
        elif strat == "删除缺失行":
            self.app.df = df.dropna().reset_index(drop=True)
        elif strat == "线性插值":
            df[nc] = df[nc].interpolate(method='linear')
            self.app.df = df.bfill().ffill()
        self.app.all_columns = self.app.df.columns.tolist()
        after = self.app.df.isnull().sum().sum()
        self.app.update_status_bar()
        show_message(self.app, "✅ 处理完成",
                     f"策略: {strat}\n处理前: {before}\n处理后: {after}\n"
                     f"当前行数: {len(self.app.df)}", "info")

    def _detect_outliers(self):
        if not self._require_data():
            return
        df = self.app.df
        method = self.outlier_method.get()
        nc = df.select_dtypes(include=[np.number]).columns
        self.outlier_text.delete("1.0", "end")
        info = f"📊  异常值检测 ({method})\n" + "━" * 55 + "\n"
        info += f"{'列名':28s}  {'异常值数':>10s}  {'占比':>8s}\n"
        info += "─" * 55 + "\n"
        total_out = 0
        for col in nc:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            if method == "IQR箱线图法":
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                n_out = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
            else:
                th = self.zscore_th.get()
                z = np.abs((s - s.mean()) / max(s.std(), 1e-10))
                n_out = (z > th).sum()
            if n_out > 0:
                info += f"{col:28s}  {n_out:10d}  {n_out/len(s)*100:7.2f}%\n"
                total_out += n_out
        if total_out == 0:
            info += "  ✅ 未检测到异常值\n"
        info += "─" * 55 + f"\n共 {total_out} 个异常值\n"
        self.outlier_text.insert("1.0", info)

    def _remove_outliers(self):
        if not self._require_data():
            return
        df = self.app.df
        method = self.outlier_method.get()
        nc = df.select_dtypes(include=[np.number]).columns
        before = len(df)
        mask = pd.Series([True] * before)
        for col in nc:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            if method == "IQR箱线图法":
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                cm = (df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr) | df[col].isna()
            else:
                th = self.zscore_th.get()
                mean, std = s.mean(), max(s.std(), 1e-10)
                cm = (np.abs((df[col] - mean) / std) <= th) | df[col].isna()
            mask = mask & cm
        self.app.df = df[mask].reset_index(drop=True)
        after = len(self.app.df)
        self.app.update_status_bar()
        show_message(self.app, "✅ 完成",
                     f"移除前: {before} 行\n移除后: {after} 行\n"
                     f"删除 {before - after} 行", "info")

    def _apply_norm(self):
        if not self._require_data():
            return
        df = self.app.df
        nc = df.select_dtypes(include=[np.number]).columns.tolist()
        if not nc:
            show_message(self.app, "ℹ️ 提示", "没有数值型列", "info")
            return
        self.norm_text.delete("1.0", "end")
        # 仅对选中的特征和目标进行缩放，防止污染无关列
        selected_features = self.app.get_selected_features()
        target_cols = self.app.get_selected_targets()
        cols_to_scale = [c for c in nc if c in selected_features or c in target_cols]
        
        if not cols_to_scale:
            show_message(self.app, "⚠️ 提示", "请先在【📋 特征】或【🎯 目标】页面勾选要缩放的变量", "warning")
            return
            
        method = self.norm_method.get()
        if "Standard" in method:
            scaler = StandardScaler()
            formula = "z = (x - μ) / σ"
        else:
            scaler = MinMaxScaler()
            formula = "x' = (x - min) / (max - min)"
            
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        self.app.df = df
        self.app._preprocessing_applied = {
            'type': 'scale',
            'method': method,
            'columns': cols_to_scale
        }
        
        info = f"✅ 已应用 {method}\n公式: {formula}\n处理列数: {len(cols_to_scale)}\n"
        info += f"列: {', '.join(cols_to_scale[:8])}"
        if len(cols_to_scale) > 8:
            info += f" … 等 {len(cols_to_scale)} 列"
        self.norm_text.insert("1.0", info)

    def _remove_collinear(self):
        if not self._require_data():
            return
        df = self.app.df
        nc = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = self.app.get_selected_features()
        target_cols = self.app.get_selected_targets()
        feat_cols = [c for c in selected_features if c in nc and c not in target_cols]
        if len(feat_cols) < 2:
            show_message(
                self.app,
                "⚠️ 前置条件未满足",
                "相关性过滤需要先在【📋 特征】页勾选至少 2 个数值型特征。",
                "warning"
            )
            return
        
        self.dim_text.delete("1.0", "end")
        th = self.corr_th.get()
        corr_matrix = df[feat_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > th)]
        
        if not to_drop:
            self.dim_text.insert("1.0", f"✅ 未发现相关系数大于 {th} 的特征对。")
            return
            
        self.app.df = df.drop(columns=to_drop)
        self.app.all_columns = self.app.df.columns.tolist()
        self.app.update_status_bar()
        self.refresh_columns()
        if hasattr(self.app, 'event_bus'):
            self.app.event_bus.publish('data_shape_changed')
        else:
            if hasattr(self.app, 'tabs'):
                if 'features' in self.app.tabs: self.app.tabs['features'].populate()
                if 'targets' in self.app.tabs: self.app.tabs['targets'].populate()
        self.check_unlock()
            
        info = f"✅ 已移除 {len(to_drop)} 个高共线性特征 (阈值 > {th}):\n"
        info += ", ".join(to_drop)
        self.dim_text.insert("1.0", info)

    def _apply_pca(self):
        if not self._require_data():
            return
        df = self.app.df
        nc = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = self.app.get_selected_features()
        feat_cols = [c for c in selected_features if c in nc]
        if len(feat_cols) < 2:
            show_message(
                self.app,
                "⚠️ 前置条件未满足",
                "PCA 降维需要先在【📋 特征】页勾选至少 2 个数值型特征。",
                "warning"
            )
            return
            
        self.dim_text.delete("1.0", "end")
        var_th = self.pca_var.get()
        
        # PCA requires data without NaNs
        if df[feat_cols].isnull().sum().sum() > 0:
            show_message(self.app, "❌ 错误", "数据中存在缺失值，请先处理缺失值再进行PCA降维", "error")
            return
            
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[feat_cols])
        
        pca = PCA(n_components=var_th)
        pca_result = pca.fit_transform(scaled_data)
        
        n_components = pca.n_components_
        pca_cols = [f'PCA_Component_{i+1}' for i in range(n_components)]
        
        df_pca = pd.DataFrame(pca_result, columns=pca_cols)
        
        other_cols = [c for c in df.columns if c not in feat_cols]
        self.app.df = pd.concat([df[other_cols].reset_index(drop=True), df_pca], axis=1)
        self.app.all_columns = self.app.df.columns.tolist()
        self.app.update_status_bar()
        self.refresh_columns()
        
        if hasattr(self.app, 'event_bus'):
            self.app.event_bus.publish('data_shape_changed')
        else:
            if hasattr(self.app, 'tabs'):
                if 'features' in self.app.tabs: self.app.tabs['features'].populate()
                if 'targets' in self.app.tabs: self.app.tabs['targets'].populate()
        self.check_unlock()
            
        info = f"✅ PCA降维完成 (保留 {var_th*100}% 方差):\n"
        info += f"参与降维特征数: {len(feat_cols)}  →  降维后主成分数: {n_components}\n"
        info += f"各主成分方差贡献率: {', '.join([f'{v*100:.1f}%' for v in pca.explained_variance_ratio_])}"
        self.dim_text.insert("1.0", info)

    def _apply_poly(self):
        if not self._require_data():
            return
        df = self.app.df
        # 仅对数值列进行多项式展开，并且排除了可能已经是目标变量的列
        # 为了安全，这里提示用户最好在指定好目标变量后再操作，或者先选择需要交互的特征
        nc = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 排除可能是目标变量的列（如果在目标变量页面已经勾选）
        target_cols = self.app.get_selected_targets()
            
        # 仅对用户在特征页面勾选的特征进行组合
        selected_features = self.app.get_selected_features()

        if not selected_features:
            show_message(
                self.app,
                "⚠️ 前置条件未满足",
                "多项式交互特征需要先在【📋 特征】页勾选特征变量。",
                "warning"
            )
            return

        if not target_cols:
            show_message(
                self.app,
                "⚠️ 建议先选目标",
                "尚未在【🎯 目标】页勾选目标变量。\n\n建议先选择目标变量，避免目标列被错误纳入特征工程。",
                "warning"
            )
            return
            
        feat_cols = [c for c in selected_features if c in nc and c not in target_cols]
        msg = f"将对勾选的 {len(feat_cols)} 个特征进行组合"
            
        if len(feat_cols) < 2:
            show_message(self.app, "ℹ️ 提示", "可用的数值型特征太少，无法生成交互特征。请先在【📋 特征】页面勾选至少2个特征。", "info")
            return
            
        degree = self.poly_degree.get()
        if degree > 3:
            show_message(self.app, "❌ 错误", "阶数过高会导致维度爆炸，建议不超过3", "error")
            return
            
        self.dim_text.delete("1.0", "end")
        self.dim_text.insert("1.0", msg + "\n正在计算...\n")
        self.dim_text.update()
        
        # 处理缺失值
        if df[feat_cols].isnull().sum().sum() > 0:
            show_message(self.app, "❌ 错误", "数据中存在缺失值，请先处理缺失值", "error")
            return
            
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
        poly_data = poly.fit_transform(df[feat_cols])
        
        # 获取新特征名
        new_names = poly.get_feature_names_out(feat_cols)
        
        # 将原始列中已经被展开的部分去掉，替换为多项式特征
        df_poly = pd.DataFrame(poly_data, columns=new_names)
        
        other_cols = [c for c in df.columns if c not in feat_cols]
        self.app.df = pd.concat([df[other_cols].reset_index(drop=True), df_poly], axis=1)
        self.app.all_columns = self.app.df.columns.tolist()
        self.app.update_status_bar()
        self.refresh_columns()
        
        if hasattr(self.app, 'event_bus'):
            self.app.event_bus.publish('data_shape_changed')
        else:
            if hasattr(self.app, 'tabs'):
                if 'features' in self.app.tabs: self.app.tabs['features'].populate()
                if 'targets' in self.app.tabs: self.app.tabs['targets'].populate()
        self.check_unlock()
            
        info = f"✅ 多项式特征生成完成 (阶数: {degree}):\n"
        info += f"原特征数: {len(feat_cols)}  →  新特征总数: {len(new_names)}\n"
        info += f"新增了如 {', '.join(new_names[len(feat_cols):len(feat_cols)+3])} 等交互/高阶特征\n"
        info += "⚠️ 请注意：数据列已改变，请务必重新选择特征和目标！"
        self.dim_text.insert("1.0", info)
        
        show_message(self.app, "⚠️ 需重新选择特征和目标", 
                     "交互特征生成完毕，数据列已发生变化。\n\n请务必前往【📋 特征】和【🎯 目标】标签页\n重新勾选需要分析的变量！", 
                     "info")

    def _apply_filter(self):
        if not self._require_data():
            return
        col = self.filter_col_var.get()
        op = self.filter_op.get()
        val_s = self.filter_val.get().strip()
        if col not in self.app.df.columns:
            show_message(self.app, "❌ 错误", f"列 '{col}' 不存在", "error")
            return
        try:
            val = float(val_s)
        except ValueError:
            show_message(self.app, "❌ 错误", "请输入有效数值", "error")
            return
        before = len(self.app.df)
        s = pd.to_numeric(self.app.df[col], errors='coerce')
        ops = {">": s > val, ">=": s >= val, "<": s < val,
               "<=": s <= val, "==": s == val, "!=": s != val}
        self.app.df = self.app.df[ops[op]].reset_index(drop=True)
        after = len(self.app.df)
        self.app.update_status_bar()
        self.filter_label.configure(
            text=f"筛选完成: {col} {op} {val}  →  {before} → {after} 行  "
                 f"(移除 {before - after} 行)")
