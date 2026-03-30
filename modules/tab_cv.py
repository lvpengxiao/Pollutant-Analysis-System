"""
交叉验证标签页模块 (🔄 验证)
---------------------------------------------------------------------
用于评估各机器学习模型在当前数据集上的稳定性和泛化能力。

核心设计:
1. 始终生成全新的、未拟合 (Unfitted) 的模型实例参与 KFold。
2. 数据安全：独立执行 `log10(y+1)` 目标变换（如果未在预处理执行），避免数据泄露。
3. 对 PyGAM 提供特殊支持：由于 GAM 不完全兼容 scikit-learn 的 `clone` 机制，
   这里为其手写了定制的 CV 循环逻辑。
4. 图表展示：使用箱线图直观对比模型在各个折 (Fold) 上的得分波动范围。
"""

import os
import threading
from datetime import datetime
import numpy as np
import pandas as pd
import customtkinter as ctk
from tkinter import filedialog
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from joblib import parallel_backend

from .theme import (C, FONTS, show_message, make_card, make_inner_frame,
                    make_section_title, make_btn_primary, make_btn_secondary,
                    make_optionmenu, make_textbox, make_empty_state)
from .analysis_engine import build_model_pipeline, prepare_feature_frame
from .reporting import reproducibility_dataframe

try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False
try:
    import lightgbm as lgb
    _LGB = True
except ImportError:
    _LGB = False
try:
    from catboost import CatBoostRegressor
    _CAT = True
except ImportError:
    _CAT = False


try:
    from pygam import LinearGAM, s, f
    _GAM = True
except ImportError:
    _GAM = False

class CrossValidationTab:

    def __init__(self, parent, app):
        self.app = app
        self.parent = parent
        self._canvas = None
        self._build()

    def _build(self):
        card = make_card(self.parent)
        card.pack(fill="x", padx=18, pady=(14, 6))

        make_section_title(card, "K 折交叉验证配置", icon="🔄").pack(
            anchor="w", padx=18, pady=(16, 10))

        row = make_inner_frame(card)
        row.pack(fill="x", padx=18, pady=(0, 16))

        ctk.CTkLabel(row, text="K 值:", font=FONTS["body"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(0, 4))
        self.k_var = ctk.IntVar(value=5)
        make_optionmenu(row, variable=self.k_var, values=["3", "5", "10"],
                        width=70, command=lambda v: self.k_var.set(int(v))
                        ).pack(side="left", padx=4)

        ctk.CTkLabel(row, text="评估指标:", font=FONTS["body"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(20, 4))
        self.scoring_var = ctk.StringVar(value="r2")
        make_optionmenu(
            row, variable=self.scoring_var, width=230,
            values=["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]
        ).pack(side="left", padx=4)

        ctk.CTkLabel(row, text="视图:", font=FONTS["body"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(20, 4))
        self.view_mode_var = ctk.StringVar(value="按模型看")
        make_optionmenu(
            row, variable=self.view_mode_var, width=120,
            values=["按模型看", "按目标看"],
            command=lambda _v: self._plot()
        ).pack(side="left", padx=4)

        self.run_btn = make_btn_primary(
            row, text="🔄  运行验证", width=120,
            command=self._run)
        self.run_btn.pack(side="left", padx=16)

        make_btn_secondary(
            row, text="💾 导出结果", width=110,
            command=self._export_results
        ).pack(side="left")

        target_row = make_inner_frame(card)
        target_row.pack(fill="x", padx=18, pady=(0, 14))
        ctk.CTkLabel(target_row, text="目标变量:", font=FONTS["body"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(0, 4))
        self.target_view_var = ctk.StringVar(value="(全部)")
        self.target_menu = make_optionmenu(
            target_row, variable=self.target_view_var, width=220,
            values=["(全部)"],
            command=lambda _v: self._plot()
        )
        self.target_menu.pack(side="left", padx=4)

        self.textbox = make_textbox(self.parent, height=200, wrap="word")
        self.textbox.pack(fill="x", padx=18, pady=6)

        self.summary_frame = make_card(self.parent)
        self.summary_frame.pack(fill="x", padx=18, pady=(0, 6))
        self.summary_labels = {}
        summary_items = [
            ("best_model", "🏆 最优模型", "等待验证"),
            ("best_score", "📈 最佳平均得分", "-"),
            ("stability", "🧭 稳定性", "-"),
            ("coverage", "🎯 覆盖范围", "-"),
        ]
        for idx, (key, title, value) in enumerate(summary_items):
            card = ctk.CTkFrame(self.summary_frame, fg_color=C["bg_primary"], corner_radius=10)
            card.grid(row=0, column=idx, padx=8, pady=10, sticky="nsew")
            self.summary_frame.grid_columnconfigure(idx, weight=1)
            ctk.CTkLabel(card, text=title, font=FONTS["body_bold"](),
                         text_color=C["text_secondary"]).pack(anchor="w", padx=14, pady=(12, 4))
            lbl = ctk.CTkLabel(card, text=value, font=FONTS["body"](),
                               text_color=C["text_primary"], anchor="w", justify="left")
            lbl.pack(fill="x", padx=14, pady=(0, 12))
            self.summary_labels[key] = lbl

        self.chart_frame = make_card(self.parent)
        self.chart_frame.pack(fill="both", expand=True, padx=18, pady=(6, 14))
        
        # 为图表区域添加滚动支持
        self.chart_scroll = ctk.CTkScrollableFrame(self.chart_frame, fg_color="transparent")
        self.chart_scroll.pack(fill="both", expand=True, padx=4, pady=4)
        self.empty_state = make_empty_state(
            self.chart_scroll,
            "🔄",
            "暂无交叉验证结果",
            "请先完成数据导入、特征选择和目标选择，然后点击上方“运行验证”。",
            button_text="前往数据页",
            command=lambda: self.app.navigate_to_tab("data_load", force=True) if hasattr(self.app, 'navigate_to_tab') else None,
        )
        self.empty_state.pack(fill="both", expand=True, padx=6, pady=6)
        self.refresh_empty_state()

    def _show_empty_state(self):
        if self.empty_state is not None and not self.empty_state.winfo_manager():
            self.empty_state.pack(fill="both", expand=True, padx=6, pady=6)

    def _hide_empty_state(self):
        if self.empty_state is not None and self.empty_state.winfo_manager():
            self.empty_state.pack_forget()

    def refresh_empty_state(self):
        if self.app.df is None:
            self.empty_state.title_label.configure(text="暂无交叉验证结果")
            self.empty_state.message_label.configure(
                text="请先在【📂 数据】页面导入数据，然后依次选择特征与目标变量。"
            )
            if self.empty_state.action_button is not None:
                self.empty_state.action_button.configure(
                    text="前往数据页",
                    command=lambda: self.app.navigate_to_tab("data_load", force=True) if hasattr(self.app, 'navigate_to_tab') else None,
                )
                if not self.empty_state.action_button.winfo_manager():
                    self.empty_state.action_button.pack(pady=(0, 24))
            return

        if not self.app.get_selected_features() or not self.app.get_selected_targets():
            self.empty_state.title_label.configure(text="等待分析配置完成")
            self.empty_state.message_label.configure(
                text="请先在【📋 特征】和【🎯 目标】页面完成勾选，再回来运行 K 折交叉验证。"
            )
            if self.empty_state.action_button is not None:
                self.empty_state.action_button.configure(
                    text="前往特征页",
                    command=lambda: self.app.navigate_to_tab("features", force=True) if hasattr(self.app, 'navigate_to_tab') else None,
                )
                if not self.empty_state.action_button.winfo_manager():
                    self.empty_state.action_button.pack(pady=(0, 24))
            return

        self.empty_state.title_label.configure(text="准备运行交叉验证")
        self.empty_state.message_label.configure(
            text="配置好 K 值和评估指标后，点击上方“运行验证”即可查看模型稳定性。"
        )
        if self.empty_state.action_button is not None and self.empty_state.action_button.winfo_manager():
            self.empty_state.action_button.pack_forget()

    def _run(self):
        app = self.app
        if app.df is None:
            show_message(app, "❌ 错误", "请先加载数据", "error")
            return
        feats = [(c, i['type'].get()) for c, i in app.feature_vars.items()
                 if i['selected'].get()]
        if not feats:
            show_message(app, "❌ 错误", "请先选择特征", "error")
            return
        targets = [c for c, v in app.target_vars.items() if v.get()]
        if not targets:
            show_message(app, "❌ 错误", "请先选择目标变量", "error")
            return

        # 收集主线程的 Tkinter 变量值，避免在子线程调用 .get() 导致 RuntimeError
        params = {
            'k': self.k_var.get(),
            'scoring': self.scoring_var.get(),
            'ne': app.n_estimators_var.get(),
            'md': app.max_depth_var.get(),
            'lr': app.learning_rate_var.get(),
            'mss': app.min_samples_split_var.get(),
            'msl': app.min_samples_leaf_var.get(),
            'sub': app.subsample_var.get(),
            'col_s': app.colsample_var.get(),
            'models_enabled': {
                name: app.model_vars.get(name, ctk.BooleanVar(value=False)).get()
                for name in ['RandomForest', 'AdaBoost', 'XGBoost', 'LightGBM', 'CatBoost', 'GAM', 'Stacking']
            }
        }

        self.run_btn.configure(state="disabled", text="⏳ 验证中…")
        self.textbox.delete("1.0", "end")
        threading.Thread(target=self._worker, args=(feats, targets, params),
                         daemon=True).start()

    def _commit_cv_results(self, cv_results, cv_fold_df, cv_detail_df, cv_summary_df):
        self.app.cv_results = dict(cv_results)
        self.app.cv_fold_df = cv_fold_df
        self.app.cv_detail_df = cv_detail_df
        self.app.cv_summary_df = cv_summary_df

    def _finalize_cv_results(self, cv_results, cv_fold_df, cv_detail_df, cv_summary_df, text):
        self._commit_cv_results(cv_results, cv_fold_df, cv_detail_df, cv_summary_df)
        self._show_results(text)

    def _score_gam_fold(self, X_train, X_test, y_train, y_test, feats, scoring):
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        model = self._build_gam(feats)
        X_train_imp = X_train.copy()
        X_test_imp = X_test.copy()

        for col, feature_type in feats:
            if feature_type == 'numeric':
                mean_val = X_train_imp[col].mean()
                if pd.isna(mean_val) or np.isinf(mean_val):
                    mean_val = 0.0
                X_train_imp[col] = X_train_imp[col].fillna(mean_val).replace([np.inf, -np.inf], mean_val)
                X_test_imp[col] = X_test_imp[col].fillna(mean_val).replace([np.inf, -np.inf], mean_val)
            else:
                mode_s = X_train_imp[col].mode()
                mode_val = mode_s.iloc[0] if not mode_s.empty else "Unknown"
                X_train_imp[col] = X_train_imp[col].fillna(mode_val)
                X_test_imp[col] = X_test_imp[col].fillna(mode_val)

                encoder = LabelEncoder()
                X_train_imp[col] = encoder.fit_transform(X_train_imp[col].astype(str))

                def safe_transform(value, le=encoder):
                    return le.transform([value])[0] if value in le.classes_ else 0

                X_test_imp[col] = X_test_imp[col].astype(str).map(safe_transform)

        model.fit(X_train_imp.values, y_train)
        preds = model.predict(X_test_imp.values)

        if scoring == 'r2':
            return float(r2_score(y_test, preds))
        if scoring == 'neg_mean_squared_error':
            return float(-mean_squared_error(y_test, preds))
        if scoring == 'neg_mean_absolute_error':
            return float(-mean_absolute_error(y_test, preds))
        return np.nan

    def _worker(self, feats, targets, params):
        app = self.app
        try:
            df_w, X, _feat_names, numeric_features, categorical_features = prepare_feature_frame(app.df, feats)

            k = params['k']
            scoring = params['scoring']
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            ne = params['ne']
            md = params['md']
            lr = params['lr']
            mss = params['mss']
            msl = params['msl']
            sub = params['sub']
            col_s = params['col_s']

            model_builders = {
                'RandomForest': lambda: RandomForestRegressor(
                    n_estimators=ne, max_depth=md,
                    min_samples_split=mss, min_samples_leaf=msl,
                    random_state=42, n_jobs=-1),
                'AdaBoost': lambda: AdaBoostRegressor(
                    n_estimators=ne, learning_rate=lr, random_state=42),
                'XGBoost': lambda: xgb.XGBRegressor(
                    n_estimators=ne, max_depth=md, learning_rate=lr,
                    subsample=sub, colsample_bytree=col_s,
                    random_state=42, n_jobs=-1, verbosity=0),
                'LightGBM': lambda: lgb.LGBMRegressor(
                    n_estimators=ne, max_depth=md, learning_rate=lr,
                    subsample=sub, colsample_bytree=col_s,
                    random_state=42, n_jobs=-1, verbosity=-1),
                'CatBoost': lambda: CatBoostRegressor(
                    iterations=ne, depth=min(md, 10), learning_rate=lr,
                    random_seed=42, verbose=False),
                'GAM': lambda: self._build_gam(feats),
            }

            enabled_models = []
            for name in ['RandomForest', 'AdaBoost', 'XGBoost', 'LightGBM', 'CatBoost', 'GAM']:
                if not params['models_enabled'].get(name, False):
                    continue
                if name == 'XGBoost' and not _XGB:
                    continue
                if name == 'LightGBM' and not _LGB:
                    continue
                if name == 'CatBoost' and not _CAT:
                    continue
                if name == 'GAM' and not _GAM:
                    continue
                enabled_models.append(name)

            if not enabled_models:
                self.app.after(0, lambda: show_message(app, "❌", "未选择可用模型", "error"))
                return

            lines = [
                f"🔄  K 折交叉验证  (K={k},  指标={scoring})",
                f"目标变量数: {len(targets)}",
                "═" * 60,
                "注: 非 GAM 模型与分析页共享同一预处理 Pipeline；GAM 使用 pyGAM 兼容的专用编码路径。",
                "注: 由于 GAM 使用折内填补 + LabelEncoder，而其他模型使用共享 Pipeline + OneHotEncoder，其结果不宜与其他模型做严格横向比较。",
            ]
            local_cv_results = {}
            cv_rows = []
            overall_rows = []
            fold_rows = []

            for model_name in enabled_models:
                model_all_scores = []
                evaluated_targets = 0
                lines.append(f"\n▶ {model_name}:")

                for target in targets:
                    valid_idx = df_w[target].notna()
                    valid_count = int(valid_idx.sum())
                    if valid_count < k:
                        lines.append(f"  - {target}: 可用样本 {valid_count} 小于 K={k}，已跳过")
                        continue

                    X_valid = X.loc[valid_idx].reset_index(drop=True)
                    y_valid = pd.to_numeric(df_w.loc[valid_idx, target], errors='coerce').values
                    y_transformed = y_valid if app._log_transformed else np.log10(y_valid + 1)

                    if model_name == 'GAM':
                        scores = []
                        for train_idx, test_idx in kf.split(X_valid):
                            X_train = X_valid.iloc[train_idx].copy()
                            X_test = X_valid.iloc[test_idx].copy()
                            y_train = y_transformed[train_idx]
                            y_test = y_transformed[test_idx]
                            try:
                                scores.append(
                                    self._score_gam_fold(X_train, X_test, y_train, y_test, feats, scoring)
                                )
                            except Exception:
                                import logging

                                logging.getLogger("PollutantApp").error("GAM CV 失败", exc_info=True)
                                scores.append(np.nan)
                        scores = np.array(scores, dtype=float)
                    else:
                        estimator = build_model_pipeline(
                            model_builders[model_name](),
                            numeric_features,
                            categorical_features,
                        )
                        with parallel_backend('threading'):
                            scores = cross_val_score(
                                estimator,
                                X_valid,
                                y_transformed,
                                cv=kf,
                                scoring=scoring,
                                n_jobs=-1,
                            )

                    evaluated_targets += 1
                    model_all_scores.extend(scores.tolist())
                    mean_score = float(np.nanmean(scores))
                    std_score = float(np.nanstd(scores))
                    stab = "稳定" if std_score < 0.1 else "不稳定" if std_score > 0.2 else "中等"

                    for fold_idx, score in enumerate(scores, start=1):
                        fold_rows.append({
                            '模型': model_name,
                            '目标变量': target,
                            '折次': fold_idx,
                            '得分': round(float(score), 6),
                            '评估指标': scoring,
                            'K值': k,
                        })

                    lines.append(f"  - {target}: 平均={mean_score:.4f}  ±{std_score:.4f}  稳定性={stab}")
                    cv_rows.append({
                        '模型': model_name,
                        '目标变量': target,
                        '评估指标': scoring,
                        'K值': k,
                        '平均得分': round(mean_score, 6),
                        '标准差': round(std_score, 6),
                        '最小得分': round(float(np.nanmin(scores)), 6),
                        '最大得分': round(float(np.nanmax(scores)), 6),
                        '稳定性': stab,
                        '各折得分': ', '.join(f'{s:.6f}' for s in scores),
                    })

                local_cv_results[model_name] = np.array(model_all_scores, dtype=float)
                if model_all_scores:
                    overall_rows.append({
                        '模型': model_name,
                        '目标变量数': evaluated_targets,
                        '评估指标': scoring,
                        '总体平均得分': round(float(np.nanmean(model_all_scores)), 6),
                        '总体标准差': round(float(np.nanstd(model_all_scores)), 6),
                        '最佳得分': round(float(np.nanmax(model_all_scores)), 6),
                        '最差得分': round(float(np.nanmin(model_all_scores)), 6),
                    })
                    lines.append(
                        f"  汇总: 平均={np.nanmean(model_all_scores):.4f}  ±{np.nanstd(model_all_scores):.4f}"
                    )
                else:
                    overall_rows.append({
                        '模型': model_name,
                        '目标变量数': 0,
                        '评估指标': scoring,
                        '总体平均得分': np.nan,
                        '总体标准差': np.nan,
                        '最佳得分': np.nan,
                        '最差得分': np.nan,
                    })
                    lines.append("  汇总: 无可用目标完成交叉验证")

            cv_fold_df = pd.DataFrame(fold_rows)
            cv_detail_df = pd.DataFrame(cv_rows)
            cv_summary_df = pd.DataFrame(overall_rows)

            lines.append("\n" + "═" * 60 + "\n✅ 完成\n💾 如需保存，请点击上方“导出结果”按钮")
            text = "\n".join(lines)
            self.app.after(
                0,
                lambda: self._finalize_cv_results(
                    local_cv_results,
                    cv_fold_df,
                    cv_detail_df,
                    cv_summary_df,
                    text,
                ),
            )

        except Exception as e:
            err_msg = str(e)
            self.app.after(0, lambda m=err_msg: show_message(app, "❌ 交叉验证出错", m, "error"))
        finally:
            self.app.after(0, lambda: self.run_btn.configure(
                state="normal", text="🔄  运行验证"))

    def _build_gam(self, feats):
        """构建 LinearGAM 实例"""
        if not _GAM:
            return None
        formula = None
        for i, (_, ft) in enumerate(feats):
            if ft == 'categorical':
                formula = formula + f(i) if formula else f(i)
            else:
                formula = formula + s(i) if formula else s(i)
        return LinearGAM(formula)

    def _show_results(self, text):
        self.textbox.delete("1.0", "end")
        self.textbox.insert("1.0", text)
        self._update_summary_cards()
        if hasattr(self.app, 'cv_fold_df') and self.app.cv_fold_df is not None and len(self.app.cv_fold_df) > 0:
            targets = ["(全部)"] + sorted(self.app.cv_fold_df['目标变量'].unique().tolist())
            self.target_menu.configure(values=targets)
            if self.target_view_var.get() not in targets:
                self.target_view_var.set(targets[0])
        if self.app.cv_results:
            self._plot()

    def _update_summary_cards(self):
        if not hasattr(self.app, 'cv_summary_df') or self.app.cv_summary_df is None or self.app.cv_summary_df.empty:
            return
        summary_df = self.app.cv_summary_df.sort_values('总体平均得分', ascending=False).reset_index(drop=True)
        best = summary_df.iloc[0]
        stability = "稳定"
        if best['总体标准差'] > 0.2:
            stability = "波动较大"
        elif best['总体标准差'] > 0.1:
            stability = "中等"
        self.summary_labels["best_model"].configure(text=str(best['模型']))
        self.summary_labels["best_score"].configure(
            text=f"{best['总体平均得分']:.4f} ± {best['总体标准差']:.4f}"
        )
        self.summary_labels["stability"].configure(
            text=f"{stability}\n最佳={best['最佳得分']:.4f}｜最差={best['最差得分']:.4f}"
        )
        self.summary_labels["coverage"].configure(
            text=f"{int(best['目标变量数'])} 个目标｜{len(summary_df)} 个模型"
        )

    def _clear(self):
        if hasattr(self, '_canvas') and self._canvas:
            import matplotlib.pyplot as plt
            fig = self._canvas.figure
            self._canvas.get_tk_widget().destroy()
            plt.close(fig)
            self._canvas = None
        self._show_empty_state()

    def _plot(self):
        if not hasattr(self.app, 'cv_fold_df') or self.app.cv_fold_df is None or len(self.app.cv_fold_df) == 0:
            return
            
        self._clear()

        import matplotlib.pyplot as plt
        fig = Figure(figsize=(10, 5), dpi=90, facecolor=C["bg_primary"])
        ax = fig.add_subplot(111)
        data, labels = [], []

        cv_df = self.app.cv_fold_df.copy()
        title_suffix = "按模型汇总"
        if self.view_mode_var.get() == "按目标看":
            target = self.target_view_var.get()
            if target != "(全部)":
                cv_df = cv_df[cv_df['目标变量'] == target]
                title_suffix = f"目标变量：{target}"
            else:
                title_suffix = "目标变量：全部"

        for mn in cv_df['模型'].drop_duplicates().tolist():
            scores = cv_df.loc[cv_df['模型'] == mn, '得分'].values
            if len(scores) > 0:
                data.append(scores)
                labels.append(mn)

        if not data:
            ax.text(0.5, 0.5, "当前筛选条件下无交叉验证数据", ha='center', va='center',
                    fontsize=12, color=C["text_primary"])
            ax.set_facecolor(C["bg_primary"])
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            self._hide_empty_state()
            self._canvas = FigureCanvasTkAgg(fig, master=self.chart_scroll)
            self._canvas.draw()
            self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
            return

        self._hide_empty_state()
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        medianprops=dict(color='white', linewidth=2))
        palette = ['#6366f1', '#34d399', '#60a5fa', '#fbbf24', '#fb7185', '#a78bfa']
        for patch, color in zip(bp['boxes'], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor(C["text_secondary"])
        for el in ['whiskers', 'caps']:
            for item in bp[el]:
                item.set_color(C["text_muted"])
        for fl in bp['fliers']:
            fl.set(marker='o', markerfacecolor=C["accent"], markersize=5)
        ax.set_ylabel(self.scoring_var.get(), fontsize=11, color=C["text_primary"])
        ax.set_title(f"交叉验证得分分布（{title_suffix}）", fontsize=13, color=C["text_primary"], pad=10)
        ax.set_facecolor(C["bg_primary"])
        ax.tick_params(colors=C["text_muted"], labelsize=9)
        for sp in ax.spines.values():
            sp.set_color(C["border"])
        ax.yaxis.grid(True, alpha=0.15, color=C["text_muted"])
        fig.tight_layout()

        self._canvas = FigureCanvasTkAgg(fig, master=self.chart_scroll)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    def _export_results(self):
        if not hasattr(self.app, 'cv_fold_df') or self.app.cv_fold_df is None or len(self.app.cv_fold_df) == 0:
            show_message(self.app, "ℹ️", "请先运行交叉验证", "info")
            return

        fp = filedialog.asksaveasfilename(
            title="导出交叉验证结果",
            defaultextension=".xlsx",
            filetypes=[("Excel 文件", "*.xlsx")],
            initialfile=f"交叉验证结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        if not fp:
            return

        try:
            with pd.ExcelWriter(fp, engine='openpyxl') as w:
                df_config = reproducibility_dataframe(
                    file_path=self.app.file_path,
                    test_size=self.app.test_size_var.get() if hasattr(self.app, 'test_size_var') else None,
                    preprocessing_applied=self.app._preprocessing_applied,
                    log_transformed=self.app._log_transformed,
                    selected_features=[(c, i['type'].get()) for c, i in self.app.feature_vars.items() if i['selected'].get()],
                    selected_targets=[c for c, v in self.app.target_vars.items() if v.get()],
                    enabled_models={name: self.app.model_vars[name].get() for name in self.app.model_vars},
                    output_dir=self.app.output_dir_var.get() if hasattr(self.app, 'output_dir_var') else None,
                    extra_rows=[
                        ("导出类型", "交叉验证 Excel"),
                        ("K 值", self.k_var.get()),
                        ("评估指标", self.scoring_var.get()),
                        ("说明", "非 GAM 模型与分析页共享预处理 Pipeline；GAM 使用 pyGAM 兼容的专用预处理路径，结果不宜与其他模型严格横向比较。"),
                    ],
                )
                df_config.to_excel(w, sheet_name="实验配置", index=False)
                self.app.cv_fold_df.to_excel(w, sheet_name="每折明细", index=False)
                self.app.cv_detail_df.to_excel(w, sheet_name="逐目标结果", index=False)
                self.app.cv_summary_df.to_excel(w, sheet_name="模型汇总", index=False)
            show_message(self.app, "✅", f"交叉验证结果已导出：\n{fp}", "info")
        except Exception as e:
            show_message(self.app, "❌", str(e), "error")
