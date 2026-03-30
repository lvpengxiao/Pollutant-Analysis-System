"""
运行分析标签页模块 (🚀 分析)
---------------------------------------------------------------------
核心调度模块。负责收集用户在前置页面配置的所有参数（数据、特征、目标、超参数），
并在后台守护线程中实例化并训练多种机器学习模型（RF, XGBoost, LightGBM, CatBoost, GAM 等）。

主要功能:
1. 线程安全隔离：在主线程收集所有 Tkinter 变量，防止跨线程 `.get()` 崩溃。
2. 进度监控：实时计算并反馈各个模型的训练进度到状态栏。
3. Spearman 相关性分析：作为线性关联的基准参考。
4. 特征重要性提取：统一使用 `permutation_importance` 获取高稳健性的 RI 占比。
5. 结果聚合与导出：生成多 Sheet 的分析报告 Excel。
"""

import os
import gc
import logging
import threading
import multiprocessing
import traceback
import time
import concurrent.futures
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

def _fit_process_worker(model, X, y, queue):
    """已弃用：多进程模式已替换为线程模式，避免 pickle 死锁"""
    try:
        from joblib import parallel_backend
        with parallel_backend('threading'):
            model.fit(X, y)
        queue.put((True, model))
    except Exception as e:
        queue.put((False, e))

class GAMWrapper:
    def __init__(self, g): self.g = g
    def fit(self, X, y): self.g.gridsearch(X, y, progress=False)


class CatBoostWrapper:
    def __init__(self, model, cat_feature_indices):
        self.model = model
        self.cat_feature_indices = cat_feature_indices

    def fit(self, X, y):
        self.model.fit(X, y, cat_features=self.cat_feature_indices)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

import numpy as np
import pandas as pd
import customtkinter as ctk
from tkinter import filedialog
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from joblib import parallel_backend

from .theme import (C, FONTS, show_message, make_card, make_inner_frame,
                    make_section_title, make_entry, make_btn_primary,
                    make_btn_secondary, make_textbox, make_progress)
from .analysis_engine import (
    build_feature_preprocessor,
    compute_metrics,
    normalize_importances,
    prepare_feature_frame,
    spearman_for_target,
)
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


class AnalysisTab:

    def __init__(self, parent, app):
        self.app = app
        self.parent = parent
        self._cancel_flag = False
        self._build()

    def _build(self):
        # 输出目录
        card = make_card(self.parent)
        card.pack(fill="x", padx=18, pady=(14, 6))
        row = make_inner_frame(card)
        row.pack(fill="x", padx=18, pady=14)

        ctk.CTkLabel(row, text="📁  输出目录:", font=FONTS["body_bold"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(0, 8))
        self.app.output_dir_var = ctk.StringVar(value="./分析结果")
        make_entry(row, textvariable=self.app.output_dir_var).pack(
            side="left", fill="x", expand=True, padx=(0, 8))
        make_btn_secondary(row, text="选择目录", width=110,
                           command=self._browse_dir).pack(side="left")

        # 运行按钮
        bf = ctk.CTkFrame(self.parent, fg_color="transparent")
        bf.pack(fill="x", padx=18, pady=10)
        self.run_btn = make_btn_primary(
            bf, text="🚀  开 始 分 析", height=50,
            corner_radius=14, font=ctk.CTkFont(size=18, weight="bold"),
            command=self.run_analysis)
        self.run_btn.pack(side="left", fill="x", expand=True, padx=(80, 10))
        self.cancel_btn = ctk.CTkButton(
            bf, text="⏹ 取消", width=110, height=50,
            corner_radius=14, fg_color=C["error"], hover_color=C["error_hover"],
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self._cancel, state="disabled"
        )
        self.cancel_btn.pack(side="left", padx=(0, 80))

        # 进度
        pc = make_card(self.parent)
        pc.pack(fill="x", padx=18, pady=6)
        pi = make_inner_frame(pc)
        pi.pack(fill="x", padx=18, pady=10)

        self.progress = make_progress(pi)
        self.progress.pack(fill="x", pady=(0, 6))
        self.progress.set(0)

        self.status_label = ctk.CTkLabel(
            pi, text="⏳  等待开始…", font=FONTS["small"](),
            text_color=C["text_secondary"])
        self.status_label.pack(anchor="w")

        # 日志
        lc = make_card(self.parent)
        lc.pack(fill="both", expand=True, padx=18, pady=(6, 14))
        make_section_title(lc, "运行日志", icon="📝").pack(
            anchor="w", padx=18, pady=(14, 6))
        self.log = make_textbox(lc, text_color=C["success"], wrap="word")
        self.log.pack(fill="both", expand=True, padx=18, pady=(0, 16))

    def _browse_dir(self):
        d = filedialog.askdirectory(title="选择输出目录")
        if d:
            self.app.output_dir_var.set(d)

    def _safe_after(self, callback):
        try:
            self.app.after(0, callback)
            return True
        except RuntimeError:
            return False

    def _commit_results(
        self,
        local_label_encoders,
        spearman_results_df,
        local_all_res,
        local_pred_cache,
        local_model_cache,
        local_X_cache,
        comparison_df,
        performance_df,
    ):
        """在主线程中一次性提交分析结果，避免后台线程写入共享状态。"""
        self.app.label_encoders = dict(local_label_encoders)
        self.app.spearman_results_df = spearman_results_df
        self.app.analysis_results = dict(local_all_res)
        self.app.prediction_cache = dict(local_pred_cache)
        self.app.model_cache = dict(local_model_cache)
        self.app.X_cache = dict(local_X_cache)
        self.app.comparison_df = comparison_df
        self.app.performance_df = performance_df

    def _finalize_analysis(
        self,
        local_label_encoders,
        spearman_results_df,
        local_all_res,
        local_pred_cache,
        local_model_cache,
        local_X_cache,
        comparison_df,
        performance_df,
        output_file,
        n_targets,
        n_models,
    ):
        self._commit_results(
            local_label_encoders,
            spearman_results_df,
            local_all_res,
            local_pred_cache,
            local_model_cache,
            local_X_cache,
            comparison_df,
            performance_df,
        )
        if hasattr(self.app, 'refresh_navigation_state'):
            self.app.refresh_navigation_state()
        if 'simulation' in self.app.tabs:
            self.app.tabs['simulation'].populate_targets()
        if 'visualization' in self.app.tabs and hasattr(self.app.tabs['visualization'], 'refresh_targets'):
            self.app.tabs['visualization'].refresh_targets()
        self._status_main_thread("🎀  分析完成!", 100)
        self._show_completion_dialog(output_file, n_targets, n_models)
        self._notify_visualization()

    def _log(self, msg):
        self._safe_after(lambda: self._log_main_thread(msg))

    def _log_main_thread(self, msg):
        try:
            # 记录当前用户的滚动位置
            scroll_pos = self.log.yview()
            is_at_bottom = scroll_pos[1] >= 0.99  # 接近 1.0 表示在最底部
            
            self.log.insert("end", msg + "\n")
            
            # 只有当用户原本就处于底部时，才自动滚动到新内容
            if is_at_bottom:
                self.log.see("end")
        except Exception as e:
            logger.debug(f"日志更新失败: {e}")

    def _status(self, msg, pct=None):
        self._safe_after(lambda: self._status_main_thread(msg, pct))
        
    def _status_main_thread(self, msg, pct=None):
        try:
            self.status_label.configure(text=msg)
            if pct is not None:
                self.progress.set(pct / 100.0)
            self.app.status_bar.set(msg, pct)
        except Exception as e:
            logger.debug(f"状态栏更新失败: {e}")

    def _model_enabled(self, name):
        """检查某个模型是否被用户启用"""
        var = self.app.model_vars.get(name)
        return var is not None and var.get()

    def _validate_params(self):
        errors = []
        ts = self.app.test_size_var.get()
        if not (0.05 <= ts <= 0.5):
            errors.append(f"测试集比例 {ts} 超出范围 [0.05, 0.5]")
        ne = self.app.n_estimators_var.get()
        if not (10 <= ne <= 2000):
            errors.append(f"树数量 {ne} 超出范围 [10, 2000]")
        md = self.app.max_depth_var.get()
        if not (1 <= md <= 50):
            errors.append(f"最大深度 {md} 超出范围 [1, 50]")
        lr = self.app.learning_rate_var.get()
        if not (0.001 <= lr <= 1.0):
            errors.append(f"学习率 {lr} 超出范围 [0.001, 1.0]")
        if errors:
            show_message(self.app, "❌ 参数错误", "以下参数设置不合法:\n\n" + "\n".join(errors), "error")
            return False
        return True

    def _show_completion_dialog(self, output_file, n_targets, n_models):
        dialog = ctk.CTkToplevel(self.app)
        dialog.title("🎉 分析完成")
        dialog.geometry("520x320")
        dialog.transient(self.app)
        dialog.grab_set()

        ct = ctk.CTkFrame(dialog, corner_radius=0, fg_color=C["bg_primary"])
        ct.pack(fill="both", expand=True)

        ctk.CTkLabel(ct, text="🎉 分析完成！", font=FONTS["h1"](),
                     text_color=C["success"]).pack(pady=(25, 10))
        ctk.CTkLabel(
            ct,
            text=f"分析 {n_targets} 个目标 × {n_models} 个模型\n结果文件: {output_file}",
            font=FONTS["body"](),
            text_color=C["text_primary"],
            wraplength=460,
            justify="center"
        ).pack(pady=10)

        btn_frame = ctk.CTkFrame(ct, fg_color="transparent")
        btn_frame.pack(pady=20)

        make_btn_primary(
            btn_frame, text="📈 查看可视化", width=130,
            command=lambda: [dialog.destroy(), self.app.navigate_to_tab("visualization", force=True) if hasattr(self.app, 'navigate_to_tab') else self.app.tabview.set("📈 图表")]
        ).pack(side="left", padx=8)
        make_btn_secondary(
            btn_frame, text="🔄 交叉验证", width=130,
            command=lambda: [dialog.destroy(), self.app.navigate_to_tab("cv", force=True) if hasattr(self.app, 'navigate_to_tab') else self.app.tabview.set("🔄 CV")]
        ).pack(side="left", padx=8)
        make_btn_secondary(
            btn_frame, text="🔮 情景模拟", width=130,
            command=lambda: [dialog.destroy(), self.app.navigate_to_tab("simulation", force=True) if hasattr(self.app, 'navigate_to_tab') else self.app.tabview.set("🔮 模拟")]
        ).pack(side="left", padx=8)

    def _estimate_time(self, n_targets, n_models, n_samples, enable_gs):
        base_per_model_per_target = 2
        if enable_gs:
            base_per_model_per_target *= 10
        if n_samples > 5000:
            base_per_model_per_target *= 2
        total_seconds = max(1, n_targets * max(n_models, 1) * base_per_model_per_target)
        if total_seconds < 60:
            return f"约 {total_seconds:.0f} 秒"
        if total_seconds < 3600:
            return f"约 {total_seconds / 60:.0f} 分钟"
        return f"约 {total_seconds / 3600:.1f} 小时"

    def _show_confirm_dialog(self, summary, feats, targets):
        dialog = ctk.CTkToplevel(self.app)
        dialog.title("确认分析配置")
        dialog.geometry("560x420")
        dialog.transient(self.app)
        dialog.grab_set()

        card = ctk.CTkFrame(dialog, fg_color=C["bg_primary"], corner_radius=0)
        card.pack(fill="both", expand=True)

        ctk.CTkLabel(card, text="⚙️ 请确认分析配置", font=FONTS["h2"](),
                     text_color=C["text_primary"]).pack(pady=(20, 12))
        text = make_textbox(card, height=220, wrap="word")
        text.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        text.insert("1.0", summary)
        text.configure(state="disabled")

        btn_frame = ctk.CTkFrame(card, fg_color="transparent")
        btn_frame.pack(pady=(0, 20))
        make_btn_secondary(
            btn_frame, text="取消", width=110,
            command=dialog.destroy
        ).pack(side="left", padx=10)
        make_btn_primary(
            btn_frame, text="开始分析", width=130,
            command=lambda: [dialog.destroy(), self._start_analysis(feats, targets)]
        ).pack(side="left", padx=10)

    def _start_analysis(self, feats, targets):
        app = self.app
        self._cancel_flag = False
        self.run_btn.configure(state="disabled", text="⏳ 分析中…")
        self.cancel_btn.configure(state="normal", text="⏹ 取消")
        self.log.delete("1.0", "end")
        app.analysis_results.clear()
        app.prediction_cache.clear()
        app.model_cache = {}
        app.X_cache = {}

        # 收集主线程的 Tkinter 变量值，避免在子线程调用 .get() 导致 RuntimeError
        params = {
            'cat_map': {col: app.category_vars.get(col, ctk.StringVar(value='')).get().strip() for col in targets},
            'ts': app.test_size_var.get(),
            'ne': app.n_estimators_var.get(),
            'md': app.max_depth_var.get(),
            'lr': app.learning_rate_var.get(),
            'mss': app.min_samples_split_var.get(),
            'msl': app.min_samples_leaf_var.get(),
            'sub': app.subsample_var.get(),
            'col_s': app.colsample_var.get(),
            'min_r2': app.min_r2_var.get(),
            'min_scc': app.min_scc_var.get(),
            'min_pcc': app.min_pcc_var.get(),
            'enable_gs': getattr(app, 'enable_grid_search_var', ctk.BooleanVar(value=False)).get(),
            'output_dir': app.output_dir_var.get() or "./分析结果",
            'models_enabled': {
                name: app.model_vars.get(name, ctk.BooleanVar(value=False)).get()
                for name in ['RandomForest', 'AdaBoost', 'XGBoost', 'LightGBM', 'CatBoost', 'GAM', 'Stacking']
            }
        }

        threading.Thread(target=self._worker, args=(feats, targets, params), daemon=True).start()

    def _cancel(self):
        self._cancel_flag = True
        self.cancel_btn.configure(state="disabled", text="⏳ 取消中…")
        self._log("⚠️ 用户请求取消，将在当前目标或模型完成后停止...")

    def _check_cancelled(self):
        if self._cancel_flag:
            raise InterruptedError("用户取消分析")

    def _fit_with_cancel(
        self,
        model: Any,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        timeout: int = 600
    ) -> Any:
        """
        在线程中运行 model.fit()，支持取消和超时。
        
        相比多进程方案的优势:
        - 避免大对象 pickle 序列化的开销
        - 避免进程间通信的 Queue 缓冲区死锁问题
        - Pipeline + 大 DataFrame 可以直接传入共享内存
        
        Args:
            model: sklearn Pipeline 或其他具有 fit() 方法的模型
            X: 特征数据
            y: 目标变量
            timeout: 最大超时秒数（默认 600s = 10分钟）
            
        Returns:
            已 fit 的模型对象（改装后直接返回）
            
        Raises:
            InterruptedError: 用户取消训练
            TimeoutError: 超出超时时间
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # 将 model.fit() 提交到线程池
            # 注意: sklearn 的 fit() 是 in-place 修改，无返回值，但我们需要返回 model
            def fit_and_return():
                # 在线程后端环境中执行 fit，使得即使 n_jobs=-1 也不会创建子进程
                with parallel_backend('threading'):
                    model.fit(X, y)
                return model
            
            future = executor.submit(fit_and_return)
            
            elapsed = 0.0
            poll_interval = 0.5
            
            # 轮询等待，定期检查取消标志
            while elapsed < timeout:
                if self._cancel_flag:
                    logger.warning("用户取消了模型训练")
                    raise InterruptedError("用户取消分析")
                
                try:
                    # 短超时轮询，这样可以实时响应 _cancel_flag
                    result = future.result(timeout=poll_interval)
                    logger.debug(f"模型训练完成，耗时 {elapsed:.1f}s")
                    return result
                except concurrent.futures.TimeoutError:
                    elapsed += poll_interval
                    continue
            
            # 超时
            logger.error(f"模型训练超出 {timeout}s 限制")
            raise TimeoutError(
                f"模型训练超时 ({timeout}s)，"
                f"请检查数据规模或增大超时设置"
            )

    def _target_progress(self, ci, mc, idx, total):
        model_start = 10 + ((ci - 1) / max(mc, 1)) * 60
        model_span = 60 / max(mc, 1)
        return model_start + ((idx + 1) / max(total, 1)) * model_span

    def _pi_repeats(self, sample_count: int) -> int:
        # Keep PI on the test split while increasing repeats for small samples.
        if sample_count < 80:
            return 40
        if sample_count < 150:
            return 30
        return 20

    def _compute_permutation_importance(self, estimator, X_eval, y_eval):
        repeats = self._pi_repeats(len(X_eval))
        return permutation_importance(
            estimator,
            X_eval,
            y_eval,
            n_repeats=repeats,
            random_state=42,
            n_jobs=1,
        )

    # ── 入口 ──────────────────────────────────────────────

    def run_analysis(self):
        app = self.app
        if app.df is None:
            show_message(app, "❌ 错误", "请先加载数据", "error")
            return
        feats = [(c, i['type'].get()) for c, i in app.feature_vars.items()
                 if i['selected'].get()]
        if not feats:
            show_message(app, "❌ 错误", "请至少选择一个影响因素", "error")
            return
        targets = [c for c, v in app.target_vars.items() if v.get()]
        if not targets:
            show_message(app, "❌ 错误", "请至少选择一个目标变量", "error")
            return
        overlap = {c for c, _ in feats} & set(targets)
        if overlap:
            show_message(
                app,
                "⚠️ 警告",
                "以下列同时被选为特征和目标变量:\n"
                + ", ".join(sorted(overlap))
                + "\n\n请取消其中一侧的勾选后再运行分析。",
                "warning"
            )
            return
        if not self._validate_params():
            return
        models = [k for k, v in app.model_vars.items() if v.get()]
        enable_gs = getattr(app, 'enable_grid_search_var', ctk.BooleanVar(value=False)).get()
        estimate = self._estimate_time(len(targets), len(models), len(app.df), enable_gs)
        summary = (
            "即将开始分析，请确认以下配置:\n\n"
            f"📊 数据: {len(app.df)} 行 × {len(app.df.columns)} 列\n"
            f"📋 特征变量: {len(feats)} 个\n"
            f"🎯 目标变量: {len(targets)} 个\n"
            f"🤖 模型: {', '.join(models)}\n"
            f"⚙️ AutoML: {'开启' if enable_gs else '关闭'}\n"
            f"⏱️ 预计耗时: {estimate}\n"
            f"📁 输出目录: {app.output_dir_var.get() or './分析结果'}"
        )
        self._show_confirm_dialog(summary, feats, targets)

    # ── 工作线程 ──────────────────────────────────────────

    def _worker(self, selected_features, selected_targets, params):
        app = self.app
        try:
            self._log("═" * 70)
            self._log("🚀  开始分析…")
            self._log("═" * 70)
            self._status("🔄  准备数据…", 0)

            df_work, X, feat_names, numeric_features, categorical_features = prepare_feature_frame(
                app.df,
                selected_features,
            )
            local_label_encoders = {}  # Keep for compatibility if needed elsewhere
            
            # 只有当所有目标变量都缺失时才剔除行，因为目标变量可以分开处理
            # 暂时保留所有行，在训练具体的 target 时再 dropna(subset=[col])
            
            self._log(f"\n📊  总样本数: {len(X)}   特征: {len(feat_names)}   目标: {len(selected_targets)}")

            cat_map = params['cat_map']
            for col in selected_targets:
                if not cat_map[col]:
                    cat_map[col] = '未分类'

            target_cols = []
            for col in selected_targets:
                if col in df_work.columns:
                    df_work[col] = pd.to_numeric(df_work[col], errors='coerce')
                    target_cols.append(col)


            # 检查是否已在预处理中做了 log 变换
            if hasattr(self.app, '_log_transformed') and self.app._log_transformed:
                self._log("\n⚠️ 检测到数据已在预处理阶段做过对数变换，分析时将跳过额外的 log10 变换")
                transform_y = lambda y: y
            else:
                transform_y = lambda y: np.log10(y + 1)

            # Spearman
            self._log(f"\n{'━' * 50}")
            self._log("📈  Spearman 相关性分析…")
            self._status("📈  Spearman…", 10)
            sp_rows = []
            for idx, col in enumerate(target_cols):
                self._check_cancelled()
                self._status(f"📈  Spearman - {col} [{idx + 1}/{len(target_cols)}]", 10)
                sp_rows += spearman_for_target(
                    X, df_work[col].values, feat_names, col, cat_map.get(col, '未分类'))
            df_sp = pd.DataFrame(sp_rows)
            self._log("   ✅ Spearman 完成")

            # 参数
            ts = params['ts']
            ne = params['ne']
            md = params['md']
            lr = params['lr']
            mss = params['mss']
            msl = params['msl']
            sub = params['sub']
            col_s = params['col_s']
            min_r2 = params['min_r2']
            min_scc = params['min_scc']
            min_pcc = params['min_pcc']
            
            enable_gs = params['enable_gs']

            active = [(k, True) for k, v in params['models_enabled'].items() if v]
            mc = len(active)
            ci = 0
            
            # 使用局部变量收集结果，最后一次性提交，防止多线程竞争
            local_all_res = {}
            local_pred_cache = {}
            local_model_cache = {}
            local_X_cache = {col: X.copy() for col in target_cols}

            def _prog():
                return 10 + (ci / max(mc, 1)) * 60

            # RandomForest
            if params['models_enabled']['RandomForest']:
                ci += 1
                self._log(f"\n🌲  [{ci}/{mc}] 随机森林…")
                self._status(f"🌲  随机森林 ({ci}/{mc})", _prog())
                rows = []
                for idx, col in enumerate(target_cols):
                    self._check_cancelled()
                    self._status(
                        f"🌲  随机森林 ({ci}/{mc}) - {col} [{idx + 1}/{len(target_cols)}]",
                        self._target_progress(ci, mc, idx, len(target_cols))
                    )
                    
                    # Drop rows where target is NaN
                    valid_idx = df_work[col].notna()
                    X_valid = X[valid_idx]
                    y_valid = df_work.loc[valid_idx, col].values
                    
                    y_transformed = transform_y(y_valid)
                    Xtr, Xte, ytr, yte = train_test_split(X_valid, y_transformed, test_size=ts, random_state=42)
                    
                    if enable_gs:
                        param_dist = {
                            'model__n_estimators': [50, 100, 200],
                            'model__max_depth': [None, 5, 10, 20],
                            'model__min_samples_split': [2, 5, 10],
                            'model__min_samples_leaf': [1, 2, 4]
                        }
                        base_m = Pipeline(steps=[('preprocessor', build_feature_preprocessor(numeric_features, categorical_features)), ('model', RandomForestRegressor(random_state=42, n_jobs=-1))])
                        search = RandomizedSearchCV(base_m, param_distributions=param_dist, n_iter=10, cv=3, random_state=42, n_jobs=-1)
                        with parallel_backend('threading'):
                            search = self._fit_with_cancel(search, Xtr, ytr)
                        m = search.best_estimator_
                        self._log(f"   [{col}] 最佳参数: {search.best_params_}")
                    else:
                        m = Pipeline(steps=[
                            ('preprocessor', build_feature_preprocessor(numeric_features, categorical_features)),
                            ('model', RandomForestRegressor(n_estimators=ne, max_depth=md,
                                                  min_samples_split=mss, min_samples_leaf=msl,
                                                  random_state=42, n_jobs=-1))
                        ])
                        m = self._fit_with_cancel(m, Xtr, ytr)
                        
                    yp = m.predict(Xte)
                    met = compute_metrics(yte, yp)
                    
                    local_pred_cache[('RandomForest', col)] = (yte, yp)
                    local_model_cache[('RandomForest', col)] = m
                    local_X_cache[('RandomForest', col)] = X_valid.copy()
                    
                    # 优先使用 permutation_importance 获取更稳健的重要性评估
                    pi_result = self._compute_permutation_importance(m, Xte, yte)
                    imp = normalize_importances(pi_result.importances_mean)
                    
                    # Since OneHotEncoder expands features, we should get the new feature names
                    try:
                        feature_names_out = m.named_steps['preprocessor'].get_feature_names_out()
                        # If the lengths don't match, permutation importance is evaluated on the *original* Xte columns
                        # because we pass `m` (the pipeline) and `Xte` (the original DataFrame).
                        # permutation_importance will permute the original columns in Xte!
                        # So `imp` has the same length as `Xte.columns` (which is `feat_names`).
                    except Exception:
                        # OneHotEncoder 展开后的特征名与原始特征数量不匹配，无关紧要
                        logger.debug("无法获取预处理后的特征名（正常情况）")

                    row = {'目标变量': col, '类别': cat_map.get(col, '未分类'), **met}
                    for i, fn in enumerate(feat_names):
                        row[f'{fn}_RI(%)'] = round(imp[i] * 100, 2)
                    rows.append(row)
                local_all_res['RandomForest'] = pd.DataFrame(rows)
                self._log("   ✅ 随机森林完成")

            # AdaBoost
            if params['models_enabled']['AdaBoost']:
                ci += 1
                self._log(f"\n🚀  [{ci}/{mc}] AdaBoost…")
                self._status(f"🚀  AdaBoost ({ci}/{mc})", _prog())
                rows = []
                for idx, col in enumerate(target_cols):
                    self._check_cancelled()
                    self._status(
                        f"🚀  AdaBoost ({ci}/{mc}) - {col} [{idx + 1}/{len(target_cols)}]",
                        self._target_progress(ci, mc, idx, len(target_cols))
                    )
                    
                    valid_idx = df_work[col].notna()
                    X_valid = X[valid_idx]
                    y_valid = df_work.loc[valid_idx, col].values
                    
                    y_transformed = transform_y(y_valid)
                    Xtr, Xte, ytr, yte = train_test_split(X_valid, y_transformed, test_size=ts, random_state=42)
                    
                    if enable_gs:
                        param_dist = {
                            'model__n_estimators': [50, 100, 200],
                            'model__learning_rate': [0.01, 0.05, 0.1, 0.3]
                        }
                        base_m = Pipeline(steps=[('preprocessor', build_feature_preprocessor(numeric_features, categorical_features)), ('model', AdaBoostRegressor(random_state=42))])
                        search = RandomizedSearchCV(base_m, param_distributions=param_dist, n_iter=10, cv=3, random_state=42, n_jobs=1)
                        with parallel_backend('threading'):
                            search = self._fit_with_cancel(search, Xtr, ytr)
                        m = search.best_estimator_
                        self._log(f"   [{col}] 最佳参数: {search.best_params_}")
                    else:
                        m = Pipeline(steps=[
                            ('preprocessor', build_feature_preprocessor(numeric_features, categorical_features)),
                            ('model', AdaBoostRegressor(n_estimators=ne, learning_rate=lr, random_state=42))
                        ])
                        m = self._fit_with_cancel(m, Xtr, ytr)
                        
                    yp = m.predict(Xte)
                    met = compute_metrics(yte, yp)
                    local_pred_cache[('AdaBoost', col)] = (yte, yp)
                    local_model_cache[('AdaBoost', col)] = m
                    local_X_cache[('AdaBoost', col)] = X_valid.copy()
                    
                    pi_result = self._compute_permutation_importance(m, Xte, yte)
                    imp = normalize_importances(pi_result.importances_mean)
                    
                    row = {'目标变量': col, '类别': cat_map.get(col, '未分类'), **met}
                    for i, fn in enumerate(feat_names):
                        row[f'{fn}_RI(%)'] = round(imp[i] * 100, 2)
                    rows.append(row)
                local_all_res['AdaBoost'] = pd.DataFrame(rows)
                self._log("   ✅ AdaBoost 完成")

            # XGBoost
            if params['models_enabled']['XGBoost'] and _XGB:
                ci += 1
                self._log(f"\n⚡  [{ci}/{mc}] XGBoost…")
                self._status(f"⚡  XGBoost ({ci}/{mc})", _prog())
                rows = []
                for idx, col in enumerate(target_cols):
                    self._check_cancelled()
                    self._status(
                        f"⚡  XGBoost ({ci}/{mc}) - {col} [{idx + 1}/{len(target_cols)}]",
                        self._target_progress(ci, mc, idx, len(target_cols))
                    )
                    
                    valid_idx = df_work[col].notna()
                    X_valid = X[valid_idx]
                    y_valid = df_work.loc[valid_idx, col].values
                    
                    y_transformed = transform_y(y_valid)
                    Xtr, Xte, ytr, yte = train_test_split(X_valid, y_transformed, test_size=ts, random_state=42)
                    
                    if enable_gs:
                        param_dist = {
                            'model__n_estimators': [50, 100, 200],
                            'model__max_depth': [3, 6, 10],
                            'model__learning_rate': [0.01, 0.05, 0.1],
                            'model__subsample': [0.6, 0.8, 1.0]
                        }
                        base_m = Pipeline(steps=[('preprocessor', build_feature_preprocessor(numeric_features, categorical_features)), ('model', xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0))])
                        search = RandomizedSearchCV(base_m, param_distributions=param_dist, n_iter=10, cv=3, random_state=42, n_jobs=-1)
                        with parallel_backend('threading'):
                            search = self._fit_with_cancel(search, Xtr, ytr)
                        m = search.best_estimator_
                        self._log(f"   [{col}] 最佳参数: {search.best_params_}")
                    else:
                        m = Pipeline(steps=[
                            ('preprocessor', build_feature_preprocessor(numeric_features, categorical_features)),
                            ('model', xgb.XGBRegressor(n_estimators=ne, max_depth=md, learning_rate=lr,
                                             subsample=sub, colsample_bytree=col_s,
                                             reg_alpha=0.1, reg_lambda=1.0,
                                             random_state=42, n_jobs=-1, verbosity=0))
                        ])
                        m = self._fit_with_cancel(m, Xtr, ytr)
                        
                    yp = m.predict(Xte)
                    met = compute_metrics(yte, yp)
                    local_pred_cache[('XGBoost', col)] = (yte, yp)
                    local_model_cache[('XGBoost', col)] = m
                    local_X_cache[('XGBoost', col)] = X_valid.copy()
                    
                    pi_result = self._compute_permutation_importance(m, Xte, yte)
                    imp = normalize_importances(pi_result.importances_mean)
                    
                    row = {'目标变量': col, '类别': cat_map.get(col, '未分类'), **met}
                    for i, fn in enumerate(feat_names):
                        row[f'{fn}_RI(%)'] = round(imp[i] * 100, 2)
                    rows.append(row)
                local_all_res['XGBoost'] = pd.DataFrame(rows)
                self._log("   ✅ XGBoost 完成")

            # LightGBM
            if params['models_enabled']['LightGBM'] and _LGB:
                ci += 1
                self._log(f"\n💡  [{ci}/{mc}] LightGBM…")
                self._status(f"💡  LightGBM ({ci}/{mc})", _prog())
                rows = []
                for idx, col in enumerate(target_cols):
                    self._check_cancelled()
                    self._status(
                        f"💡  LightGBM ({ci}/{mc}) - {col} [{idx + 1}/{len(target_cols)}]",
                        self._target_progress(ci, mc, idx, len(target_cols))
                    )
                    
                    valid_idx = df_work[col].notna()
                    X_valid = X[valid_idx]
                    y_valid = df_work.loc[valid_idx, col].values
                    
                    y_transformed = transform_y(y_valid)
                    Xtr, Xte, ytr, yte = train_test_split(X_valid, y_transformed, test_size=ts, random_state=42)
                    if enable_gs:
                        param_dist = {
                            'model__n_estimators': [50, 100, 200],
                            'model__max_depth': [3, 6, 10, -1],
                            'model__learning_rate': [0.01, 0.05, 0.1],
                            'model__subsample': [0.6, 0.8, 1.0],
                            'model__colsample_bytree': [0.6, 0.8, 1.0]
                        }
                        base_m = Pipeline(steps=[('preprocessor', build_feature_preprocessor(numeric_features, categorical_features)), ('model', lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1))])
                        search = RandomizedSearchCV(base_m, param_distributions=param_dist,
                                                    n_iter=10, cv=3, random_state=42, n_jobs=-1)
                        with parallel_backend('threading'):
                            search = self._fit_with_cancel(search, Xtr, ytr)
                        m = search.best_estimator_
                        self._log(f"   [{col}] 最佳参数: {search.best_params_}")
                    else:
                        m = Pipeline(steps=[
                            ('preprocessor', build_feature_preprocessor(numeric_features, categorical_features)),
                            ('model', lgb.LGBMRegressor(n_estimators=ne, max_depth=md, learning_rate=lr,
                                              num_leaves=31, min_child_samples=20,
                                              subsample=sub, colsample_bytree=col_s,
                                              reg_alpha=0.1, reg_lambda=1.0,
                                              random_state=42, n_jobs=-1, verbosity=-1))
                        ])
                        m = self._fit_with_cancel(m, Xtr, ytr)
                    yp = m.predict(Xte)
                    met = compute_metrics(yte, yp)
                    
                    pi_result = self._compute_permutation_importance(m, Xte, yte)
                    imp = normalize_importances(pi_result.importances_mean)
                    
                    local_pred_cache[('LightGBM', col)] = (yte, yp)
                    local_model_cache[('LightGBM', col)] = m
                    local_X_cache[('LightGBM', col)] = X_valid.copy()
                    row = {'目标变量': col, '类别': cat_map.get(col, '未分类'), **met}
                    for i, fn in enumerate(feat_names):
                        row[f'{fn}_RI(%)'] = round(imp[i] * 100, 2)
                    rows.append(row)
                local_all_res['LightGBM'] = pd.DataFrame(rows)
                self._log("   ✅ LightGBM 完成")

            # CatBoost
            if params['models_enabled']['CatBoost'] and _CAT:
                ci += 1
                self._log(f"\n🐱  [{ci}/{mc}] CatBoost…")
                self._status(f"🐱  CatBoost ({ci}/{mc})", _prog())
                rows = []
                cat_feature_indices = [
                    i for i, (_, ftype) in enumerate(selected_features)
                    if ftype == 'categorical'
                ]
                for idx, col in enumerate(target_cols):
                    self._check_cancelled()
                    self._status(
                        f"🐱  CatBoost ({ci}/{mc}) - {col} [{idx + 1}/{len(target_cols)}]",
                        self._target_progress(ci, mc, idx, len(target_cols))
                    )
                    
                    valid_idx = df_work[col].notna()
                    X_valid = X[valid_idx]
                    y_valid = df_work.loc[valid_idx, col].values
                    
                    y_transformed = transform_y(y_valid)
                    Xtr, Xte, ytr, yte = train_test_split(X_valid, y_transformed, test_size=ts, random_state=42)

                    Xtr_cb = Xtr.copy()
                    Xte_cb = Xte.copy()

                    for f_col in numeric_features:
                        train_mean = pd.to_numeric(Xtr_cb[f_col], errors='coerce').mean()
                        if pd.isna(train_mean) or np.isinf(train_mean):
                            train_mean = 0.0
                        Xtr_cb[f_col] = pd.to_numeric(Xtr_cb[f_col], errors='coerce').fillna(train_mean)
                        Xte_cb[f_col] = pd.to_numeric(Xte_cb[f_col], errors='coerce').fillna(train_mean)
                        Xtr_cb[f_col] = Xtr_cb[f_col].replace([np.inf, -np.inf], train_mean)
                        Xte_cb[f_col] = Xte_cb[f_col].replace([np.inf, -np.inf], train_mean)

                    for f_col in categorical_features:
                        train_mode = Xtr_cb[f_col].mode()
                        mode_value = train_mode.iloc[0] if not train_mode.empty else 'Unknown'
                        Xtr_cb[f_col] = Xtr_cb[f_col].fillna(mode_value).astype(str)
                        Xte_cb[f_col] = Xte_cb[f_col].fillna(mode_value).astype(str)

                    if enable_gs:
                        self._log(f"   [{col}] 提示: CatBoost 当前使用原生分类特征路径，暂不启用 RandomizedSearchCV。")

                    cb_model = CatBoostRegressor(
                        iterations=ne,
                        depth=min(md, 10),
                        learning_rate=lr,
                        l2_leaf_reg=3.0,
                        random_seed=42,
                        verbose=False,
                    )
                    wrapped_model = CatBoostWrapper(cb_model, cat_feature_indices)
                    wrapped_model = self._fit_with_cancel(wrapped_model, Xtr_cb, ytr)
                    m = wrapped_model.model

                    yp = m.predict(Xte_cb)
                    met = compute_metrics(yte, yp)
                    
                    pi_result = self._compute_permutation_importance(m, Xte_cb, yte)
                    imp = normalize_importances(pi_result.importances_mean)
                    
                    local_pred_cache[('CatBoost', col)] = (yte, yp)
                    local_model_cache[('CatBoost', col)] = m
                    local_X_cache[('CatBoost', col)] = X_valid.copy()
                    row = {'目标变量': col, '类别': cat_map.get(col, '未分类'), **met}
                    for i, fn in enumerate(feat_names):
                        row[f'{fn}_RI(%)'] = round(imp[i] * 100, 2)
                    rows.append(row)
                local_all_res['CatBoost'] = pd.DataFrame(rows)
                self._log("   ✅ CatBoost 完成")

            # GAM
            if params['models_enabled']['GAM'] and _GAM:
                ci += 1
                self._log(f"\n📈  [{ci}/{mc}] GAM…")
                self._status(f"📈  GAM ({ci}/{mc})", _prog())
                rows = []
                for idx, col in enumerate(target_cols):
                    self._check_cancelled()
                    self._status(
                        f"📈  GAM ({ci}/{mc}) - {col} [{idx + 1}/{len(target_cols)}]",
                        self._target_progress(ci, mc, idx, len(target_cols))
                    )
                    
                    valid_idx = df_work[col].notna()
                    X_valid = X[valid_idx]
                    y_valid = df_work.loc[valid_idx, col].values
                    
                    y_transformed = transform_y(y_valid)
                    Xtr, Xte, ytr, yte = train_test_split(X_valid, y_transformed, test_size=ts, random_state=42)
                    
                    # For GAM, we don't use the standard Pipeline because pyGAM expects original categorical indices
                    # We will impute manually to avoid data leakage
                    Xtr_imp = Xtr.copy()
                    Xte_imp = Xte.copy()
                    
                    gam_le = {}
                    
                    for f_col, ftype in selected_features:
                        if ftype == 'numeric':
                            mean_val = Xtr_imp[f_col].mean()
                            if pd.isna(mean_val) or np.isinf(mean_val):
                                mean_val = 0.0
                            Xtr_imp[f_col] = Xtr_imp[f_col].fillna(mean_val)
                            Xte_imp[f_col] = Xte_imp[f_col].fillna(mean_val)
                            
                            # Clean Inf values
                            Xtr_imp[f_col] = Xtr_imp[f_col].replace([np.inf, -np.inf], mean_val)
                            Xte_imp[f_col] = Xte_imp[f_col].replace([np.inf, -np.inf], mean_val)
                        else:
                            mode_s = Xtr_imp[f_col].mode()
                            mode_val = mode_s[0] if len(mode_s) > 0 else 'Unknown'
                            Xtr_imp[f_col] = Xtr_imp[f_col].fillna(mode_val)
                            Xte_imp[f_col] = Xte_imp[f_col].fillna(mode_val)
                            
                            # Label encode for pyGAM
                            le = LabelEncoder()
                            Xtr_imp[f_col] = le.fit_transform(Xtr_imp[f_col].astype(str))
                            gam_le[f_col] = le
                            # Handle unseen labels in test set safely
                            # Using .map with a default value of 0 for unknown classes
                            def safe_transform(x, le=le):
                                return le.transform([x])[0] if x in le.classes_ else 0
                            Xte_imp[f_col] = Xte_imp[f_col].astype(str).map(safe_transform)

                    Xg_tr = Xtr_imp.values
                    Xg_te = Xte_imp.values
                    
                    try:
                        formula = None
                        for i, (_, ft) in enumerate(selected_features):
                            t = s(i, n_splines=10) if ft == 'numeric' else f(i)
                            formula = t if formula is None else formula + t
                        gam = LinearGAM(formula)
                        # gam.gridsearch does not follow standard fit signature, so we wrap it
                        wrapper = GAMWrapper(gam)
                        wrapper = self._fit_with_cancel(wrapper, Xg_tr, ytr)
                        gam = wrapper.g
                        
                        yp = gam.predict(Xg_te)
                        met = compute_metrics(yte, yp)
                        local_pred_cache[('GAM', col)] = (yte, yp)
                        # Save both the model and the label encoders
                        local_model_cache[('GAM', col)] = {'model': gam, 'le': gam_le}
                        local_X_cache[('GAM', col)] = X_valid.copy()
                        imps = []
                        for i in range(len(selected_features)):
                            try:
                                XX = gam.generate_X_grid(term=i)
                                pd_ = gam.partial_dependence(term=i, X=XX)
                                imps.append(np.std(pd_))
                            except Exception as e:
                                logger.warning(f"GAM 偏依赖计算失败 (term {i}): {e}，特征重要性设为 0")
                                imps.append(0)
                        tot = sum(imps) or 1
                        ri = [v / tot * 100 for v in imps]
                        row = {'目标变量': col, '类别': cat_map.get(col, '未分类'), **met}
                        for i, fn in enumerate(feat_names):
                            row[f'{fn}_RI(%)'] = round(ri[i], 2)
                        rows.append(row)
                    except Exception as e:
                        self._log(f"   ⚠️ GAM 失败 ({col}): {e}")
                        row = {'目标变量': col, '类别': cat_map.get(col, '未分类'),
                               'R²': np.nan, 'RMSE': np.nan, 'SCC': np.nan, 'PCC': np.nan}
                        for fn in feat_names:
                            row[f'{fn}_RI(%)'] = np.nan
                        rows.append(row)
                local_all_res['GAM'] = pd.DataFrame(rows)
                self._log("   ✅ GAM 完成")

            # Stacking
            if params['models_enabled'].get('Stacking', False):
                ci += 1
                self._log(f"\n🔗  [{ci}/{mc}] Stacking 融合模型…")
                self._status(f"🔗  Stacking ({ci}/{mc})", _prog())
                
                # 收集已启用的基础模型
                estimators = []
                if params['models_enabled'].get('RandomForest', False):
                    estimators.append((
                        'rf',
                        RandomForestRegressor(
                            n_estimators=ne,
                            max_depth=md,
                            min_samples_split=mss,
                            min_samples_leaf=msl,
                            random_state=42,
                            n_jobs=-1
                        )
                    ))
                if params['models_enabled'].get('AdaBoost', False):
                    estimators.append(('ada', AdaBoostRegressor(n_estimators=ne, learning_rate=lr, random_state=42)))
                if params['models_enabled'].get('XGBoost', False) and _XGB:
                    estimators.append((
                        'xgb',
                        xgb.XGBRegressor(
                            n_estimators=ne,
                            max_depth=md,
                            learning_rate=lr,
                            subsample=sub,
                            colsample_bytree=col_s,
                            random_state=42,
                            n_jobs=-1,
                            verbosity=0
                        )
                    ))
                if params['models_enabled'].get('LightGBM', False) and _LGB:
                    estimators.append((
                        'lgb',
                        lgb.LGBMRegressor(
                            n_estimators=ne,
                            max_depth=md,
                            learning_rate=lr,
                            subsample=sub,
                            colsample_bytree=col_s,
                            random_state=42,
                            n_jobs=-1,
                            verbosity=-1
                        )
                    ))
                if params['models_enabled'].get('CatBoost', False) and _CAT:
                    estimators.append(('cat', CatBoostRegressor(iterations=ne, depth=min(md, 10), learning_rate=lr, random_seed=42, verbose=False)))
                    
                if len(estimators) < 2:
                    self._log("   ⚠️ Stacking 需要至少勾选 2 个基础模型！跳过。")
                else:
                    rows = []
                    for idx, col in enumerate(target_cols):
                        self._check_cancelled()
                        self._status(
                            f"🔗  Stacking ({ci}/{mc}) - {col} [{idx + 1}/{len(target_cols)}]",
                            self._target_progress(ci, mc, idx, len(target_cols))
                        )
                        
                        valid_idx = df_work[col].notna()
                        X_valid = X[valid_idx]
                        y_valid = df_work.loc[valid_idx, col].values
                        
                        y_transformed = transform_y(y_valid)
                        Xtr, Xte, ytr, yte = train_test_split(X_valid, y_transformed, test_size=ts, random_state=42)
                        
                        stacking_model = StackingRegressor(
                            estimators=estimators,
                            final_estimator=RidgeCV(),
                            n_jobs=-1
                        )
                        
                        m = Pipeline(steps=[
                            ('preprocessor', build_feature_preprocessor(numeric_features, categorical_features)),
                            ('model', stacking_model)
                        ])
                        
                        with parallel_backend('threading'):
                            m = self._fit_with_cancel(m, Xtr, ytr)
                        yp = m.predict(Xte)
                        met = compute_metrics(yte, yp)
                        
                        local_pred_cache[('Stacking', col)] = (yte, yp)
                        local_model_cache[('Stacking', col)] = m
                        local_X_cache[('Stacking', col)] = X_valid.copy()
                        
                        row = {'目标变量': col, '类别': cat_map.get(col, '未分类'), **met}
                        
                        # Use permutation importance for Stacking (more statistically sound than averaging child models)
                        pi_result = self._compute_permutation_importance(m, Xte, yte)
                        imp = normalize_importances(pi_result.importances_mean)
                        
                        for i, fn in enumerate(feat_names):
                            row[f'{fn}_RI(%)'] = round(imp[i] * 100, 2)
                                
                        rows.append(row)
                    local_all_res['Stacking'] = pd.DataFrame(rows)
                    self._log("   ✅ Stacking 融合模型完成")            
            # ── 综合分析 ──
            self._log(f"\n{'━' * 50}")
            self._log("📊  综合分析…")
            self._status("📊  综合…", 80)

            all_res = local_all_res

            comp_data = {'影响因素': feat_names}
            for mn, dfr in all_res.items():
                ri_vals = [dfr[f'{fn}_RI(%)'].mean() if f'{fn}_RI(%)' in dfr.columns else 0
                           for fn in feat_names]
                comp_data[f'{mn}_RI(%)'] = [round(v, 2) for v in ri_vals]
            df_comp = pd.DataFrame(comp_data)
            ri_c = [c for c in df_comp.columns if 'RI(%)' in c]
            df_comp['平均RI(%)'] = df_comp[ri_c].mean(axis=1).round(2)
            df_comp['排名'] = df_comp['平均RI(%)'].rank(ascending=False).astype(int)
            df_comp = df_comp.sort_values('排名')

            perf_rows = []
            for mn, dfr in all_res.items():
                good = dfr[(dfr['SCC'] > min_scc) & (dfr['PCC'] > min_pcc) & (dfr['R²'] > min_r2)]
                perf_rows.append({
                    '模型': mn, '目标变量数': len(dfr), '良好模型数': len(good),
                    '良好比例(%)': round(len(good) / max(len(dfr), 1) * 100, 1),
                    '平均R²': round(dfr['R²'].mean(), 4),
                    '平均SCC': round(dfr['SCC'].mean(), 4),
                    '平均PCC': round(dfr['PCC'].mean(), 4)})
            df_perf = pd.DataFrame(perf_rows).sort_values('平均SCC', ascending=False)

            cat_ri = []
            for cat in sorted(set(cat_map.values())):
                r = {'类别': cat}
                for mn, dfr in all_res.items():
                    cd = dfr[dfr['类别'] == cat]
                    if len(cd) > 0:
                        for fn in feat_names:
                            cn = f'{fn}_RI(%)'
                            if cn in cd.columns:
                                r[f'{mn}_{fn}_RI均值'] = round(cd[cn].mean(), 2)
                cat_ri.append(r)
            df_cat = pd.DataFrame(cat_ri)

            # ── 导出 Excel ──
            self._log(f"\n{'━' * 50}")
            self._log("💾  导出 Excel…")
            self._status("💾  导出…", 90)

            ts_ = datetime.now().strftime("%Y%m%d_%H%M%S")
            od = params['output_dir']
            od = f"{od}_{ts_}"
            os.makedirs(od, exist_ok=True)
            of = os.path.join(od, "分析结果.xlsx")
            df_config = reproducibility_dataframe(
                file_path=app.file_path,
                test_size=ts,
                preprocessing_applied=app._preprocessing_applied,
                log_transformed=app._log_transformed,
                selected_features=selected_features,
                selected_targets=target_cols,
                enabled_models=params['models_enabled'],
                output_dir=od,
            )

            with pd.ExcelWriter(of, engine='openpyxl') as w:
                df_config.to_excel(w, sheet_name="实验配置", index=False)
                df_sp.to_excel(w, sheet_name="Spearman相关性_全部", index=False)
                df_sp.pivot(index='目标变量', columns='影响因素', values='Spearman_r'
                            ).to_excel(w, sheet_name="Spearman_矩阵")
                df_sp.pivot(index='目标变量', columns='影响因素', values='显著性'
                            ).to_excel(w, sheet_name="Spearman_显著性")
                for mn, dfr in all_res.items():
                    dfr.to_excel(w, sheet_name=f"{mn}_全部"[:31], index=False)
                    g = dfr[(dfr['SCC'] > min_scc) & (dfr['PCC'] > min_pcc) & (dfr['R²'] > min_r2)]
                    if len(g) > 0:
                        g.to_excel(w, sheet_name=f"{mn}_良好"[:31], index=False)
                df_comp.to_excel(w, sheet_name="因素重要性_综合比较", index=False)
                df_perf.to_excel(w, sheet_name="模型性能汇总", index=False)
                if len(df_cat) > 0:
                    df_cat.to_excel(w, sheet_name="各类别_因素重要性", index=False)

            self._log(f"\n💾  结果已保存: {of}")
            self._log(f"\n{'═' * 70}\n🎉  分析完成!\n{'═' * 70}")

            self._log("\n📊  模型性能:")
            for _, r in df_perf.iterrows():
                self._log(f"   {r['模型']:15s} │ R²={r['平均R²']:.4f} │ "
                          f"SCC={r['平均SCC']:.4f} │ PCC={r['平均PCC']:.4f} │ "
                          f"良好率={r['良好比例(%)']:.1f}%")

            self._log("\n🏆  因素重要性排序:")
            for _, r in df_comp.iterrows():
                bar = "█" * int(r['平均RI(%)'] / 2)
                self._log(f"   {int(r['排名']):2d}. {r['影响因素']:20s} │ "
                          f"RI = {r['平均RI(%)']:6.1f}%  {bar}")

            self._safe_after(lambda: self._finalize_analysis(
                local_label_encoders,
                df_sp,
                local_all_res,
                local_pred_cache,
                local_model_cache,
                local_X_cache,
                df_comp,
                df_perf,
                of,
                len(target_cols),
                len(all_res),
            ))

        except InterruptedError:
            self._log("\n⏹ 分析已取消")
            self._status("⏹ 已取消", 0)
        except Exception as e:
            import logging
            logging.getLogger("PollutantApp").error("分析失败", exc_info=True)
            err_text = str(e)
            self._log(f"\n❌ 错误: {err_text}\n{traceback.format_exc()}")
            self._status("❌ 分析出错", 0)
            self._safe_after(lambda msg=err_text: show_message(app, "❌ 出错", msg, "error"))
        finally:
            try:
                self._safe_after(lambda: self.run_btn.configure(
                    state="normal", text="🚀  开 始 分 析"
                ))
                self._safe_after(lambda: self.cancel_btn.configure(
                    state="disabled", text="⏹ 取消"
                ))
            except Exception as e:
                logger.exception(f"分析线程发生异常: {e}")
            gc.collect()

    def _notify_visualization(self):
        if hasattr(self.app, 'tabs') and 'visualization' in self.app.tabs:
            # Optionally update UI in visualization tab if needed
            pass
