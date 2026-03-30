"""
预测与模拟工具标签页模块 (🔮 模拟)
---------------------------------------------------------------------
What-If 情景模拟工具，允许用户在训练好的模型上进行交互式推理。

主要特性:
1. 模型读取：自动从 `app.model_cache` 提取已拟合的基模型。
2. 动态 UI：根据模型的特征列表，动态生成对应范围的滑动条 (Numeric) 或下拉框 (Categorical)。
3. 实时推理：滑动条拖动即触发预测，并自动执行指数反变换还原为实际浓度单位。
4. 特殊适配：处理了 GAM 的 `.values` 输入格式及不支持 `.predict` 的保护逻辑。
"""

import logging

import numpy as np
import pandas as pd
import customtkinter as ctk

from constants import get_app_log_path
from .theme import (C, FONTS, show_message, make_card, make_inner_frame,
                    make_section_title, make_optionmenu, make_scrollframe,
                    make_btn_primary, make_btn_secondary, make_empty_state)


logger = logging.getLogger(__name__)


class SimulationTab:

    def __init__(self, parent, app):
        self.app = app
        self.parent = parent
        self.sliders = {}
        self.current_features = []
        self._build()

    def _build(self):
        # 顶部控制区
        top_card = make_card(self.parent)
        top_card.pack(fill="x", padx=18, pady=(14, 6))
        
        row = make_inner_frame(top_card)
        row.pack(fill="x", padx=18, pady=14)
        
        ctk.CTkLabel(row, text="选择已训练模型:", font=FONTS["body_bold"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(0, 8))
        self.sim_model = ctk.StringVar(value="RandomForest")
        make_optionmenu(
            row, variable=self.sim_model, width=160,
            values=["RandomForest", "AdaBoost", "XGBoost", "LightGBM", "CatBoost", "GAM"]
        ).pack(side="left", padx=4)
        
        ctk.CTkLabel(row, text="目标变量:", font=FONTS["body_bold"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(20, 8))
        self.sim_target = ctk.StringVar(value="")
        self.target_menu = make_optionmenu(
            row, variable=self.sim_target, width=160,
            values=["(请先运行分析)"]
        )
        self.target_menu.pack(side="left", padx=4)
        
        self.load_btn = make_btn_primary(row, text="加载模型", width=100, command=self._load_model)
        self.load_btn.pack(side="left", padx=20)
        self.reset_btn = make_btn_secondary(row, text="↩️ 重置", width=90, command=self._reset_sliders)
        self.reset_btn.pack(side="left", padx=8)
        self.reset_btn.configure(state="disabled")
        
        # 预测结果展示区
        res_card = make_card(self.parent)
        res_card.pack(fill="x", padx=18, pady=6)
        res_inner = make_inner_frame(res_card)
        res_inner.pack(fill="x", padx=18, pady=20)
        
        ctk.CTkLabel(res_inner, text="预测结果 (log10处理前):", font=FONTS["h2"](),
                     text_color=C["text_primary"]).pack(side="left", padx=10)
                     
        self.pred_label = ctk.CTkLabel(res_inner, text="---", font=ctk.CTkFont(size=28, weight="bold"),
                                       text_color=C["success"])
        self.pred_label.pack(side="left", padx=20)

        # 特征滑动条区
        self.slider_frame = make_scrollframe(self.parent)
        self.slider_frame.pack(fill="both", expand=True, padx=12, pady=10)
        self.empty_state = make_empty_state(
            self.slider_frame,
            "🔮",
            "暂无模拟面板",
            "请先在【🚀 分析】页面生成模型结果，随后选择模型与目标变量并点击“加载模型”。",
            button_text="前往分析页",
            command=lambda: self.app.navigate_to_tab("analysis", force=True) if hasattr(self.app, 'navigate_to_tab') else None,
        )
        self.empty_state.pack(fill="both", expand=True, padx=8, pady=8)
        self.refresh_empty_state()
        
    def populate_targets(self):
        if not hasattr(self.app, 'model_cache') or not self.app.model_cache:
            self.target_menu.configure(values=["(请先运行分析)"])
            self.sim_target.set("(请先运行分析)")
            self.refresh_empty_state()
            return
        # 提取所有可用的目标变量
        targets = list(set([k[1] for k in self.app.model_cache.keys()]))
        if targets:
            self.target_menu.configure(values=targets)
            self.sim_target.set(targets[0])
        self.refresh_empty_state()

    def _show_empty_state(self):
        if self.empty_state is not None and not self.empty_state.winfo_manager():
            self.empty_state.pack(fill="both", expand=True, padx=8, pady=8)

    def _hide_empty_state(self):
        if self.empty_state is not None and self.empty_state.winfo_manager():
            self.empty_state.pack_forget()

    def refresh_empty_state(self):
        if not hasattr(self.app, 'model_cache') or not self.app.model_cache:
            self.empty_state.title_label.configure(text="暂无模拟面板")
            self.empty_state.message_label.configure(
                text="请先在【🚀 分析】页面生成模型结果，随后这里会解锁情景模拟。"
            )
            if self.empty_state.action_button is not None:
                self.empty_state.action_button.configure(
                    text="前往分析页",
                    command=lambda: self.app.navigate_to_tab("analysis", force=True) if hasattr(self.app, 'navigate_to_tab') else None,
                )
                if not self.empty_state.action_button.winfo_manager():
                    self.empty_state.action_button.pack(pady=(0, 24))
            return

        self.empty_state.title_label.configure(text="准备加载模型")
        self.empty_state.message_label.configure(
            text="模型缓存已准备好。选择模型与目标变量后，点击上方“加载模型”开始情景模拟。"
        )
        if self.empty_state.action_button is not None and self.empty_state.action_button.winfo_manager():
            self.empty_state.action_button.pack_forget()

    def _get_feature_type(self, feat):
        info = self.app.feature_vars.get(feat)
        if isinstance(info, dict) and 'type' in info:
            return info['type'].get()
        return "numeric"

    def _format_choice(self, value):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.4f}".rstrip("0").rstrip(".")

    def _load_model(self):
        if not hasattr(self.app, 'model_cache') or not self.app.model_cache:
            show_message(self.app, "ℹ️ 提示", "请先在【🚀 分析】页面运行模型分析", "info")
            return
            
        mn = self.sim_model.get()
        tgt = self.sim_target.get()
        
        # 兼容旧缓存 key 的结构
        key = None
        for k in self.app.model_cache.keys():
            if isinstance(k, tuple) and k[0] == mn and k[1] == tgt:
                key = k
                break
        
        if key is None:
            show_message(self.app, "❌ 错误", f"未找到 {mn} 预测 {tgt} 的模型缓存", "error")
            return
            
        self.model = self.app.model_cache[key]
        self.X_data = self.app.X_cache.get(key, self.app.X_cache.get(tgt))
        if self.X_data is None:
            show_message(self.app, "❌ 错误", "找不到特征数据缓存", "error")
            return
        self.current_features = self.X_data.columns.tolist()
        
        # 清空现有的滑动条
        for widget in self.slider_frame.winfo_children():
            if widget is not self.empty_state:
                widget.destroy()
        self.sliders.clear()
        self.value_labels = {}
        self._hide_empty_state()
        self.reset_btn.configure(state="normal")
        
        # 为每个特征创建滑动条
        self.default_values = {}
        for feat in self.current_features:
            row = ctk.CTkFrame(self.slider_frame, fg_color=C["bg_tertiary"], corner_radius=8)
            row.pack(fill="x", padx=10, pady=5)
            
            # 左侧标签
            lbl = ctk.CTkLabel(row, text=f"{feat[:20]}:", width=150, anchor="w",
                               font=FONTS["body"](), text_color=C["text_primary"])
            lbl.pack(side="left", padx=10, pady=10)

            feat_type = self._get_feature_type(feat)
            if feat_type == "categorical":
                unique_vals = self.X_data[feat].dropna().unique().tolist()
                
                # If they are numbers represented as strings, maybe sort them, else sort alphabetically
                try:
                    unique_vals = sorted(unique_vals, key=float)
                except ValueError:
                    unique_vals = sorted(unique_vals)
                    
                default_val = self.X_data[feat].mode().iloc[0] if not self.X_data[feat].mode().empty else unique_vals[0]
                
                option_values = [str(v) for v in unique_vals]
                default_display = str(default_val)
                
                val_var = ctk.StringVar(value=default_display)
                val_lbl = ctk.CTkLabel(row, text=val_var.get()[:10], width=80,
                                       font=FONTS["mono"](), text_color=C["accent_light"])
                val_lbl.pack(side="left", padx=5)
                make_optionmenu(
                    row,
                    variable=val_var,
                    width=180,
                    values=option_values,
                    command=lambda v, l=val_lbl: self._on_category_change(v, l)
                ).pack(side="left", padx=20)
                ctk.CTkLabel(
                    row,
                    text=f"{len(option_values)} 个类别",
                    width=120,
                    font=FONTS["small"](),
                    text_color=C["text_muted"]
                ).pack(side="left", padx=10)
                self.sliders[feat] = val_var
                self.default_values[feat] = default_display
                self.value_labels[feat] = val_lbl
            else:
                # 获取特征的最小值、最大值和均值
                min_val = float(self.X_data[feat].min())
                max_val = float(self.X_data[feat].max())
                mean_val = float(self.X_data[feat].mean())
                
                if min_val == max_val:
                    max_val = min_val + 1.0 # 避免滑动条异常
                    
                val_lbl = ctk.CTkLabel(row, text=f"{mean_val:.2f}", width=60,
                                       font=FONTS["mono"](), text_color=C["accent_light"])
                val_lbl.pack(side="left", padx=5)
                slider_var = ctk.DoubleVar(value=mean_val)
                slider = ctk.CTkSlider(
                    row, from_=min_val, to=max_val, variable=slider_var,
                    command=lambda v, l=val_lbl: self._on_slider_change(v, l)
                )
                slider.pack(side="left", fill="x", expand=True, padx=20)
                range_lbl = ctk.CTkLabel(row, text=f"[{min_val:.2f} ~ {max_val:.2f}]", width=120,
                                         font=FONTS["small"](), text_color=C["text_muted"])
                range_lbl.pack(side="left", padx=10)
                self.sliders[feat] = slider_var
                self.default_values[feat] = mean_val
                self.value_labels[feat] = val_lbl
            
        # 初始预测
        self._predict()
        show_message(self.app, "✅ 成功", f"成功加载 {mn} 模型，请拖动下方滑块进行情景模拟。", "info")

    def _reset_sliders(self):
        if not hasattr(self, 'default_values'):
            return
        for feat, value in self.default_values.items():
            if feat in self.sliders:
                self.sliders[feat].set(value)
            if feat in self.value_labels:
                if isinstance(value, str):
                    self.value_labels[feat].configure(text=value)
                else:
                    self.value_labels[feat].configure(text=f"{value:.2f}")
        self._predict()

    def _on_slider_change(self, value, label_widget):
        label_widget.configure(text=f"{value:.2f}")
        self._predict()

    def _on_category_change(self, value, label_widget):
        label_widget.configure(text=value)
        self._predict()
        
    def _predict(self):
        if not hasattr(self, 'model') or not self.current_features:
            return
            
        # 收集当前特征值
        input_data = {}
        for feat in self.current_features:
            raw_value = self.sliders[feat].get()
            
            if self._get_feature_type(feat) == "categorical":
                # For Pipeline with OneHotEncoder, just pass the raw string
                input_data[feat] = [raw_value]
            else:
                try:
                    input_data[feat] = [float(raw_value)]
                except ValueError:
                    input_data[feat] = [0.0]
            
        X_pred = pd.DataFrame(input_data)
        
        try:
            mn = self.sim_model.get()
            
            # 由于加入了 Pipeline，如果是 RandomForest 并且想要获取置信区间，需要提取内部模型
            if mn == "RandomForest":
                actual_model = None
                if hasattr(self.model, 'named_steps') and 'model' in self.model.named_steps:
                    actual_model = self.model.named_steps['model']
                    preprocessor = self.model.named_steps['preprocessor']
                    X_pred_transformed = preprocessor.transform(X_pred)
                elif hasattr(self.model, "estimators_"):
                    actual_model = self.model
                    X_pred_transformed = X_pred.values if hasattr(X_pred, 'values') else X_pred
                
                if actual_model is not None and hasattr(actual_model, "estimators_"):
                    tree_preds = np.array([tree.predict(X_pred_transformed)[0] for tree in actual_model.estimators_])
                    log_pred = float(np.mean(tree_preds))
                    log_std = float(np.std(tree_preds))
                    
                    if hasattr(self.app, '_log_transformed') and self.app._log_transformed:
                        real_pred = log_pred
                        real_lower = log_pred - 1.96 * log_std
                        real_upper = log_pred + 1.96 * log_std
                    else:
                        real_pred = max(0, (10 ** log_pred) - 1)
                        real_lower = max(0, (10 ** (log_pred - 1.96 * log_std)) - 1)
                        real_upper = max(0, (10 ** (log_pred + 1.96 * log_std)) - 1)
                        
                    self.pred_label.configure(text=f"{real_pred:.4f}  [{real_lower:.4f} ~ {real_upper:.4f}]")
                    return

            if mn == "GAM":
                gam_model = self.model['model'] if isinstance(self.model, dict) else self.model
                gam_le = self.model.get('le', {}) if isinstance(self.model, dict) else {}
                
                # Transform categorical features for GAM
                for feat in self.current_features:
                    if self._get_feature_type(feat) == "categorical":
                        le = gam_le.get(feat)
                        if le:
                            try:
                                X_pred[feat] = le.transform([str(X_pred[feat].iloc[0])])[0]
                            except Exception:
                                X_pred[feat] = 0
                                
                # GAM predict expects numpy array
                if hasattr(gam_model, 'predict'):
                    log_pred = gam_model.predict(X_pred.values)[0]
                else:
                    self.pred_label.configure(text="模型不支持预测", text_color=C["error"])
                    return
            else:
                if hasattr(self.model, 'predict'):
                    log_pred = self.model.predict(X_pred)[0]
                else:
                    self.pred_label.configure(text="模型不支持预测", text_color=C["error"])
                    return
                
            # 还原log10 (因为训练时可能用了 np.log10(y + 1))
            if hasattr(self.app, '_log_transformed') and self.app._log_transformed:
                # 数据已经在预处理中变换过，模型预测的是变换后的值
                # 预处理可能是 Box-Cox 等，这里为了简化，直接显示预测值，并提示用户
                pred_val = log_pred
                unit = "(已在预处理变换)"
            else:
                pred_val = max(0, (10 ** log_pred) - 1)
                unit = "μg/m³ (或其他单位)"
            
            self.pred_label.configure(
                text=f"{pred_val:.4f} {unit}",
                text_color=C["accent_light"]
            )
        except Exception as e:
            self.pred_label.configure(text="预测出错")
            print(f"预测出错: {e}")
