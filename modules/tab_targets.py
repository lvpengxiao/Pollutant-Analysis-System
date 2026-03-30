"""
目标变量标签页模块
允许用户勾选目标变量(污染物)列并为每个目标指定类别标签。
"""

import customtkinter as ctk

from .theme import (C, FONTS, make_card, make_inner_frame, make_hint,
                    make_btn_secondary, make_entry, make_checkbox,
                    make_scrollframe, make_optionmenu)


class TargetsTab:

    def __init__(self, parent, app):
        self.app = app
        self.parent = parent
        self.row_frames = {}
        self.current_sort = "original" # "original", "selected_first", "name_asc"
        self._build()
        if hasattr(self.app, 'event_bus'):
            self.app.event_bus.subscribe('data_shape_changed', self.populate)

    def _build(self):
        ctk.CTkLabel(
            self.parent, text="🎯  勾选目标变量(污染物)列，并指定类别",
            font=FONTS["h3"](), text_color=C["text_primary"]
        ).pack(anchor="w", padx=22, pady=(14, 4))

        make_hint(self.parent,
                  '类别可自定义（如 VOCs、PAHs、重金属），留空归为 "未分类"'
                  ).pack(anchor="w", padx=22, pady=(0, 8))

        bar = ctk.CTkFrame(self.parent, fg_color="transparent")
        bar.pack(fill="x", padx=22, pady=(0, 6))
        make_btn_secondary(bar, text="全选", width=80,
                           command=lambda: self._toggle(True)).pack(side="left", padx=(0, 8))
        make_btn_secondary(bar, text="取消全选", width=80,
                           command=lambda: self._toggle(False)).pack(side="left")

        search_bar = ctk.CTkFrame(self.parent, fg_color="transparent")
        search_bar.pack(fill="x", padx=22, pady=(0, 6))
        ctk.CTkLabel(search_bar, text="🔍 搜索:", font=FONTS["body"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(0, 6))
        self.search_var = ctk.StringVar(value="")
        self.search_var.trace_add("write", lambda *_: self._apply_sort_and_search())
        make_entry(
            search_bar,
            textvariable=self.search_var,
            placeholder_text="输入列名关键词过滤...",
            width=200
        ).pack(side="left", padx=4)
        
        # 新增排序控件
        ctk.CTkLabel(search_bar, text="📶 排序:", font=FONTS["body"](),
                     text_color=C["text_primary"]).pack(side="left", padx=(16, 6))
        
        self.sort_var = ctk.StringVar(value="原始顺序")
        self.sort_mapping = {
            "原始顺序": "original",
            "已勾选靠前": "selected_first",
            "名称 (A-Z)": "name_asc"
        }
        
        def on_sort_change(val):
            self.current_sort = self.sort_mapping[val]
            self._apply_sort_and_search()
            
        make_optionmenu(
            search_bar,
            variable=self.sort_var,
            values=list(self.sort_mapping.keys()),
            width=120,
            command=on_sort_change
        ).pack(side="left", padx=4)

        self.summary_label = ctk.CTkLabel(
            self.parent,
            text="共 0 列 · 已选 0 个 · 当前显示 0 个",
            font=FONTS["small"](),
            text_color=C["text_secondary"],
            anchor="w",
        )
        self.summary_label.pack(fill="x", padx=22, pady=(0, 2))

        self.next_step_label = ctk.CTkLabel(
            self.parent,
            text="请先加载数据，然后勾选分析目标。",
            font=FONTS["small"](),
            text_color=C["text_muted"],
            anchor="w",
        )
        self.next_step_label.pack(fill="x", padx=22, pady=(0, 8))

        self.scroll = make_scrollframe(
            self.parent, fg_color=C["bg_secondary"])
        self.scroll.pack(fill="both", expand=True, padx=18, pady=(0, 14))

    def _update_summary(self, visible_count=None):
        total_count = len(self.app.all_columns or [])
        selected_count = sum(1 for var in self.app.target_vars.values() if var.get())
        if visible_count is None:
            visible_count = sum(1 for row in self.row_frames.values() if row.winfo_manager())
        self.summary_label.configure(
            text=f"共 {total_count} 列 · 已选 {selected_count} 个 · 当前显示 {visible_count} 个"
        )

        if self.app.df is None:
            hint = "请先在【📂 加载】页面导入数据文件。"
            color = C["text_muted"]
        elif selected_count == 0:
            hint = "下一步：勾选需要建模或评估的目标变量。"
            color = C["warning"]
        else:
            feature_count = len(self.app.get_selected_features()) if hasattr(self.app, 'get_selected_features') else 0
            if feature_count == 0:
                hint = f"✅ 已选 {selected_count} 个目标，下一步：前往【📋 特征】页面选择影响因素。"
                color = C["info"]
            else:
                hint = f"✅ 已选 {feature_count} 个特征、{selected_count} 个目标，下一步：前往【⚙️ 参数】或【🚀 分析】继续。"
                color = C["success"]
        self.next_step_label.configure(text=hint, text_color=color)

    def _add_header(self):
        """创建表头行（"选择 / 列名 / 污染物类别"）。在主题切换时会被重新创建。"""
        hdr = ctk.CTkFrame(self.scroll, fg_color=C["bg_tertiary"], corner_radius=8)
        hdr.pack(fill="x", padx=6, pady=(6, 8))
        ctk.CTkLabel(hdr, text="选择", width=55, font=FONTS["body_bold"](),
                     text_color=C["text_primary"]).pack(side="left", padx=10)
        ctk.CTkLabel(hdr, text="列名", width=300, font=FONTS["body_bold"](),
                     text_color=C["text_primary"], anchor="w").pack(side="left", padx=10)
        ctk.CTkLabel(hdr, text="污染物类别", width=200, font=FONTS["body_bold"](),
                     text_color=C["text_primary"]).pack(side="left", padx=10)

    def populate(self):
        """重新加载目标变量列表（包括重建表头以支持主题切换）。"""
        # 尝试保留之前的状态
        old_selections = {}
        old_categories = {}
        for col, v in self.app.target_vars.items():
            old_selections[col] = v.get()
        for col, v in self.app.category_vars.items():
            old_categories[col] = v.get()

        # 清空滚动框中的所有内容，包括表头
        children = self.scroll.winfo_children()
        for ch in children:
            ch.destroy()
        self.app.target_vars.clear()
        self.app.category_vars.clear()
        self.row_frames.clear()
        
        # 重建表头（这样主题切换时颜色会正确更新）
        self._add_header()

        for i, col in enumerate(self.app.all_columns):
            bg = C["bg_tertiary"] if i % 2 == 0 else C["bg_secondary"]
            row = ctk.CTkFrame(self.scroll, fg_color=bg, corner_radius=6)
            row.pack(fill="x", padx=6, pady=1)

            is_selected = old_selections.get(col, False)
            cat_val = old_categories.get(col, "")

            sel = ctk.BooleanVar(value=is_selected)
            sel.trace_add("write", lambda *_: self._on_selection_changed())
            make_checkbox(row, text="", variable=sel, width=40).pack(
                side="left", padx=(14, 4), pady=5)

            ctk.CTkLabel(row, text=col, width=300, font=FONTS["body"](),
                         text_color=C["text_primary"], anchor="w").pack(side="left", padx=10)

            cat = ctk.StringVar(value=cat_val)
            make_entry(row, textvariable=cat, width=190, height=30,
                       placeholder_text="输入类别 (如 VOCs)…").pack(side="left", padx=10)

            self.app.target_vars[col] = sel
            self.app.category_vars[col] = cat
            self.row_frames[col] = row
        self._apply_sort_and_search()

    def _toggle(self, state):
        for v in self.app.target_vars.values():
            v.set(state)
        if self.current_sort == "selected_first":
            self._apply_sort_and_search()

    def _apply_sort_and_search(self):
        keyword = self.search_var.get().strip().lower()
        
        # 1. 决定顺序
        sorted_cols = list(self.app.all_columns)
        if self.current_sort == "selected_first":
            sorted_cols.sort(key=lambda col: not self.app.target_vars[col].get())
        elif self.current_sort == "name_asc":
            sorted_cols.sort()
            
        # 2. 移除所有
        for col, row in self.row_frames.items():
            if row.winfo_manager():
                row.pack_forget()

        # 3. 按新顺序放置
        for col in sorted_cols:
            if col not in self.row_frames:
                continue
            row = self.row_frames[col]
            if not keyword or keyword in col.lower():
                row.pack(fill="x", padx=6, pady=1)
        visible_count = sum(
            1 for col in sorted_cols
            if col in self.row_frames and (not keyword or keyword in col.lower())
        )
        self._update_summary(visible_count)

    def _on_selection_changed(self):
        if hasattr(self.app, 'tabs') and 'preprocess' in self.app.tabs:
            self.app.tabs['preprocess'].check_unlock()
        if hasattr(self.app, 'refresh_navigation_state'):
            self.app.refresh_navigation_state()
        if self.current_sort == "selected_first":
            self._apply_sort_and_search()
        else:
            self._update_summary()
