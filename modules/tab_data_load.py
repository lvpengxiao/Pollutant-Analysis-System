"""
数据加载标签页模块
提供 Excel 文件选择、数据导入和数据预览功能。
"""

import os
import pandas as pd
import customtkinter as ctk
from tkinter import filedialog

from .theme import C, FONTS, show_message, make_card, make_inner_frame
from .theme import make_section_title, make_entry, make_btn_primary
from .theme import make_btn_secondary, make_textbox


class DataLoadTab:

    def __init__(self, parent, app):
        self.app = app
        self.parent = parent
        self.file_entry = None
        self.preview_textbox = None
        self._build()

    # ── UI 构建 ──────────────────────────────────────────

    def _build(self):
        # 文件选择卡片
        card = make_card(self.parent)
        card.pack(fill="x", padx=18, pady=(14, 6))

        make_section_title(card, "选择数据文件", icon="📁").pack(
            anchor="w", padx=18, pady=(16, 8))

        row = make_inner_frame(card)
        row.pack(fill="x", padx=18, pady=(0, 16))

        self.file_entry = make_entry(
            row, placeholder_text="点击右侧按钮选择 .xlsx / .xls / .csv / .tsv 文件…")
        self.file_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        make_btn_secondary(row, text="浏览文件", width=110,
                           command=self._browse).pack(side="left", padx=(0, 8))
        make_btn_primary(row, text="✓  加载数据", width=130,
                         command=self.load_data).pack(side="left")

        # 数据预览卡片
        card2 = make_card(self.parent)
        card2.pack(fill="both", expand=True, padx=18, pady=(6, 14))

        make_section_title(card2, "数据预览", icon="📋").pack(
            anchor="w", padx=18, pady=(16, 8))

        self.preview_textbox = make_textbox(card2, wrap="none")
        self.preview_textbox.pack(fill="both", expand=True, padx=18, pady=(0, 16))

    # ── 事件处理 ──────────────────────────────────────────

    def _browse(self):
        filename = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[
                ("所有支持格式", "*.xlsx *.xls *.csv *.tsv"),
                ("Excel文件", "*.xlsx *.xls"),
                ("CSV文件", "*.csv"),
                ("TSV文件", "*.tsv"),
                ("所有文件", "*.*"),
            ])
        if filename:
            self.file_entry.delete(0, "end")
            self.file_entry.insert(0, filename)
            self.app.file_path = filename

    def _auto_suggest(self, df):
        pollutant_keywords = [
            'conc', 'concentration', '浓度', 'level', 'voc', 'pah', 'pcb',
            'pbde', 'pm', 'tvoc', 'formaldehyde', '甲醛', 'benzene', '苯',
            'toluene', '甲苯', 'acetaldehyde', '乙醛'
        ]
        suggested_targets = []
        suggested_features = []
        for col in df.columns:
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in pollutant_keywords):
                suggested_targets.append(str(col))
            else:
                suggested_features.append(str(col))
        lines = []
        if suggested_targets:
            lines.append(
                f"💡 可能的目标变量 ({len(suggested_targets)} 个): "
                + ", ".join(suggested_targets[:8])
            )
        if suggested_features:
            lines.append(
                f"💡 可能的影响因素 ({len(suggested_features)} 个): "
                + ", ".join(suggested_features[:8])
            )
        return "\n".join(lines)

    def load_data(self):
        path = self.file_entry.get().strip()
        if not path:
            show_message(self.app, "❌ 错误", "请先选择数据文件！", "error")
            return

        self.app.file_path = path
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".csv":
                # 自动尝试多种编码
                try:
                    import chardet
                    with open(path, 'rb') as f:
                        raw = f.read(min(10000, os.path.getsize(path)))
                    detected = chardet.detect(raw)
                    encoding = detected['encoding'] or 'utf-8'
                    self.app.df = pd.read_csv(path, encoding=encoding)
                except ImportError:
                    for enc in ("utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"):
                        try:
                            self.app.df = pd.read_csv(path, encoding=enc)
                            break
                        except (UnicodeDecodeError, UnicodeError):
                            continue
                    else:
                        show_message(self.app, "❌ 错误", "无法识别文件编码", "error")
                        return
            elif ext == ".tsv":
                self.app.df = pd.read_csv(path, sep="\t", encoding="utf-8")
            elif ext in (".xlsx", ".xls"):
                self.app.df = pd.read_excel(path, header=0)
            else:
                show_message(self.app, "❌ 错误", f"不支持的文件格式: {ext}", "error")
                return
            self.app.df_backup = self.app.df.copy()
            self.app.all_columns = self.app.df.columns.tolist()

            df = self.app.df
            preview = f"✅ 数据加载成功!\n\n"
            preview += f"📊  行数: {len(df)}    列数: {len(df.columns)}\n\n"
            preview += "📋  列名列表:\n"
            for i, col in enumerate(df.columns, 1):
                dtype = str(df[col].dtype)
                nn = df[col].notna().sum()
                preview += f"  {i:3d}. {col:30s}  [{dtype}]  非空: {nn}/{len(df)}\n"
            preview += f"\n{'═' * 80}\n前 5 行数据预览:\n{'═' * 80}\n"
            preview += df.head().to_string(max_colwidth=20)
            suggestions = self._auto_suggest(df)
            if suggestions:
                preview += f"\n\n{'═' * 80}\n智能推荐:\n{'═' * 80}\n{suggestions}"

            self.preview_textbox.delete("1.0", "end")
            self.preview_textbox.insert("1.0", preview)

            # 通知其他模块刷新
            if hasattr(self.app, 'event_bus'):
                self.app.event_bus.publish('data_shape_changed')
                if hasattr(self.app, 'tabs') and 'preprocess' in self.app.tabs:
                    self.app.tabs['preprocess'].refresh_columns()
                    self.app.tabs['preprocess'].check_unlock()
            else:
                if hasattr(self.app, 'tabs'):
                    tabs = self.app.tabs
                    if 'features' in tabs:
                        tabs['features'].populate()
                    if 'targets' in tabs:
                        tabs['targets'].populate()
                    if 'preprocess' in tabs:
                        tabs['preprocess'].refresh_columns()
                    if 'statistics' in tabs:
                        tabs['statistics'].enable_buttons()

            self.app.update_status_bar()
            self.app.add_recent_file(path)
            if hasattr(self.app, 'refresh_navigation_state'):
                self.app.refresh_navigation_state()
            if hasattr(self.app, 'navigate_to_tab'):
                self.app.navigate_to_tab("statistics", force=True)
            else:
                self.app.tabview.set("📊 统计")

            show_message(self.app, "✅ 加载成功",
                         f"共 {len(df)} 行, {len(df.columns)} 列\n"
                         f"已自动跳转到统计页面\n\n"
                         f"{suggestions or '可继续选择特征和目标变量'}", "info")

        except Exception as e:
            show_message(self.app, "❌ 加载失败", str(e), "error")
