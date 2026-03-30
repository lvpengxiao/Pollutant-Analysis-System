"""
对话框模块
包含关于、使用说明、参数说明、更新日志和偏好设置对话框。
"""

import sys
import customtkinter as ctk

from .theme import (C, FONTS, APP_NAME, VERSION, BUILD_DATE,
                    make_btn_primary, make_btn_secondary, make_textbox,
                    make_optionmenu)


def _center(dialog, app, w, h):
    dialog.update_idletasks()
    x = app.winfo_x() + (app.winfo_width() - w) // 2
    y = app.winfo_y() + (app.winfo_height() - h) // 2
    dialog.geometry(f"{w}x{h}+{x}+{y}")


def _base_dialog(app, title, w=600, h=500):
    d = ctk.CTkToplevel(app)
    d.title(title)
    d.geometry(f"{w}x{h}")
    d.transient(app)
    d.grab_set()
    ct = ctk.CTkFrame(d, corner_radius=0, fg_color=C["bg_primary"])
    ct.pack(fill="both", expand=True)
    _center(d, app, w, h)
    return d, ct


# ── 关于 ──────────────────────────────────────────────────

def show_about(app):
    d, ct = _base_dialog(app, "关于", 500, 440)

    ctk.CTkLabel(ct, text=f"🔬  {APP_NAME}",
                 font=FONTS["h2"](),
                 text_color=C["accent_light"]).pack(pady=(30, 6))
    ctk.CTkLabel(ct, text=f"版本 {VERSION}",
                 font=FONTS["body"](),
                 text_color=C["text_primary"]).pack(pady=4)

    lines = [
        f"构建日期: {BUILD_DATE}",
        f"Python: {sys.version.split()[0]}",
        f"CustomTkinter: {ctk.__version__}",
        "",
        "功能模块:",
        "  数据加载 · 预处理 · 统计概览",
        "  特征选择 · 目标变量 · 模型参数",
        "  运行分析 · 交叉验证 · 可视化",
        "  配置管理 · 多格式导出",
    ]
    ctk.CTkLabel(ct, text="\n".join(lines), font=FONTS["body"](),
                 text_color=C["text_secondary"],
                 justify="left").pack(pady=16, padx=30)

    make_btn_primary(ct, text="确 定", width=120,
                     command=d.destroy).pack(pady=12)


# ── 使用说明 ──────────────────────────────────────────────

def show_user_guide(app):
    d, ct = _base_dialog(app, "使用说明", 700, 600)

    ctk.CTkLabel(ct, text="📖  使用说明", font=FONTS["h2"](),
                 text_color=C["text_primary"]).pack(pady=(16, 10))

    tb = make_textbox(ct, wrap="word")
    tb.pack(fill="both", expand=True, padx=22, pady=(0, 8))
    tb.insert("1.0", _GUIDE_TEXT)
    tb.configure(state="disabled")

    make_btn_primary(ct, text="关闭", width=100,
                     command=d.destroy).pack(pady=(0, 16))


_GUIDE_TEXT = """\
=== 使用流程 ===

1. 📂 数据 — 选择 Excel 文件，点击"加载数据"导入
2. 🧹 预处 — 处理缺失值、异常值、标准化、筛选（可选）
3. 📊 统计 — 查看描述性统计、数据质量报告、分布图
4. 📋 特征 — 勾选影响因素，指定数值型/分类型
5. 🎯 目标 — 勾选目标变量（污染物），填写类别标签
6. ⚙️ 模型 — 选择模型、调整超参数和阈值
7. 🚀 分析 — 设置输出目录，点击"开始分析"
8. 🔄 CV — 运行 K 折交叉验证，评估模型稳定性
9. 📈 图表 — 查看热力图、重要性图、对比图、散点图

=== 导出功能 ===

菜单栏 → 文件：
  · 导出 CSV — 分模型导出 CSV 文件
  · 导出 PDF — 生成带图表的分析报告
  · 导出日志 — 保存运行日志

=== 配置管理 ===

  · 保存配置 — 将特征选择、参数等保存为 JSON
  · 加载配置 — 从 JSON 恢复配置
  · 最近文件 — 快速打开最近使用的数据文件
  · 偏好设置 — 主题、缩放、字体大小
"""


# ── 参数说明 ──────────────────────────────────────────────

def show_param_help(app):
    d, ct = _base_dialog(app, "参数说明", 700, 550)

    ctk.CTkLabel(ct, text="📘  参数说明", font=FONTS["h2"](),
                 text_color=C["text_primary"]).pack(pady=(16, 10))

    tb = make_textbox(ct, wrap="word")
    tb.pack(fill="both", expand=True, padx=22, pady=(0, 8))
    tb.insert("1.0", _PARAM_TEXT)
    tb.configure(state="disabled")

    make_btn_primary(ct, text="关闭", width=100,
                     command=d.destroy).pack(pady=(0, 16))


_PARAM_TEXT = """\
=== 通用超参数 ===

▶ test_size  [0.1~0.5]  测试集占比
▶ n_estimators  [50~500]  树数量
▶ max_depth  [3~20]  最大深度，防止过拟合
▶ learning_rate  [0.01~0.3]  Boosting 学习率
▶ subsample  [0.5~1.0]  行采样比例（XGB/LGB）
▶ colsample  [0.3~1.0]  列采样比例（XGB/LGB）

=== 判定阈值 ===

▶ R²（决定系数）>0.5 优秀 >0.3 良好 >0.1 可接受
▶ SCC（Spearman）>0.5 强相关 >0.3 中等相关
▶ PCC（Pearson）>0.5 强相关 >0.3 中等相关

=== 预处理参数 ===

▶ IQR 法: [Q1-1.5·IQR, Q3+1.5·IQR] 范围外为异常
▶ Z-score 法: |z| > 阈值为异常
▶ StandardScaler: z = (x-μ)/σ
▶ MinMaxScaler: x' = (x-min)/(max-min)

=== 交叉验证 ===

▶ K=5 最常用，K=3 适合小数据，K=10 更精确
"""


# ── 更新日志 ──────────────────────────────────────────────

def show_changelog(app):
    d, ct = _base_dialog(app, "更新日志", 560, 480)

    ctk.CTkLabel(ct, text="📝  更新日志", font=FONTS["h2"](),
                 text_color=C["text_primary"]).pack(pady=(16, 10))

    tb = make_textbox(ct, wrap="word")
    tb.pack(fill="both", expand=True, padx=22, pady=(0, 8))
    tb.insert("1.0", _CHANGELOG)
    tb.configure(state="disabled")

    make_btn_primary(ct, text="关闭", width=100,
                     command=d.destroy).pack(pady=(0, 16))


_CHANGELOG = """\
=== v1.0 (2025-03) ===

[新增] 模块化代码架构
[新增] 全新 Indigo 深色主题 · 卡片布局 · 自定义菜单栏
[新增] 数据预处理：缺失值 / 异常值 / 标准化 / 筛选
[新增] 数据统计：描述性统计 / 质量报告 / 分布图
[新增] 数据可视化：热力图 / 重要性 / 对比 / 散点
[新增] K 折交叉验证 + 稳定性箱线图
[新增] 自定义暗色菜单栏 + 底部状态栏
[新增] 配置保存/加载 (JSON) · 最近文件 · 偏好设置
[新增] 导出 CSV / PDF 报告 / 日志
[新增] 帮助：使用说明 / 参数说明 / 更新日志 / 关于

=== v0.9 (2024) ===

  初始版本
  5 个标签页 · 6 种 ML 模型 · Excel 导出
"""


# ── 偏好设置 ──────────────────────────────────────────────

def show_preferences(app):
    d, ct = _base_dialog(app, "偏好设置", 500, 380)

    ctk.CTkLabel(ct, text="⚙️  偏好设置", font=FONTS["h2"](),
                 text_color=C["text_primary"]).pack(pady=(22, 18))

    # 主题
    r1 = ctk.CTkFrame(ct, fg_color="transparent")
    r1.pack(fill="x", padx=36, pady=8)
    ctk.CTkLabel(r1, text="外观主题:", width=120, anchor="w",
                 font=FONTS["body"](), text_color=C["text_primary"]).pack(side="left")
    theme_v = ctk.StringVar(value=app.user_prefs.get('theme', 'Dark'))
    make_optionmenu(r1, variable=theme_v, width=180,
                    values=["Dark", "Light"]).pack(side="left", padx=10)

    # 缩放
    r2 = ctk.CTkFrame(ct, fg_color="transparent")
    r2.pack(fill="x", padx=36, pady=8)
    ctk.CTkLabel(r2, text="UI 缩放:", width=120, anchor="w",
                 font=FONTS["body"](), text_color=C["text_primary"]).pack(side="left")
    scale_v = ctk.IntVar(value=app.user_prefs.get('scaling', 100))
    ctk.CTkSlider(r2, from_=80, to=150, number_of_steps=14,
                  variable=scale_v, width=180,
                  fg_color=C["bg_tertiary"], progress_color=C["accent"],
                  button_color=C["accent"],
                  button_hover_color=C["accent_hover"]).pack(side="left", padx=10)
    sl = ctk.CTkLabel(r2, text=f"{scale_v.get()}%", font=FONTS["body"](),
                      text_color=C["text_primary"])
    sl.pack(side="left", padx=6)
    scale_v.trace_add('write', lambda *_: sl.configure(text=f"{scale_v.get()}%"))

    # 字体
    r3 = ctk.CTkFrame(ct, fg_color="transparent")
    r3.pack(fill="x", padx=36, pady=8)
    ctk.CTkLabel(r3, text="字体大小:", width=120, anchor="w",
                 font=FONTS["body"](), text_color=C["text_primary"]).pack(side="left")
    font_v = ctk.IntVar(value=app.user_prefs.get('font_size', 13))
    make_optionmenu(r3, variable=font_v, width=100,
                    values=["10", "11", "12", "13", "14", "15", "16"],
                    command=lambda v: font_v.set(int(v))).pack(side="left", padx=10)

    def _apply():
        new_theme = theme_v.get()
        new_scale = scale_v.get()
        app.user_prefs['theme'] = new_theme
        app.user_prefs['scaling'] = new_scale
        app.user_prefs['font_size'] = font_v.get()
        app.config_manager._save_prefs()
        d.destroy()
        app.switch_theme(new_theme, new_scale)

    bf = ctk.CTkFrame(ct, fg_color="transparent")
    bf.pack(pady=22)
    make_btn_secondary(bf, text="取消", width=100,
                       command=d.destroy).pack(side="left", padx=10)
    make_btn_primary(bf, text="应用并保存", width=130,
                     command=_apply).pack(side="left", padx=10)
