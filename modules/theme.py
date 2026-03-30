"""
主题配置模块
定义 Dark / Light 双配色方案、字体规格和可复用 UI 组件工厂函数。
"""

import logging
import os
import json
"""
UI 主题和组件工厂
---------------------------------------------------------------------
⚠️ 注意：C 字典中的颜色值在主题切换时会被整体清空并重新填充。
    因此：
    1. 所有 UI 组件如果需要支持动态主题切换，必须在 `switch_theme()` 中被销毁并重建。
    2. 不要将 C["xxx"] 的值缓存到类实例变量中，应该在需要时直接读取 C["xxx"]。
"""

import customtkinter as ctk

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  全局常量
# ═══════════════════════════════════════════════════════════════

VERSION = "1.0"
BUILD_DATE = "2025-03"
APP_NAME = "新污染物影响因素相对重要性分析系统"

# ═══════════════════════════════════════════════════════════════
#  双配色方案
# ═══════════════════════════════════════════════════════════════

_DARK = {
    "bg_base":       "#1a1a20",
    "bg_primary":    "#222229",
    "bg_secondary":  "#2b2b33",
    "bg_tertiary":   "#35353f",
    "bg_hover":      "#40404c",
    "bg_active":     "#4b4b58",
    "bg_input":      "#1e1e25",

    "accent":        "#6366f1",
    "accent_hover":  "#5558e3",
    "accent_light":  "#818cf8",
    "accent_muted":  "#33305a",

    "success":       "#34d399",
    "success_hover": "#2ab882",
    "warning":       "#fbbf24",
    "warning_hover": "#d9a620",
    "error":         "#fb7185",
    "error_hover":   "#e55b6e",
    "info":          "#60a5fa",

    "text_primary":  "#eaeaef",
    "text_secondary":"#9a9ab0",
    "text_muted":    "#6c6c80",
    "text_accent":   "#a5b4fc",

    "border":        "#3a3a46",
    "border_light":  "#48485a",

    "scrollbar":     "#3a3a46",
    "scrollbar_hover":"#505060",

    "tab_bg":        "#2b2b33",
    "tab_selected":  "#6366f1",
    "tab_hover":     "#40404c",
}

_LIGHT = {
    "bg_base":       "#eaedf2",
    "bg_primary":    "#f2f4f7",
    "bg_secondary":  "#ffffff",
    "bg_tertiary":   "#ebeef3",
    "bg_hover":      "#dde1e9",
    "bg_active":     "#ced4de",
    "bg_input":      "#ffffff",

    "accent":        "#6366f1",
    "accent_hover":  "#4f46e5",
    "accent_light":  "#818cf8",
    "accent_muted":  "#e0e0fb",

    "success":       "#10b981",
    "success_hover": "#059669",
    "warning":       "#f59e0b",
    "warning_hover": "#d97706",
    "error":         "#ef4444",
    "error_hover":   "#dc2626",
    "info":          "#3b82f6",

    "text_primary":  "#1e1e2e",
    "text_secondary":"#52526a",
    "text_muted":    "#8c8ca0",
    "text_accent":   "#6366f1",

    "border":        "#d0d4de",
    "border_light":  "#bfc4d0",

    "scrollbar":     "#ccd0da",
    "scrollbar_hover":"#b0b6c4",

    "tab_bg":        "#e2e5ec",
    "tab_selected":  "#6366f1",
    "tab_hover":     "#d4d8e2",
}

# ── 读取用户上次保存的主题偏好 ──

def _read_saved_theme():
    try:
        p = os.path.join(os.path.expanduser('~'),
                         '.pollutant_analysis_config.json')
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f).get('preferences', {}).get('theme', 'Dark')
    except Exception as e:
        logger.warning(f"无法读取主题配置: {e}，使用默认 Dark 主题")
    return 'Dark'

# ── C 是全局可变字典，所有模块都引用它 ──

C = {}

def apply_theme(mode='Dark'):
    """用指定主题覆盖 C 的全部键值，mode 为 'Dark' / 'Light' / 'System'。"""
    palette = _LIGHT if mode == 'Light' else _DARK
    C.clear()
    C.update(palette)

# 启动时立即按保存的偏好设置主题
apply_theme(_read_saved_theme())

# ═══════════════════════════════════════════════════════════════
#  字体规格
# ═══════════════════════════════════════════════════════════════

FONTS = {
    "h1":         lambda: ctk.CTkFont(size=20, weight="bold"),
    "h2":         lambda: ctk.CTkFont(size=16, weight="bold"),
    "h3":         lambda: ctk.CTkFont(size=14, weight="bold"),
    "body":       lambda: ctk.CTkFont(size=13),
    "body_bold":  lambda: ctk.CTkFont(size=13, weight="bold"),
    "small":      lambda: ctk.CTkFont(size=11),
    "tiny":       lambda: ctk.CTkFont(size=10),
    "mono":       lambda: ctk.CTkFont(family="Consolas", size=12),
    "mono_small": lambda: ctk.CTkFont(family="Consolas", size=11),
    "btn":        lambda: ctk.CTkFont(size=13, weight="bold"),
    "btn_small":  lambda: ctk.CTkFont(size=12),
}


# ═══════════════════════════════════════════════════════════════
#  组件工厂函数
# ═══════════════════════════════════════════════════════════════

def make_card(parent, **kw):
    defaults = dict(
        corner_radius=14,
        fg_color=C["bg_secondary"],
        border_width=1,
        border_color=C["border"],
    )
    defaults.update(kw)
    return ctk.CTkFrame(parent, **defaults)


def make_inner_frame(parent, **kw):
    defaults = dict(fg_color="transparent")
    defaults.update(kw)
    return ctk.CTkFrame(parent, **defaults)


def make_section_title(parent, text, icon=""):
    display = f"{icon}  {text}" if icon else text
    lbl = ctk.CTkLabel(
        parent, text=display,
        font=FONTS["h3"](),
        text_color=C["text_primary"],
    )
    return lbl


def make_hint(parent, text):
    return ctk.CTkLabel(
        parent, text=text,
        font=FONTS["small"](),
        text_color=C["text_secondary"],
    )


def make_btn_primary(parent, **kw):
    defaults = dict(
        corner_radius=10, height=36,
        fg_color=C["accent"],
        hover_color=C["accent_hover"],
        text_color="white",
        font=FONTS["btn"](),
    )
    defaults.update(kw)
    return ctk.CTkButton(parent, **defaults)


def make_btn_secondary(parent, **kw):
    defaults = dict(
        corner_radius=10, height=36,
        fg_color=C["bg_tertiary"],
        hover_color=C["bg_hover"],
        text_color=C["text_primary"],
        font=FONTS["btn_small"](),
        border_width=1,
        border_color=C["border"],
    )
    defaults.update(kw)
    return ctk.CTkButton(parent, **defaults)


def make_btn_danger(parent, **kw):
    defaults = dict(
        corner_radius=10, height=36,
        fg_color=C["error"],
        hover_color=C["error_hover"],
        text_color="white",
        font=FONTS["btn_small"](),
    )
    defaults.update(kw)
    return ctk.CTkButton(parent, **defaults)


def make_btn_warning(parent, **kw):
    defaults = dict(
        corner_radius=10, height=36,
        fg_color=C["warning"],
        hover_color=C["warning_hover"],
        text_color=C["bg_base"],
        font=FONTS["btn_small"](),
    )
    defaults.update(kw)
    return ctk.CTkButton(parent, **defaults)


def make_entry(parent, **kw):
    defaults = dict(
        corner_radius=10, height=38,
        fg_color=C["bg_input"],
        border_color=C["border"],
        text_color=C["text_primary"],
        font=FONTS["body"](),
    )
    defaults.update(kw)
    return ctk.CTkEntry(parent, **defaults)


def make_optionmenu(parent, **kw):
    defaults = dict(
        corner_radius=10, height=36,
        fg_color=C["bg_tertiary"],
        button_color=C["bg_tertiary"],
        button_hover_color=C["accent"],
        dropdown_fg_color=C["bg_tertiary"],
        dropdown_hover_color=C["accent"],
        dropdown_text_color=C["text_primary"],
        text_color=C["text_primary"],
        font=FONTS["body"](),
    )
    defaults.update(kw)
    return ctk.CTkOptionMenu(parent, **defaults)


def make_textbox(parent, **kw):
    defaults = dict(
        corner_radius=10,
        fg_color=C["bg_input"],
        text_color=C["text_primary"],
        font=FONTS["mono"](),
    )
    defaults.update(kw)
    return ctk.CTkTextbox(parent, **defaults)


def make_checkbox(parent, **kw):
    defaults = dict(
        fg_color=C["accent"],
        hover_color=C["accent_hover"],
        border_color=C["border_light"],
        corner_radius=6,
        font=FONTS["body"](),
        text_color=C["text_primary"],
    )
    defaults.update(kw)
    return ctk.CTkCheckBox(parent, **defaults)


def make_scrollframe(parent, **kw):
    defaults = dict(
        corner_radius=14,
        fg_color="transparent",
        scrollbar_button_color=C["scrollbar"],
        scrollbar_button_hover_color=C["scrollbar_hover"],
    )
    defaults.update(kw)
    return ctk.CTkScrollableFrame(parent, **defaults)


def make_progress(parent, **kw):
    defaults = dict(
        height=16, corner_radius=10,
        fg_color=C["bg_tertiary"],
        progress_color=C["accent"],
    )
    defaults.update(kw)
    return ctk.CTkProgressBar(parent, **defaults)


def make_tabview(parent, **kw):
    defaults = dict(
        corner_radius=14,
        fg_color=C["bg_base"],
        segmented_button_fg_color=C["tab_bg"],
        segmented_button_selected_color=C["tab_selected"],
        segmented_button_selected_hover_color=C["accent_hover"],
        segmented_button_unselected_color=C["tab_bg"],
        segmented_button_unselected_hover_color=C["tab_hover"],
        text_color=C["text_primary"],
    )
    defaults.update(kw)
    return ctk.CTkTabview(parent, **defaults)


def make_empty_state(parent, icon, title, message, button_text=None, command=None, **kw):
    defaults = dict(
        corner_radius=14,
        fg_color=C["bg_secondary"],
        border_width=1,
        border_color=C["border"],
    )
    defaults.update(kw)
    frame = ctk.CTkFrame(parent, **defaults)

    icon_label = ctk.CTkLabel(
        frame,
        text=icon,
        font=ctk.CTkFont(size=30, weight="bold"),
        text_color=C["accent_light"],
    )
    icon_label.pack(pady=(26, 8))

    title_label = ctk.CTkLabel(
        frame,
        text=title,
        font=FONTS["h2"](),
        text_color=C["text_primary"],
    )
    title_label.pack(pady=(0, 6))

    message_label = ctk.CTkLabel(
        frame,
        text=message,
        font=FONTS["body"](),
        text_color=C["text_secondary"],
        justify="center",
        wraplength=420,
    )
    message_label.pack(padx=24, pady=(0, 18))

    action_button = None
    if button_text and command is not None:
        action_button = make_btn_primary(frame, text=button_text, width=140, command=command)
        action_button.pack(pady=(0, 24))

    frame.icon_label = icon_label
    frame.title_label = title_label
    frame.message_label = message_label
    frame.action_button = action_button
    return frame


# ═══════════════════════════════════════════════════════════════
#  通用弹窗
# ═══════════════════════════════════════════════════════════════

def show_message(app, title, message, msg_type="info"):
    message = str(message)
    lines = message.splitlines() or [message]
    max_line_length = max((len(line) for line in lines), default=0)
    width = min(760, max(440, 360 + max_line_length * 4))
    height = min(520, max(220, 150 + len(lines) * 18))

    dialog = ctk.CTkToplevel(app)
    dialog.title(title)
    dialog.geometry(f"{width}x{height}")
    dialog.resizable(False, False)
    dialog.transient(app)
    dialog.grab_set()

    ct = ctk.CTkFrame(dialog, corner_radius=0, fg_color=C["bg_primary"])
    ct.pack(fill="both", expand=True)

    color = {"info": C["success"], "error": C["error"],
             "warning": C["warning"]}.get(msg_type, C["text_primary"])

    ctk.CTkLabel(ct, text=title, font=FONTS["h2"](),
                 text_color=color).pack(pady=(28, 8))

    body = ctk.CTkTextbox(
        ct,
        height=max(90, height - 150),
        corner_radius=10,
        fg_color=C["bg_input"],
        text_color=C["text_primary"],
        font=FONTS["body"](),
        wrap="word",
        border_width=1,
        border_color=C["border"],
    )
    body.pack(fill="both", expand=True, pady=8, padx=25)
    body.insert("1.0", message)
    body.configure(state="disabled")

    if len(lines) <= 3 and max_line_length <= 80:
        try:
            body.configure(height=82)
        except Exception:
            logger.debug("Failed to shrink message dialog body", exc_info=True)

    make_btn_primary(ct, text="确 定", width=120,
                     command=dialog.destroy).pack(pady=(12, 22))

    dialog.update_idletasks()
    
    # 获取主窗口和对话框尺寸
    app_w = app.winfo_width()
    app_h = app.winfo_height()
    dlg_w = width
    dlg_h = height
    
    # 计算居中坐标，并确保不小于 0 (防止越界到屏幕外)
    x = max(0, app.winfo_x() + (app_w - dlg_w) // 2)
    y = max(0, app.winfo_y() + (app_h - dlg_h) // 2)
    
    dialog.geometry(f"+{x}+{y}")
