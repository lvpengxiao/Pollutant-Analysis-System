import tkinter as tk
from tkinter import ttk
import threading
def run_splash():
    # 极简的原生 Tkinter 闪屏，无任何重度第三方依赖
    root = tk.Tk()
    # 隐藏边框和标题栏
    root.overrideredirect(True)
    
    # 闪屏尺寸
    width = 400
    height = 250
    
    # 屏幕居中
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # 浅蓝色背景，现代感设计
    root.configure(bg='#2B3A55')
    
    # 标题
    title_label = tk.Label(
        root, 
        text="🔬 Pollutant Analysis System", 
        font=("Microsoft YaHei", 18, "bold"), 
        bg='#2B3A55', 
        fg='white'
    )
    title_label.pack(pady=(60, 20))
    
    # 状态文本
    status_label = tk.Label(
        root, 
        text="正在初始化核心引擎 (Pandas/Sklearn)...", 
        font=("Microsoft YaHei", 10), 
        bg='#2B3A55', 
        fg='#A0B2C6'
    )
    status_label.pack(pady=(0, 15))
    
    # 进度条
    progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="indeterminate")
    progress.pack()
    progress.start(15)
    
    state = {"main_module": None, "error": None}

    def close_splash():
        try:
            progress.stop()
        except Exception:
            pass
        root.destroy()

    # 在后台线程中加载沉重的主程序
    def load_main_app():
        try:
            # 导入真正的业务主入口，这里会触发那些耗时的 imports。
            # Use a normal import so PyInstaller can statically discover app.py.
            import app as app_module
            state["main_module"] = app_module

            def notify_ready():
                status_label.config(text="核心模块加载完成，正在启动主界面...")
                root.after(250, close_splash)

            root.after(0, notify_ready)
        except Exception as e:
            state["error"] = e
            error_message = str(e)
            # 如果加载失败，在闪屏上显示错误
            def show_error():
                progress.stop()
                progress.pack_forget()
                status_label.config(text=f"启动失败:\n{error_message}", fg='#FF6B6B')
                
                # 加一个退出按钮
                btn = tk.Button(root, text="退出", command=root.destroy, bg='#FF6B6B', fg='white')
                btn.pack(pady=10)
            root.after(0, show_error)

    # 启动后台加载线程
    threading.Thread(target=load_main_app, daemon=True).start()
    
    root.mainloop()

    if state["error"] is not None:
        return

    if state["main_module"] is not None:
        app = state["main_module"].main()
        app.mainloop()

if __name__ == "__main__":
    # 防止多进程在 Windows 下被 spawn 时重复执行闪屏逻辑
    import multiprocessing
    multiprocessing.freeze_support()
    
    # 只有当前进程是主进程时，才执行闪屏和主界面加载
    if multiprocessing.current_process().name == 'MainProcess':
        run_splash()
