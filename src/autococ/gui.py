"""Local Tk desktop interface. Opening it never contacts an emulator."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from queue import Empty
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .config import load_config
from .desktop import (DesktopController, RunOptions, STATUS_LABELS, TASK_LABELS, desktop_config, display_reason,
                      load_options, report_history, report_text, save_options, settings_path)
from .errors import AutoCOCError


BG, CARD, INK, MUTED, ACCENT = "#eef2f6", "#ffffff", "#172b3a", "#607381", "#147d73"
PRESETS = {"日常循环": ("launch", "collect", "request", "donate", "battle"),
           "村庄维护": ("launch", "collect", "request", "donate"),
           "仅对战": ("launch", "battle"), "部落事务": ("launch", "request", "donate")}


class AutoCOCApp:
    def __init__(self, root: tk.Tk, config_path: Path, *, controller: DesktopController | None = None) -> None:
        self.root = root
        self.config_path = config_path.resolve()
        self.controller = controller or DesktopController()
        self.closing = False
        self.active = False
        self.report_path: Path | None = None
        self.history: dict[str, tuple[Path, dict]] = {}
        self.evidence_paths: list[Path] = []
        self.preview_image = None
        self.editors: list[ttk.Widget] = []
        root.title("AutoCOC · 部落助手")
        root.geometry("1180x920")
        root.minsize(1060, 870)
        root.configure(bg=BG)
        root.option_add("*Font", ("Microsoft YaHei UI", 10))
        self._style()
        self._build()
        self._load()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self._poll)

    def _style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=CARD, foreground=INK)
        style.configure("Muted.TLabel", foreground=MUTED, background=CARD)
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("TCheckbutton", background=CARD, foreground=INK, padding=(0, 4))
        style.map("TCheckbutton", background=[("active", CARD)])
        style.configure("TButton", padding=(12, 7), font=("Microsoft YaHei UI", 10))
        style.configure("Accent.TButton", background=ACCENT, foreground="white", borderwidth=0)
        style.map("Accent.TButton", background=[("disabled", "#95aaa8"), ("active", "#0d685f")])
        style.configure("Treeview", rowheight=29, background=CARD, fieldbackground=CARD, foreground=INK)
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 10, "bold"), padding=5)
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(15, 8))
        style.configure("TEntry", padding=5)

    def _build(self) -> None:
        header = tk.Frame(self.root, bg=INK, padx=24, pady=18)
        header.pack(fill="x")
        tk.Label(header, text="AutoCOC", bg=INK, fg="white", font=("Segoe UI", 23, "bold")).pack(side="left")
        tk.Label(header, text="部落助手  /  本地任务控制台", bg=INK, fg="#b5c9d7",
                 font=("Microsoft YaHei UI", 11)).pack(side="left", padx=22)
        self.badge = tk.Label(header, text="离线待命", bg="#284956", fg="#b7ece0", padx=14, pady=5)
        self.badge.pack(side="right")
        self.status = tk.StringVar(value="选择任务后开始；离线预演只检查任务编排。")
        tk.Label(self.root, textvariable=self.status, bg="#dcece9", fg="#18564f", anchor="w",
                 padx=24, pady=11).pack(fill="x")

        body = ttk.Frame(self.root, padding=(18, 14))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body, style="Card.TFrame", padding=18)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        left.columnconfigure(1, weight=1)
        ttk.Label(left, text="任务编排", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.preset = tk.StringVar(value="日常循环")
        self.preset_box = ttk.Combobox(left, textvariable=self.preset, values=(*PRESETS, "自定义"),
                                       state="readonly", width=13)
        self.preset_box.grid(row=0, column=1, sticky="e")
        self.preset_box.bind("<<ComboboxSelected>>", self._preset_changed)
        self.editors.append(self.preset_box)
        self.task_vars = {}
        for row, (task, label) in enumerate(TASK_LABELS.items(), 1):
            var = tk.BooleanVar(value=task == "launch")
            self.task_vars[task] = var
            check = ttk.Checkbutton(left, text=label, variable=var, command=self._custom)
            check.grid(row=row, column=0, columnspan=2, sticky="w")
            if task == "launch":
                check.state(["disabled"])
            else:
                self.editors.append(check)
        ttk.Label(left, text="捐兵识别待真实窗口校准；实战尚未验收。", style="Muted.TLabel",
                  wraplength=300).grid(row=8, column=0, columnspan=2, sticky="w", pady=(4, 12))
        ttk.Separator(left).grid(row=9, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.values = {}
        for row, (key, label, value) in enumerate((
                ("max_runs", "最多循环", "10"), ("minutes", "最长运行 / 分钟", "30"),
                ("resources", "最低金币 + 圣水", "300000"), ("searches", "每场搜索上限", "30"),
                ("serial", "设备地址", "")), 10):
            ttk.Label(left, text=label).grid(row=row, column=0, sticky="w", pady=4)
            var = tk.StringVar(value=value)
            self.values[key] = var
            entry = ttk.Entry(left, textvariable=var, width=19)
            entry.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=4)
            self.editors.append(entry)
        self.mode = tk.StringVar(value="离线预演")
        ttk.Label(left, text="运行模式").grid(row=15, column=0, sticky="w", pady=(10, 4))
        self.mode_box = ttk.Combobox(left, textvariable=self.mode, values=("离线预演", "实机执行"),
                                     state="readonly", width=17)
        self.mode_box.grid(row=15, column=1, sticky="ew", padx=(10, 0), pady=(10, 4))
        self.mode_box.bind("<<ComboboxSelected>>", self._mode_changed)
        self.editors.append(self.mode_box)
        self.mode_note = tk.StringVar(value="不连接模拟器，不操作游戏。")
        ttk.Label(left, textvariable=self.mode_note, style="Muted.TLabel", wraplength=300).grid(
            row=16, column=0, columnspan=2, sticky="w", pady=(3, 12))
        buttons = ttk.Frame(left, style="Card.TFrame")
        buttons.grid(row=17, column=0, columnspan=2, sticky="ew")
        buttons.columnconfigure((0, 1), weight=1)
        self.start_button = ttk.Button(buttons, text="开始预演", style="Accent.TButton", command=self.start)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 7))
        self.stop_button = ttk.Button(buttons, text="停止", command=self.stop, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew")
        self.save_button = ttk.Button(left, text="保存界面方案", command=self._save)
        self.save_button.grid(row=18, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.editors.append(self.save_button)

        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)
        counters = ttk.Frame(right)
        counters.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self.counters = {}
        for col, (key, label) in enumerate((("successes", "真实成功"), ("failures", "失败"),
                                            ("skipped", "跳过"), ("simulated", "预演任务"))):
            counters.columnconfigure(col, weight=1)
            card = ttk.Frame(counters, style="Card.TFrame", padding=(15, 10))
            card.grid(row=0, column=col, sticky="ew", padx=(0, 8 if col < 3 else 0))
            ttk.Label(card, text=label, style="Muted.TLabel").pack(anchor="w")
            var = tk.StringVar(value="—")
            self.counters[key] = var
            ttk.Label(card, textvariable=var, font=("Segoe UI", 24, "bold")).pack(anchor="w")
        task_card = ttk.Frame(right, style="Card.TFrame", padding=12)
        task_card.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        self.run_label = tk.StringVar(value="当前任务 · 尚未运行")
        ttk.Label(task_card, textvariable=self.run_label, style="Title.TLabel").pack(anchor="w", pady=(0, 8))
        self.task_table = ttk.Treeview(task_card, columns=("task", "status", "reason"), show="headings", height=5)
        for col, title, width in (("task", "任务", 120), ("status", "状态", 65), ("reason", "结果说明", 300)):
            self.task_table.heading(col, text=title)
            self.task_table.column(col, width=width, minwidth=50, stretch=col == "reason")
        self.task_table.pack(fill="x")
        self.tabs = ttk.Notebook(right)
        self.tabs.grid(row=2, column=0, sticky="nsew")
        logs = ttk.Frame(self.tabs, style="Card.TFrame")
        records = ttk.Frame(self.tabs, style="Card.TFrame", padding=8)
        evidence = ttk.Frame(self.tabs, style="Card.TFrame", padding=10)
        self.tabs.add(logs, text="运行日志")
        self.tabs.add(records, text="历史报告")
        self.tabs.add(evidence, text="截图证据")
        self.log = ScrolledText(logs, wrap="word", state="disabled", height=8, bg="#182a37", fg="#d8e4ec",
                                relief="flat", font=("Microsoft YaHei UI", 9), padx=12, pady=12)
        self.log.pack(fill="both", expand=True)
        toolbar = ttk.Frame(records, style="Card.TFrame")
        toolbar.pack(fill="x", pady=(0, 6))
        ttk.Button(toolbar, text="刷新记录", command=self.refresh_history).pack(side="left")
        ttk.Button(toolbar, text="打开选中报告", command=self._open_report).pack(side="left", padx=6)
        history_frame = ttk.Frame(records, style="Card.TFrame")
        history_frame.pack(fill="x")
        self.history_table = ttk.Treeview(history_frame, columns=("time", "mode", "result"), show="headings", height=4)
        for col, title, width in (("time", "开始时间", 190), ("mode", "模式", 85), ("result", "成功 / 失败 / 跳过", 160)):
            self.history_table.heading(col, text=title)
            self.history_table.column(col, width=width, stretch=True)
        self.history_table.pack(side="left", fill="x", expand=True)
        history_scroll = ttk.Scrollbar(history_frame, orient="vertical", command=self.history_table.yview)
        history_scroll.pack(side="right", fill="y")
        self.history_table.configure(yscrollcommand=history_scroll.set)
        self.history_table.bind("<<TreeviewSelect>>", self._select_report)
        self.details = ScrolledText(records, wrap="word", state="disabled", height=5, relief="flat",
                                    font=("Microsoft YaHei UI", 9), padx=6, pady=8)
        self.details.pack(fill="both", expand=True, pady=(6, 0))
        self.evidence_choice = ttk.Combobox(evidence, state="readonly")
        self.evidence_choice.pack(fill="x")
        self.evidence_choice.bind("<<ComboboxSelected>>", self._show_evidence)
        self.preview = tk.Label(evidence, bg="#e7edf2", fg=MUTED,
                                text="选择一份历史报告，查看它保存的截图。")
        self.preview.pack(fill="both", expand=True, pady=8)
        ttk.Button(evidence, text="打开原始截图", command=self._open_evidence).pack(anchor="e")
        footer = tk.Frame(self.root, bg=BG, padx=18, pady=8)
        footer.pack(side="bottom", fill="x", before=body)
        self.config_label = tk.StringVar()
        tk.Label(footer, textvariable=self.config_label, bg=BG, fg=MUTED, anchor="w").pack(side="left")
        self.config_button = ttk.Button(footer, text="选择配置", command=self._choose_config)
        self.config_button.pack(side="right")
        self.editors.append(self.config_button)

    def _load(self) -> None:
        self.config_label.set(str(self.config_path))
        self.report_path = None
        self.history.clear()
        self.history_table.delete(*self.history_table.get_children())
        try:
            config = load_config(self.config_path)
            try:
                options = load_options(settings_path(self.config_path), config)
            except (AutoCOCError, OSError) as exc:
                self._log(f"界面方案读取失败，使用基础配置：{exc}")
                options = RunOptions.from_config(config)
            self._apply(options)
            resolved = desktop_config(self.config_path, options)
            self.report_dir = resolved.runtime.report_dir
            transport = f"MuMu 原生通道 · 实例 {config.mumu.instance_index}" if config.mumu else "ADB 通道"
            self._log(f"已加载配置；{transport}。当前为离线预演。")
            self.start_button.state(["!disabled"])
            self.refresh_history()
        except (AutoCOCError, OSError, ValueError) as exc:
            self.status.set(f"配置加载失败：{exc}")
            self._log(str(exc))
            self.start_button.state(["disabled"])
            if hasattr(self, "report_dir"):
                del self.report_dir

    def _apply(self, options: RunOptions) -> None:
        for task, var in self.task_vars.items():
            var.set(task in options.tasks)
        self.task_order = options.tasks
        self.preset.set(next((name for name, tasks in PRESETS.items() if tasks == options.tasks), "自定义"))
        for key, value in (("max_runs", options.max_runs), ("minutes", f"{options.max_duration_sec / 60:g}"),
                           ("resources", options.min_expected_resources), ("searches", options.max_searches),
                           ("serial", options.serial)):
            self.values[key].set(str(value))
        self.mode.set("离线预演")
        self._mode_changed()

    def _options(self) -> RunOptions:
        minutes = float(self.values["minutes"].get())
        if not math.isfinite(minutes) or minutes <= 0 or minutes * 60 != int(minutes * 60):
            raise ValueError("运行时长必须为正数，精确到整秒")
        tasks = tuple(task for task in self.task_order if self.task_vars[task].get())
        tasks += tuple(task for task in TASK_LABELS if self.task_vars[task].get() and task not in tasks)
        options = RunOptions(tasks, int(self.values["max_runs"].get()), int(minutes * 60),
                             int(self.values["resources"].get()), int(self.values["searches"].get()),
                             self.values["serial"].get(), self.mode.get() == "离线预演")
        options.validate()
        return options

    def _preset_changed(self, event=None) -> None:
        tasks = PRESETS.get(self.preset.get())
        if tasks is not None:
            self.task_order = tasks
            for task, var in self.task_vars.items():
                var.set(task in tasks)

    def _custom(self) -> None:
        self.preset.set("自定义")
        self.task_order = tuple(TASK_LABELS)

    def _mode_changed(self, event=None) -> None:
        offline = self.mode.get() == "离线预演"
        self.start_button.configure(text="开始预演" if offline else "开始实机执行")
        self.mode_note.set("不连接模拟器，不操作游戏。" if offline else "连接指定设备，执行勾选的游戏任务。")
        self.badge.configure(text="离线待命" if offline else "实机待命")

    def _save(self) -> None:
        try:
            save_options(settings_path(self.config_path), self._options())
            self.status.set("界面方案已保存。下次打开仍默认离线预演。")
        except (AutoCOCError, OSError, ValueError) as exc:
            messagebox.showerror("无法保存方案", str(exc), parent=self.root)

    def start(self) -> None:
        try:
            options = self._options()
            self.controller.start(self.config_path, options)
        except (AutoCOCError, OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("无法开始任务", str(exc), parent=self.root)
            return
        self.active = True
        self._running_controls(True)
        self.task_table.delete(*self.task_table.get_children())
        for task in options.tasks:
            self.task_table.insert("", "end", iid=task, values=(TASK_LABELS[task], "等待", ""))
        for value in self.counters.values():
            value.set("0")
        self.run_label.set("当前任务 · 准备运行")
        self.status.set("离线预演中：不连接模拟器。" if options.dry_run else "正在连接指定设备…")
        self.badge.configure(text="离线预演中" if options.dry_run else "实机运行中")
        self.tabs.select(0)

    def stop(self) -> None:
        self.controller.stop()
        self.stop_button.state(["disabled"])
        self.status.set("停止请求已发送，等待当前有时限的调用返回并保存报告。")
        self.badge.configure(text="正在停止")

    def _running_controls(self, running: bool) -> None:
        for widget in self.editors:
            widget.state(["disabled"] if running else ["!disabled"])
        self.start_button.state(["disabled"] if running else ["!disabled"])
        self.stop_button.state(["!disabled"] if running else ["disabled"])

    def _poll(self) -> None:
        for _ in range(200):
            try:
                event = self.controller.events.get_nowait()
            except Empty:
                break
            self._event(event)
        if self.active and not self.controller.running:
            self.active = False
            self._running_controls(False)
        if self.closing and not self.controller.running:
            self.root.destroy()
            return
        self.root.after(100, self._poll)

    def _event(self, event: dict) -> None:
        kind = event["kind"]
        if kind in {"log", "error"}:
            self._log(event["text"])
        elif kind == "run_started":
            self.run_label.set("当前运行 · " + event["run_id"])
        elif kind in {"task_started", "task_result"}:
            task = event["task"]
            state = "running" if kind == "task_started" else event["status"]
            if self.task_table.exists(task):
                self.task_table.item(task, values=(TASK_LABELS[task], STATUS_LABELS[state], display_reason(event.get("reason", ""))))
            self._log(f"{TASK_LABELS[task]} · {STATUS_LABELS[state]}  {display_reason(event.get('reason', ''))}")
        elif kind == "finished":
            payload = event["summary"]
            self._show_counts(payload)
            final_tasks = {result["task"]: result for result in payload.get("task_results", [])}
            for task in self.task_table.get_children():
                result = final_tasks.get(task)
                state = STATUS_LABELS[result["status"]] if result else "未执行"
                reason = display_reason(result["reason"]) if result else "运行已结束"
                self.task_table.item(task, values=(TASK_LABELS[task], state, reason))
            self.report_path = event["report"]
            offline = payload["mode"] == "dry-run"
            stopped = "interrupted" in payload.get("stop_reason", "")
            self.badge.configure(text="已停止" if stopped else "运行失败" if payload["failures"] else "预演完成" if offline else "运行结束")
            self.status.set("已保存报告。" + ("预演结果不计入真实成功或收益。" if offline else payload["stop_reason"])
                            if self.report_path else "运行已结束，但报告未保存；请查看日志。")
            self._log("停止原因：" + display_reason(payload["stop_reason"]))
            self.refresh_history()

    def _log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 600:
            self.log.delete("1.0", "101.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def refresh_history(self) -> None:
        if not hasattr(self, "report_dir"):
            return
        try:
            rows, errors = report_history(self.report_dir)
        except OSError as exc:
            self._log(f"无法读取历史报告：{exc}")
            return
        self.history_table.delete(*self.history_table.get_children())
        self.history = {str(path): (path, payload) for path, payload in rows}
        for path, payload in rows:
            self.history_table.insert("", "end", iid=str(path), values=(
                str(payload.get("started_at", "未知"))[:19].replace("T", " "),
                "实机" if payload["mode"] == "live" else "离线预演",
                f"{payload.get('successes', 0)} / {payload.get('failures', 0)} / {payload.get('skipped', 0)}"))
        for error in errors:
            self._log("跳过损坏报告：" + error)
        if self.report_path is not None and str(self.report_path) in self.history:
            self.history_table.selection_set(str(self.report_path))
            self.history_table.see(str(self.report_path))

    def _show_counts(self, payload: dict) -> None:
        for key, var in self.counters.items():
            var.set(str(payload.get(key, 0)) if key != "successes" or payload.get("mode") == "live" else "0")

    def _select_report(self, event=None) -> None:
        selection = self.history_table.selection()
        if not selection or selection[0] not in self.history:
            return
        path, payload = self.history[selection[0]]
        self.report_path = path
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", report_text(payload))
        self.details.configure(state="disabled")
        self.evidence_paths = []
        for result in payload.get("task_results", []):
            for name in result.get("evidence", []):
                image = Path(name)
                if not image.is_absolute():
                    image = self.config_path.parent / image
                if image not in self.evidence_paths:
                    self.evidence_paths.append(image)
        self.evidence_choice.configure(values=[f"{i + 1}. {p.name}" for i, p in enumerate(self.evidence_paths)])
        self.preview_image = None
        self.preview.configure(image="", text="该报告没有保存截图。")
        self.evidence_choice.set("")
        if self.evidence_paths:
            self.evidence_choice.current(len(self.evidence_paths) - 1)
            self._show_evidence()

    def _show_evidence(self, event=None) -> None:
        index = self.evidence_choice.current()
        if not 0 <= index < len(self.evidence_paths):
            return
        try:
            image = tk.PhotoImage(file=str(self.evidence_paths[index]))
            factor = max(1, math.ceil(image.width() / 600), math.ceil(image.height() / 230))
            self.preview_image = image.subsample(factor)
            self.preview.configure(image=self.preview_image, text="")
        except (tk.TclError, OSError) as exc:
            self.preview_image = None
            self.preview.configure(image="", text=f"无法预览截图：{exc}")

    def _open_path(self, path: Path) -> None:
        try:
            os.startfile(str(path.resolve()))
        except OSError as exc:
            messagebox.showerror("无法打开文件", str(exc), parent=self.root)

    def _open_report(self) -> None:
        if self.report_path is not None:
            markdown = self.report_path.with_suffix(".md")
            self._open_path(markdown if markdown.is_file() else self.report_path)

    def _open_evidence(self) -> None:
        index = self.evidence_choice.current()
        if 0 <= index < len(self.evidence_paths):
            self._open_path(self.evidence_paths[index])

    def _choose_config(self) -> None:
        name = filedialog.askopenfilename(parent=self.root, title="选择 AutoCOC 配置",
                                          initialdir=self.config_path.parent, filetypes=[("TOML 配置", "*.toml")])
        if name:
            self.config_path = Path(name).resolve()
            self._load()

    def close(self) -> None:
        if self.controller.running:
            self.closing = True
            self.stop()
        else:
            self.root.destroy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autococ-gui")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    args = parser.parse_args(argv)
    root = tk.Tk()
    AutoCOCApp(root, args.config)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
