"""Local Tk desktop interface. Opening it never contacts an emulator."""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
import os
from pathlib import Path
from queue import Empty
import tomllib
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .config import VisionAgentConfig, load_config
from .desktop import (DesktopController, RunOptions, STATUS_LABELS, STRATEGY_LABELS, TASK_LABELS, desktop_config, display_reason,
                      load_options, report_history, report_text, save_options, settings_path)
from .errors import AutoCOCError
from .routine_config import (BATTLE_KINDS, TASK_KINDS, GoalConfig, ResourceFilter,
                             RoutineConfig, TaskSpec, default_routine)


BG, CARD, INK, MUTED, ACCENT = "#eef2f6", "#ffffff", "#172b3a", "#607381", "#147d73"
RESOURCE_LABELS = {"gold": "金币", "elixir": "圣水", "dark_elixir": "黑油", "gems": "宝石"}
class AutoCOCApp:
    def __init__(self, root: tk.Tk, config_path: Path, *, controller: DesktopController | None = None) -> None:
        self.root = root
        self.config_path = config_path.resolve()
        self.controller = controller or DesktopController()
        self.closing = False
        self.active = False
        self.progress_state = {}
        self.report_path: Path | None = None
        self.history: dict[str, tuple[Path, dict]] = {}
        self.evidence_paths: list[Path] = []
        self.preview_image = None
        self.editors: list[ttk.Widget] = []
        root.title("AutoCOC · 部落助手")
        root.geometry("1100x820")
        root.minsize(940, 650)
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
        """Two-page task console; configuration is deliberately off the run page."""
        header = tk.Frame(self.root, bg=INK, padx=24, pady=14)
        header.pack(fill="x")
        tk.Label(header, text="AutoCOC", bg=INK, fg="white", font=("Segoe UI", 22, "bold")).pack(side="left")
        self.badge = tk.Label(header, text="离线待命", bg="#284956", fg="#b7ece0", padx=14, pady=5)
        self.badge.pack(side="right")
        self.status = tk.StringVar(value="选择任务后开始；默认离线预演。")
        tk.Label(self.root, textvariable=self.status, bg="#dcece9", fg="#18564f", anchor="w",
                 padx=24, pady=9).pack(fill="x")
        self.pages = ttk.Notebook(self.root)
        self.pages.pack(fill="both", expand=True, padx=16, pady=12)
        run_page = ttk.Frame(self.pages)
        settings_page = ttk.Frame(self.pages)
        self.pages.add(run_page, text="运行")
        self.pages.add(settings_page, text="配置")
        run_page.columnconfigure(1, weight=1)
        run_page.rowconfigure(0, weight=1)
        left = ttk.Frame(run_page, style="Card.TFrame", padding=16)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        ttk.Label(left, text="本次任务", style="Title.TLabel").pack(anchor="w")
        buttons = ttk.Frame(left, style="Card.TFrame")
        buttons.pack(side="bottom", fill="x", pady=(8, 0))
        self.start_button = ttk.Button(buttons, text="开始预演", style="Accent.TButton", command=self.start)
        self.start_button.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.stop_button = ttk.Button(buttons, text="停止", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", fill="x", expand=True)
        self.task_order = list(TASK_KINDS)
        self.task_kinds = {kind: kind for kind in TASK_KINDS}
        self.task_vars = {kind: tk.BooleanVar(value=False) for kind in TASK_KINDS}
        self.task_rows = ttk.Frame(left, style="Card.TFrame")
        self.task_rows.pack(fill="x", pady=(12, 5))
        move_buttons = ttk.Frame(left, style="Card.TFrame")
        move_buttons.pack(fill="x", pady=(0, 8))
        self.up_button = ttk.Button(move_buttons, text="上移", command=lambda: self._move_task(-1))
        self.up_button.pack(side="left", padx=(0, 5))
        self.down_button = ttk.Button(move_buttons, text="下移", command=lambda: self._move_task(1))
        self.down_button.pack(side="left")
        self.selected_task = tk.StringVar(value="resources")
        self._render_tasks()
        ttk.Label(left, text="捐兵窗口识别尚待实机校准；无法核验时会报告未支持。",
                  style="Muted.TLabel", wraplength=265).pack(anchor="w", pady=(6, 12))

        right = ttk.Frame(run_page)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(3, weight=1)
        self.run_label = tk.StringVar(value="当前任务 · 尚未运行")
        ttk.Label(right, textvariable=self.run_label, style="Title.TLabel").grid(row=0, column=0, sticky="w", pady=(2, 8))
        self.progress_label = tk.StringVar(value="战斗尚未开始")
        ttk.Label(right, textvariable=self.progress_label, wraplength=650, justify="left").grid(
            row=1, column=0, sticky="w", pady=(0, 8))
        counters = ttk.Frame(right)
        counters.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.counters = {}
        for col, (key, label) in enumerate((("battles_completed", "完成战斗"), ("battles_won", "游戏胜利"),
                                            ("goals_completed", "目标达成"), ("simulated", "预演任务"))):
            counters.columnconfigure(col, weight=1)
            card = ttk.Frame(counters, style="Card.TFrame", padding=(10, 8))
            card.grid(row=0, column=col, sticky="ew", padx=(0, 6))
            ttk.Label(card, text=label, style="Muted.TLabel").pack(anchor="w")
            var = tk.StringVar(value="—")
            self.counters[key] = var
            ttk.Label(card, textvariable=var, font=("Segoe UI", 20, "bold")).pack(anchor="w")
        self.counters["successes"] = tk.StringVar(value="0")
        self.counters["failures"] = tk.StringVar(value="0")
        self.counters["skipped"] = tk.StringVar(value="0")
        task_card = ttk.Frame(right, style="Card.TFrame", padding=8)
        task_card.grid(row=3, column=0, sticky="nsew")
        self.task_table = ttk.Treeview(task_card, columns=("task", "status", "reason"), show="headings", height=3)
        for col, title, width in (("task", "任务", 110), ("status", "状态", 75), ("reason", "结果说明", 330)):
            self.task_table.heading(col, text=title)
            self.task_table.column(col, width=width, minwidth=50, stretch=col == "reason")
        self.task_table.pack(fill="x")
        ttk.Label(task_card, text="最近记录", style="Muted.TLabel").pack(anchor="w", pady=(8, 2))
        self.short_log = tk.Listbox(task_card, height=1, relief="flat", fg=INK, bg="#f4f7fa")
        self.short_log.pack(fill="x")
        self.tabs = ttk.Notebook(task_card)
        self.tabs.pack(fill="both", expand=True, pady=(8, 0))
        logs = ttk.Frame(self.tabs, style="Card.TFrame")
        records = ttk.Frame(self.tabs, style="Card.TFrame", padding=8)
        evidence = ttk.Frame(self.tabs, style="Card.TFrame", padding=8)
        self.tabs.add(logs, text="详细日志")
        self.tabs.add(records, text="历史报告")
        self.tabs.add(evidence, text="截图证据")
        self.log = ScrolledText(logs, wrap="word", state="disabled", height=3, bg="#182a37", fg="#d8e4ec",
                                relief="flat", font=("Microsoft YaHei UI", 9), padx=10, pady=8)
        self.log.pack(fill="both", expand=True)
        bar = ttk.Frame(records, style="Card.TFrame")
        bar.pack(fill="x")
        ttk.Button(bar, text="刷新记录", command=self.refresh_history).pack(side="left")
        ttk.Button(bar, text="打开选中报告", command=self._open_report).pack(side="left", padx=6)
        self.history_table = ttk.Treeview(records, columns=("time", "mode", "result"), show="headings", height=4)
        for col, title, width in (("time", "开始时间", 190), ("mode", "模式", 90), ("result", "成功 / 失败 / 跳过", 170)):
            self.history_table.heading(col, text=title)
            self.history_table.column(col, width=width, stretch=True)
        self.history_table.pack(fill="x", pady=(6, 0))
        self.history_table.bind("<<TreeviewSelect>>", self._select_report)
        self.details = ScrolledText(records, wrap="word", state="disabled", height=5, relief="flat",
                                    font=("Microsoft YaHei UI", 9), padx=6, pady=8)
        self.details.pack(fill="both", expand=True, pady=(6, 0))
        self.evidence_choice = ttk.Combobox(evidence, state="readonly")
        self.evidence_choice.pack(fill="x")
        self.evidence_choice.bind("<<ComboboxSelected>>", self._show_evidence)
        self.preview = tk.Label(evidence, bg="#e7edf2", fg=MUTED,
                                text="选择历史报告，查看保存的截图。")
        self.preview.pack(fill="both", expand=True, pady=8)
        ttk.Button(evidence, text="打开原始截图", command=self._open_evidence).pack(anchor="e")

        settings_page.columnconfigure(0, weight=1)
        settings_page.rowconfigure(0, weight=1)
        settings_canvas = tk.Canvas(settings_page, bg=BG, borderwidth=0, highlightthickness=0)
        settings_canvas.grid(row=0, column=0, sticky="nsew")
        settings_scroll = ttk.Scrollbar(settings_page, orient="vertical", command=settings_canvas.yview)
        settings_scroll.grid(row=0, column=1, sticky="ns")
        settings_canvas.configure(yscrollcommand=settings_scroll.set)
        config_content = ttk.Frame(settings_canvas)
        window = settings_canvas.create_window((0, 0), window=config_content, anchor="nw")
        config_content.bind("<Configure>", lambda _event: settings_canvas.configure(scrollregion=settings_canvas.bbox("all")))
        settings_canvas.bind("<Configure>", lambda event: settings_canvas.itemconfigure(window, width=event.width))
        config_content.columnconfigure(0, weight=1)
        common = ttk.Frame(config_content, style="Card.TFrame", padding=12)
        common.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.mode = tk.StringVar(value="离线预演")
        self.mode_note = tk.StringVar(value="不连接模拟器，不操作游戏。")
        self.values = {key: tk.StringVar(value=value) for key, value in
                       (("max_runs", "10"), ("minutes", "30"), ("resources", "300000"),
                        ("searches", "30"), ("serial", ""), ("maintenance", "0"))}
        ttk.Label(common, text="运行模式").grid(row=0, column=0, sticky="w")
        self.mode_box = ttk.Combobox(common, textvariable=self.mode, values=("离线预演", "实机执行"), state="readonly", width=12)
        self.mode_box.grid(row=0, column=1, padx=5)
        self.mode_box.bind("<<ComboboxSelected>>", self._mode_changed)
        ttk.Label(common, textvariable=self.mode_note, style="Muted.TLabel").grid(row=0, column=2, sticky="w", padx=10)
        ttk.Label(common, text="设备地址").grid(row=1, column=0, sticky="w", pady=8)
        ttk.Entry(common, textvariable=self.values["serial"], width=24).grid(row=1, column=1, sticky="w")
        ttk.Label(common, text="回村维护间隔 / 分钟（0 关闭）").grid(row=1, column=2, sticky="w", padx=10)
        ttk.Entry(common, textvariable=self.values["maintenance"], width=8).grid(row=1, column=3, sticky="w")
        task_config = ttk.Frame(config_content, style="Card.TFrame", padding=12)
        task_config.grid(row=1, column=0, sticky="nsew")
        ttk.Label(task_config, text="任务配置", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.selected_task_label = tk.StringVar()
        self.task_choice = ttk.Combobox(task_config, textvariable=self.selected_task_label, state="readonly", width=18)
        self.task_choice.grid(row=0, column=1, sticky="w")
        self.task_choice.bind("<<ComboboxSelected>>", self._choose_task_label)
        self.selected_task.trace_add("write", lambda *_: self._task_selection_changed())
        self.task_config = ttk.Frame(task_config, style="Card.TFrame")
        self.task_config.grid(row=1, column=0, columnspan=4, sticky="nsew", pady=8)
        self.task_fields = {}
        self.full_flags = {}
        self._strategy_contents = {}
        self._editor_task = None
        self.strategy = tk.StringVar(value=STRATEGY_LABELS["two_edge"])
        self._show_task_config()
        self.strategy_text = ScrolledText(task_config, wrap="none", height=9, font=("Consolas", 9))
        self.strategy_text.grid(row=2, column=0, columnspan=4, sticky="nsew")
        self._switch_strategy_editor()
        self.load_strategy_button = ttk.Button(task_config, text="加载打法文件", command=self._load_strategy_file)
        self.load_strategy_button.grid(row=3, column=0, pady=6, sticky="w")
        self.save_strategy_button = ttk.Button(task_config, text="校验并保存打法", command=self._save_strategy_file)
        self.save_strategy_button.grid(row=3, column=1, pady=6, sticky="w")
        self.strategy_preset = tk.StringVar()
        self.strategy_presets = ttk.Combobox(task_config, textvariable=self.strategy_preset, state="readonly", width=22)
        self.strategy_presets.grid(row=3, column=2, pady=6, sticky="w")
        self.strategy_presets.bind("<<ComboboxSelected>>", self._choose_strategy_preset)
        self.save_button = ttk.Button(task_config, text="保存配置", command=self._save)
        self.save_button.grid(row=3, column=3, sticky="e", pady=6)
        task_config.rowconfigure(2, weight=1)
        agent_card = ttk.LabelFrame(config_content, text="战前视觉规划", padding=12)
        agent_card.grid(row=2, column=0, sticky="ew", pady=8)
        self.agent_enabled = tk.BooleanVar(value=False)
        self.agent_jev = tk.BooleanVar(value=False)
        ttk.Checkbutton(agent_card, text="启用战前分析与连续投放", variable=self.agent_enabled).grid(
            row=0, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(agent_card, text="Jev 战前选择（需环境变量凭据）", variable=self.agent_jev).grid(
            row=0, column=2, columnspan=2, sticky="w")
        self.agent_values = {}
        for row, (key, label, default) in enumerate((
                ("model_dir", "本地模型目录", ""), ("layout_profile", "已验收布局文件", ""),
                ("preparation_reserve_sec", "准备期预留 / 秒", "5"),
                ("evidence_limit_mb", "本次证据额度 / MB", "256"),
                ("guide_file", "打法条目 JSON", ""), ("jev_model", "Jev 固定版本", "jev-1.13.0"),
                ("jev_timeout_sec", "Jev 最长等待 / 秒", "2")), start=1):
            variable = tk.StringVar(value=default)
            self.agent_values[key] = variable
            ttk.Label(agent_card, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(agent_card, textvariable=variable, width=52).grid(row=row, column=1, columnspan=3, sticky="ew")
        self.agent_values["mode"] = tk.StringVar(value="continuous")
        ttk.Label(agent_card, text="执行模式").grid(row=8, column=0, sticky="w")
        ttk.Combobox(agent_card, textvariable=self.agent_values["mode"], values=("continuous", "enhanced"),
                     state="readonly", width=20).grid(row=8, column=1, sticky="w")
        ttk.Label(agent_card, text="continuous：战前规划后连续投放。enhanced：接口保留，尚未开放。\n"
                  "模型或布局未验收时会拒绝投放；旧打法需关闭本开关。", style="Muted.TLabel").grid(
                      row=9, column=0, columnspan=4, sticky="w", pady=6)
        self.editors = [self.mode_box, self.save_button, self.up_button, self.down_button]
        footer = tk.Frame(self.root, bg=BG, padx=18, pady=5)
        footer.pack(fill="x")
        self.config_label = tk.StringVar()
        tk.Label(footer, textvariable=self.config_label, bg=BG, fg=MUTED).pack(side="left")
        self.config_button = ttk.Button(footer, text="选择配置", command=self._choose_config)
        self.config_button.pack(side="right")
        self.editors.append(self.config_button)

    def _render_tasks(self) -> None:
        self._task_labels = {task_id: TASK_LABELS[self.task_kinds[task_id]] +
                             (f"（{task_id}）" if task_id != self.task_kinds[task_id] else "")
                             for task_id in self.task_order}
        for widget in self.task_rows.winfo_children():
            widget.destroy()
        self.task_editors = []
        for row, kind in enumerate(self.task_order):
            label = self._task_labels[kind]
            check = ttk.Checkbutton(self.task_rows, text=label, variable=self.task_vars[kind])
            check.grid(row=row, column=0, sticky="w")
            setting = ttk.Button(self.task_rows, text="设置", width=5,
                                 command=lambda name=kind: self._open_task_config(name))
            setting.grid(row=row, column=1, padx=5)
            self.task_editors.extend((check, setting))
        if hasattr(self, "task_choice"):
            self.task_choice.configure(values=tuple(self._task_labels[task_id] for task_id in self.task_order))
            self.selected_task_label.set(self._task_labels.get(self.selected_task.get(), ""))

    def _task_label(self, task_id: str) -> str:
        return self._task_labels.get(task_id, TASK_LABELS.get(task_id, task_id))

    def _move_task(self, step: int) -> None:
        kind = self.selected_task.get()
        index = self.task_order.index(kind)
        other = index + step
        if 0 <= other < len(self.task_order):
            self.task_order[index], self.task_order[other] = self.task_order[other], self.task_order[index]
            self._render_tasks()

    def _open_task_config(self, kind: str) -> None:
        self.selected_task.set(kind)
        self.pages.select(1)

    def _task_selection_changed(self) -> None:
        self.selected_task_label.set(self._task_labels.get(self.selected_task.get(), ""))
        self._show_task_config()
        if hasattr(self, "strategy_text"):
            self._switch_strategy_editor()

    def _choose_task_label(self, event=None) -> None:
        label = self.selected_task_label.get()
        task_id = next((task_id for task_id, value in self._task_labels.items() if value == label), None)
        if task_id is not None:
            self.selected_task.set(task_id)

    def _switch_strategy_editor(self) -> None:
        if self._editor_task is not None:
            self._strategy_contents[self._editor_task] = self.strategy_text.get("1.0", "end-1c")
        task_id = self.selected_task.get()
        if task_id not in self._strategy_contents:
            path_text = self.task_fields.get(task_id, {}).get("strategy_file")
            path = Path(path_text.get()) if path_text and path_text.get().strip() else None
            if path is not None and not path.is_absolute():
                path = self.config_path.parent / path
            try:
                content = path.read_text(encoding="utf-8") if path is not None and path.is_file() else "# 选择打法 TOML 文件后可在这里编辑。\n"
            except OSError:
                content = "# 打法文件读取失败，请重新选择。\n"
            self._strategy_contents[task_id] = content
        self.strategy_text.delete("1.0", "end")
        self.strategy_text.insert("1.0", self._strategy_contents[task_id])
        self._editor_task = task_id

    def _show_task_config(self) -> None:
        if not hasattr(self, "task_config"):
            return
        for widget in self.task_config.winfo_children():
            widget.destroy()
        task_id = self.selected_task.get()
        if not task_id or task_id not in self.task_kinds:
            return
        kind = self.task_kinds[task_id]
        values = self.task_fields.setdefault(task_id, {})
        defaults = {"strategy": "two_edge", "strategy_file": "", "max_battles": "10", "minutes": "30",
                    "searches": "30", "filter": "1", "gold": "", "elixir": "", "dark_elixir": "", "total": "300000",
                    "target_gold": "", "target_elixir": "", "target_dark_elixir": "", "full_resources": "",
                    "adapter_path": "", "target": "", "building_type": ""}
        for key, default in defaults.items():
            values.setdefault(key, tk.StringVar(value=default))
        if kind not in BATTLE_KINDS:
            ttk.Label(self.task_config, text="此任务按本次清单执行一次。", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
            if hasattr(self, "strategy_text"):
                for widget in (self.strategy_text, self.load_strategy_button, self.save_strategy_button, self.strategy_presets):
                    widget.grid_remove()
            return
        if hasattr(self, "strategy_text"):
            for widget in (self.strategy_text, self.load_strategy_button, self.save_strategy_button, self.strategy_presets):
                widget.grid()

        def entry(row: int, col: int, key: str, label: str) -> None:
            ttk.Label(self.task_config, text=label).grid(row=row, column=col, sticky="w", padx=(0, 7), pady=3)
            ttk.Entry(self.task_config, textvariable=values[key], width=22).grid(
                row=row, column=col + 1, sticky="ew", padx=(0, 16), pady=3)

        ttk.Label(self.task_config, text="内置打法").grid(row=0, column=0, sticky="w")
        strategy_label = values.setdefault("strategy_label", tk.StringVar())
        strategy_label.set(STRATEGY_LABELS.get(values["strategy"].get(), values["strategy"].get()))
        strategy_combo = ttk.Combobox(self.task_config, textvariable=strategy_label,
                                      values=tuple(STRATEGY_LABELS.values()), state="readonly", width=22)
        strategy_combo.grid(row=0, column=1, sticky="ew", padx=(0, 16), pady=3)
        strategy_combo.bind("<<ComboboxSelected>>", lambda _event: values["strategy"].set(
            next(key for key, label in STRATEGY_LABELS.items() if label == strategy_label.get())))
        entry(0, 2, "strategy_file", "打法 TOML 文件")
        entry(1, 0, "max_battles", "最多战斗")
        entry(1, 2, "minutes", "最长运行 / 分钟")
        entry(2, 0, "searches", "每场搜索上限")
        ttk.Checkbutton(self.task_config, text="启用资源筛选", variable=values["filter"],
                        onvalue="1", offvalue="0").grid(row=2, column=2, columnspan=2, sticky="w")
        entry(3, 0, "gold", "最低金币")
        entry(3, 2, "elixir", "最低圣水")
        entry(4, 0, "dark_elixir", "最低黑油")
        entry(4, 2, "total", "最低金币 + 圣水")
        goal_frame = ttk.Frame(self.task_config, style="Card.TFrame")
        goal_frame.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        ttk.Label(goal_frame, text="完成目标", style="Title.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 5))
        if kind == "resources":
            for row, (key, label) in enumerate((("target_gold", "金币库存"), ("target_elixir", "圣水库存"),
                                                 ("target_dark_elixir", "黑油库存")), 1):
                ttk.Label(goal_frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
                ttk.Entry(goal_frame, textvariable=values[key], width=22).grid(row=row, column=1, sticky="w", padx=8, pady=2)
            ttk.Label(goal_frame, text="达到满仓").grid(row=1, column=2, sticky="w", padx=(15, 7))
            flags = self.full_flags.setdefault(task_id, {name: tk.BooleanVar() for name in ("gold", "elixir", "dark_elixir")})
            selected = {name.strip() for name in values["full_resources"].get().split(",") if name.strip()}
            for row, (name, label) in enumerate((("gold", "金币"), ("elixir", "圣水"), ("dark_elixir", "黑油")), 1):
                flags[name].set(name in selected)
                ttk.Checkbutton(goal_frame, text=label, variable=flags[name],
                                command=lambda task=task_id: self._sync_full_resources(task)).grid(
                    row=row, column=3, sticky="w")
        else:
            ttk.Label(goal_frame, text="进度适配 TOML").grid(row=1, column=0, sticky="w", pady=2)
            ttk.Entry(goal_frame, textvariable=values["adapter_path"], width=32).grid(row=1, column=1, sticky="w", padx=8, pady=2)
            ttk.Label(goal_frame, text="进度目标").grid(row=2, column=0, sticky="w", pady=2)
            ttk.Entry(goal_frame, textvariable=values["target"], width=22).grid(row=2, column=1, sticky="w", padx=8, pady=2)
            if kind == "clan_games":
                ttk.Label(goal_frame, text="目标建筑").grid(row=3, column=0, sticky="w", pady=2)
                building_labels = {"": "由任务决定", "air_defense": "防空火箭", "spell_factory": "法术工厂"}
                label_var = values.setdefault("building_label", tk.StringVar())
                label_var.set(building_labels.get(values["building_type"].get(), "由任务决定"))
                building_combo = ttk.Combobox(goal_frame, textvariable=label_var,
                                              values=tuple(building_labels.values()), state="readonly", width=20)
                building_combo.grid(row=3, column=1, sticky="w", padx=8, pady=2)
                building_combo.bind("<<ComboboxSelected>>", lambda _event: values["building_type"].set(
                    next(key for key, label in building_labels.items() if label == label_var.get())))

    def _sync_full_resources(self, task_id: str) -> None:
        flags = self.full_flags[task_id]
        self.task_fields[task_id]["full_resources"].set(
            ",".join(name for name in ("gold", "elixir", "dark_elixir") if flags[name].get()))

    def _load_strategy_file(self) -> None:
        name = filedialog.askopenfilename(parent=self.root, title="选择打法 TOML", filetypes=[("TOML", "*.toml")])
        if not name:
            return
        self._load_strategy_path(Path(name))

    def _strategy_file_choices(self) -> None:
        folder = self.config_path.parent / "strategies"
        self.strategy_presets.configure(values=tuple(path.name for path in sorted(folder.glob("*.toml"))))

    def _choose_strategy_preset(self, event=None) -> None:
        name = self.strategy_preset.get()
        if name:
            self._load_strategy_path(self.config_path.parent / "strategies" / name)

    def _load_strategy_path(self, path: Path) -> None:
        try:
            from .strategy_config import load_strategy
            content = path.read_text(encoding="utf-8")
            load_strategy(path)
            try:
                display_path = str(path.relative_to(self.config_path.parent))
            except ValueError:
                display_path = str(path)
            self.task_fields[self.selected_task.get()]["strategy_file"].set(display_path)
            self._strategy_contents[self.selected_task.get()] = content
            self.strategy_text.delete("1.0", "end")
            self.strategy_text.insert("1.0", content)
        except (AutoCOCError, OSError, ValueError) as exc:
            messagebox.showerror("无法加载打法", str(exc), parent=self.root)

    def _save_strategy_file(self) -> None:
        name = self.task_fields[self.selected_task.get()]["strategy_file"].get().strip()
        if not name:
            messagebox.showerror("无法保存打法", "先选择打法 TOML 文件。", parent=self.root)
            return
        try:
            content = self.strategy_text.get("1.0", "end-1c")
            tomllib.loads(content)
            from .strategy_config import load_strategy
            path = Path(name)
            if not path.is_absolute():
                path = (self.config_path.parent / path).resolve()
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(content, encoding="utf-8")
            try:
                load_strategy(temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            self.status.set("打法已校验并保存。")
        except (AutoCOCError, OSError, ValueError, ImportError) as exc:
            messagebox.showerror("无法保存打法", str(exc), parent=self.root)

    def _load(self) -> None:
        self.config_label.set(str(self.config_path))
        self._strategy_file_choices()
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
        config = desktop_config(self.config_path, replace(options, routine=None))
        agent = config.vision_agent
        self.agent_enabled.set(agent.enabled)
        self.agent_jev.set(agent.jev_enabled)
        for key, variable in self.agent_values.items():
            variable.set(str(getattr(agent, key)))
        routine = options.routine or default_routine(config)
        self.task_kinds = {task.id: task.kind for task in routine.tasks}
        self.task_order = [task.id for task in routine.tasks]
        self.task_fields = {}
        self.full_flags = {}
        self._strategy_contents = {}
        self._editor_task = None
        for kind in TASK_KINDS:
            if kind not in self.task_kinds:
                self.task_kinds[kind] = kind
                self.task_order.append(kind)
        self.task_vars = {task_id: tk.BooleanVar(value=next((task.enabled for task in routine.tasks if task.id == task_id), False))
                          for task_id in self.task_order}
        self._render_tasks()
        for task in routine.tasks:
            self.selected_task.set(task.id)
            fields = self.task_fields[task.id]
            rf, goal = task.resource_filter, task.goal
            entries = {"strategy": task.strategy, "strategy_file": task.strategy_file,
                       "max_battles": task.max_battles, "minutes": f"{task.max_duration_sec / 60:g}",
                       "searches": task.max_searches, "filter": int(rf.enabled),
                       "gold": rf.min_gold, "elixir": rf.min_elixir, "dark_elixir": rf.min_dark_elixir,
                       "total": rf.min_total, "target_gold": goal.resource_targets.get("gold"),
                       "target_elixir": goal.resource_targets.get("elixir"),
                       "target_dark_elixir": goal.resource_targets.get("dark_elixir"),
                       "full_resources": ",".join(goal.full_resources), "adapter_path": goal.adapter_path,
                       "target": goal.target, "building_type": goal.building_type}
            for key, value in entries.items():
                fields[key].set("" if value is None else str(value))
        for task_id in self.task_order:
            self.selected_task.set(task_id)
        for key, value in (("max_runs", options.max_runs), ("minutes", f"{options.max_duration_sec / 60:g}"),
                           ("resources", options.min_expected_resources), ("searches", options.max_searches),
                           ("serial", options.serial), ("maintenance", f"{routine.maintenance_interval_sec / 60:g}")):
            self.values[key].set(str(value))
        self.mode.set("离线预演")
        self.strategy.set(STRATEGY_LABELS[options.strategy])
        self._strategy_contents = {}
        self._editor_task = None
        self.selected_task.set("resources" if "resources" in self.task_kinds else self.task_order[0])
        self._mode_changed()

    def _options(self) -> RunOptions:
        def seconds(value: str, name: str, allow_zero: bool = False) -> int:
            minutes = float(value)
            if not math.isfinite(minutes) or minutes < 0 or (not allow_zero and minutes == 0) or minutes * 60 != int(minutes * 60):
                raise ValueError(f"{name}必须为有效分钟数，精确到整秒")
            return int(minutes * 60)

        def optional(value: str) -> int | None:
            return None if not value.strip() else int(value)

        routine_tasks = []
        for task_id in self.task_order:
            kind = self.task_kinds[task_id]
            values = self.task_fields[task_id]
            goal = GoalConfig(
                resource_targets={name: int(values["target_" + name].get()) for name in
                                  ("gold", "elixir", "dark_elixir") if values["target_" + name].get().strip()},
                full_resources=tuple(item.strip() for item in values["full_resources"].get().split(",") if item.strip()),
                adapter_path=values["adapter_path"].get().strip(), target=optional(values["target"].get()),
                building_type=values["building_type"].get().strip())
            enabled_text = values["filter"].get().strip()
            if enabled_text not in {"0", "1"}:
                raise ValueError("资源筛选开关只能填写 1 或 0")
            resource_filter = ResourceFilter(enabled=enabled_text == "1",
                min_gold=optional(values["gold"].get()), min_elixir=optional(values["elixir"].get()),
                min_dark_elixir=optional(values["dark_elixir"].get()), min_total=optional(values["total"].get()))
            routine_tasks.append(TaskSpec(id=task_id, kind=kind, enabled=self.task_vars[task_id].get(),
                strategy=values["strategy"].get(), strategy_file=values["strategy_file"].get().strip(),
                max_battles=int(values["max_battles"].get()),
                max_duration_sec=seconds(values["minutes"].get(), "任务时长"),
                max_searches=int(values["searches"].get()), resource_filter=resource_filter, goal=goal))
        routine = RoutineConfig(tuple(routine_tasks),
                                maintenance_interval_sec=seconds(self.values["maintenance"].get(), "维护间隔", True))
        routine.validate()
        if not any(task.enabled for task in routine.tasks):
            raise ValueError("请至少勾选一项本次任务")
        tasks = ("launch",) + tuple(dict.fromkeys(self.task_kinds[task_id] for task_id in self.task_order
                                                    if self.task_kinds[task_id] in ("collect", "request", "donate")
                                                    and self.task_vars[task_id].get()))
        if any(self.task_vars[task_id].get() and self.task_kinds[task_id] in BATTLE_KINDS for task_id in self.task_order):
            tasks += ("battle",)
        resource_id = next((task_id for task_id in self.task_order if self.task_kinds[task_id] == "resources"), None)
        options = RunOptions(tasks, int(self.values["max_runs"].get()),
                             seconds(self.values["minutes"].get(), "最长运行"),
                             int(self.values["resources"].get()), int(self.values["searches"].get()),
                             self.values["serial"].get(), self.mode.get() == "离线预演",
                             strategy=self.task_fields[resource_id]["strategy"].get() if resource_id else "verified",
                             routine=routine,
                             vision_agent=VisionAgentConfig(
                                 enabled=self.agent_enabled.get(), jev_enabled=self.agent_jev.get(),
                                 **{key: (float(var.get()) if key in {"preparation_reserve_sec", "jev_timeout_sec"}
                                          else int(var.get()) if key == "evidence_limit_mb" else var.get().strip())
                                    for key, var in self.agent_values.items()}))
        options.validate()
        return options

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
        for task in (spec.id for spec in options.routine.tasks if spec.enabled):
            self.task_table.insert("", "end", iid=task, values=(self._task_label(task), "等待", ""))
        for value in self.counters.values():
            value.set("0")
        self.run_label.set("当前任务 · 准备运行")
        self.progress_label.set("等待首场战斗")
        self.status.set("离线预演中：不连接模拟器。" if options.dry_run else "正在连接指定设备…")
        self.badge.configure(text="离线预演中" if options.dry_run else "实机运行中")
        self.tabs.select(0)

    def stop(self) -> None:
        self.controller.stop()
        self.stop_button.state(["disabled"])
        self.status.set("停止请求已发送，等待当前有时限的调用返回并保存报告。")
        self.badge.configure(text="正在停止")

    def _running_controls(self, running: bool) -> None:
        for widget in (*self.editors, *self.task_editors):
            widget.state(["disabled"] if running else ["!disabled"])
        self.pages.tab(1, state="disabled" if running else "normal")
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
            self.progress_state = {}
            self.run_label.set("当前运行 · " + event["run_id"])
        elif kind == "battle_progress":
            task = event.get("task_id", "")
            if self.progress_state.get("task_id") != task:
                self.progress_state = {"resource_gains": self.progress_state.get("resource_gains")}
            self.progress_state.update({key: event[key] for key in
                ("task_id", "resources", "resource_gains", "goal_current", "goal_target") if key in event})
            progress = self.progress_state
            phase = event.get("phase", "进行中")
            number = event.get("battle_number", 0)
            self.run_label.set(f"{self._task_label(task)} · 第 {number} 场 · {phase}")
            for key in ("battles_completed", "battles_won", "goals_completed"):
                if key in event:
                    self.counters[key].set(str(event[key]))
            resources = progress.get("resources") or {}
            resource_text = "库存 " + " / ".join(f"{RESOURCE_LABELS.get(name, name)}: {value if value is not None else '未知'}" for name, value in resources.items()) if resources else ""
            gains = progress.get("resource_gains")
            gain_text = ("已核验收益 " + " / ".join(f"{RESOURCE_LABELS.get(name, name)}: {value if value is not None else '未知'}"
                                               for name, value in gains.items())) if isinstance(gains, dict) else ""
            goal = ""
            if progress.get("goal_target") is not None:
                current, target = progress.get("goal_current", "未知"), progress["goal_target"]
                if isinstance(current, dict) and isinstance(target, dict):
                    goal = "目标 " + " / ".join(f"{RESOURCE_LABELS.get(name, name)}: {current.get(name) if current.get(name) is not None else '未知'} / {value}" for name, value in target.items())
                else:
                    goal = f"目标 {current} / {target}"
            self.progress_label.set("  ".join(part for part in (resource_text, gain_text, goal) if part) or str(phase))
        elif kind in {"task_started", "task_result"}:
            task = event["task"]
            if "goals_completed" in event.get("metrics", {}):
                self.counters["goals_completed"].set(str(event["metrics"]["goals_completed"]))
            state = "running" if kind == "task_started" else event["status"]
            if self.task_table.exists(task):
                self.task_table.item(task, values=(self._task_label(task), STATUS_LABELS.get(state, state), display_reason(event.get("reason", ""))))
            self._log(f"{self._task_label(task)} · {STATUS_LABELS.get(state, state)}  {display_reason(event.get('reason', ''))}")
        elif kind == "finished":
            payload = event["summary"]
            self._show_counts(payload)
            final_tasks = {result["task"]: result for result in payload.get("task_results", [])}
            for task in self.task_table.get_children():
                result = final_tasks.get(task)
                state = STATUS_LABELS.get(result["status"], result["status"]) if result else "未执行"
                reason = display_reason(result["reason"]) if result else "运行已结束"
                self.task_table.item(task, values=(self._task_label(task), state, reason))
            self.report_path = event["report"]
            offline = payload["mode"] == "dry-run"
            stopped = "interrupted" in payload.get("stop_reason", "")
            self.badge.configure(text="已停止" if stopped else "运行失败" if payload["failures"] else "预演完成" if offline else "运行结束")
            self.status.set("已保存报告。" + ("预演结果不计入真实成功或收益。" if offline else payload["stop_reason"])
                            if self.report_path else "运行已结束，但报告未保存；请查看日志。")
            self._log("停止原因：" + display_reason(payload["stop_reason"]))
            self.refresh_history()

    def _log(self, text: str) -> None:
        self.short_log.insert("end", text)
        if self.short_log.size() > 4:
            self.short_log.delete(0)
        self.short_log.see("end")
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
