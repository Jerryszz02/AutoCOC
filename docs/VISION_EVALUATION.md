# 离线视觉验收

`scripts/evaluate_vision.py` 只读取本地图片、人工标签和配置，运行本地 `ScreenshotRecognizer`，不连接 ADB、不点击游戏。

工具的统计只代表提供的标注集。即使小样本得分为 100%，也不表示满足 [ACCEPTANCE.md](ACCEPTANCE.md) 的样本量、独立性、类别覆盖或现场任务要求。

## 人工标签

必须逐张打开原图，由人复核场景和数字后写入标签。**不能把 `inspect` 或识别器的输出直接复制为真值，也不能运行模型生成一批标签后直接称为人工验收集。** 验收截图应独立于用于调参、提取模板的截图；相邻连拍也不应被用于扩大独立样本数。

以下只展示格式，路径和数字是假设例子，不是已标注的项目图片：

```json
{
  "description": "人工复核的独立测试集；记录复核人和日期",
  "samples": [
    {
      "path": "images/manually-reviewed-village.png",
      "scene": "village",
      "resource_source": "village_inventory",
      "resources": {"gold": 123456, "elixir": 234567, "dark_elixir": 3456, "gems": null}
    },
    {"path": "images/manually-reviewed-unrecognized.png", "scene": "unknown"}
  ]
}
```

图片路径相对 manifest 所在目录解析，也可使用绝对路径。`scene` 使用项目的场景名：`starting`、`village`、`training`、`search`、`enemy_village`、`battle`、`settlement`、`clan_chat`、`donation`、`request`、`popup`、`disconnected`、`maintenance`、`unknown`。

没有 `scene` 的条目会列入 `excluded_unlabeled`，不会进入分母。`resources` 与 `resource_source` 可省略；未标注或为 `null` 的数字不计入数字分母，真实零值必须显式写为 `0`。数字标签只能是非负整数。

## 运行

按项目 README 安装后，在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_vision.py path\to\labels.json --config config.toml
```

默认结果写到配置的 `report_dir`，文件名包含时间与随机后缀。也可指定新文件：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_vision.py path\to\labels.json --config config.toml --output reports\vision-test-001.json
```

已有结果不会被覆盖。标准输出是带结果路径和主要指标的 JSON；结果文件另含每张图的真值、预测、错误及像素哈希，便于人工复算。识别/图片读取异常仍保留在分母，记为错误而非悄悄删除；有此类异常或没有带标签样本时退出码为 `1`。模型得分低本身不会伪装成工具故障，正常计算完成的退出码为 `0` 不表示通过完整验收。

## 统计口径

- 混淆矩阵的行是真值、列是预测。识别失败用 `__error__` 列记录。
- `macro_f1` 对本集合中**实际有真值样本**的类别等权平均；每类同时给出 precision、recall、F1、support。没有样本时为 `null`。
- `accepted_accuracy` 只统计置信度在 `[0.8, 1]` 且未发生识别错误的输出；`coverage` 是这些输出占带标签唯一输入的比例。没有接受结果时准确率为 `null`，不是 100%。
- `numeric_exact_ratio` 是已标注数字字段的精确正确数 / 字段总数，另按金币、圣水、黑油、宝石给出分项。若标注了 `resource_source`，来源不符时即使数值碰巧相等也计为错误。
- `unknown_actionable_false_positives` 统计真值为 `unknown`，却以至少 0.8 置信度输出可用于动作的游戏场景的次数。可动作场景集合写入结果；`starting`、断线、维护、未知不属于此集合。
- 同一解码后像素内容使用 SHA-256 去重，复制、改名或 PNG 重编码不会扩大样本数。重复条目的互补数字标签会合并；场景、来源或同一数字字段的标签冲突会明确报错，要求人工纠正。
- 像素去重不能识别“同一场景略微移动了一个像素”等近似重复；样本独立性仍须由人复核。工具不自动添加标签或生成所谓 200 张验收样本。

当前资源字段统计范围是 `resources` 中的金币、圣水、黑油、宝石。军队容量、请求冷却、按钮框准确率等仍须额外人工标注与验证，不能拿这些数字结果代替全部 `VIS-03` 要求。
