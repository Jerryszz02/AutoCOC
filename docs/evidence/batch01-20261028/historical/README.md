# 批次 1：旧 TRAIN 来源的 8 图试标

这是一组 **18.647.2 的 `agent_visual` 待复核候选**。8 张原图均与已有 TRAIN 来源关联；本次只是补充可讨论的建筑框，**新增已核实独立阵型数为 0**。当前客户端已更新到 18.647.5，这组旧图不能用于新版质量结论。没有视觉审阅 VAL/TEST 关联图片，也没有读取其标签或预测结果。来源筛选时曾对 10 张候选副本解码计算像素哈希，其中 2 张随后发现关联旧 VAL/TEST 并被排除；不能将这项哈希核对写成“零像素读取”。没有打开封存快照的原图/标签目录，也没有重新分组、冻结数据、导出标签或训练。

原图副本和可由现有工具读取的 `manifest.jsonl`、`annotations.jsonl` 在 `E:\projects\AutoCOC-training\candidates\building-th-ad-trial-20261028-v1`。本目录保留相同的两份元数据、[八图叠框缩览](contactsheet.jpg)、[来源与输出哈希回执](receipt.json)、[标签规则](LABEL_GUIDE.md)以及[混淆和缺口表](GAPS.md)。图中黄色 `TH?`、绿色 `AD?` 和橙色 `UNKNOWN` 都是提案；缩览仅供索引，应回到带 SHA 的 2560×1440 原图逐图复核。

来源筛选以 `reports/batch01-20261028/historical-reuse-audit.json` 为依据，PR 中另存[精简来源关联审计](source-association.json)：10 张历史候选中 8 张只关联既有 TRAIN，另外 2 张有 VAL/TEST 关联，被排除。前 4 张框来自 `reports/batch01-20261028/trial-first4.json`；后 4 张来自同轮原图视觉试标。源图和复制件 SHA、像素 SHA、尺寸均逐张核对。`split=unassigned` 是本次候选目录的状态；`historical_association=existing_train_only` 记录其不可重新分入 VAL/TEST。8 个来源会话组是**本目录内的关联键**，不增加历史快照的独立阵型数量，阵型是否相似仍待人工判读。

复核命令（在 `E:\projects\AutoCOC-batch01` 执行；输出报告已保存于本目录）：

```powershell
$root = 'E:\projects\AutoCOC-training\candidates\building-th-ad-trial-20261028-v1'
$python = 'E:\projects\AutoCOC\.venv\Scripts\python.exe'
& $python scripts\vision_data.py validate --root $root --manifest "$root\manifest.jsonl" --annotations "$root\annotations.jsonl"
& $python scripts\vision_data.py inspect --root $root --manifest "$root\manifest.jsonl" --annotations "$root\annotations.jsonl" --near-duplicates
& $python scripts\vision_candidate_coverage.py --root $root --manifest "$root\manifest.jsonl" --annotations "$root\annotations.jsonl" --task building --classes town_hall air_defense
```

结果：[schema 与原图校验](validation.json)为 8/8 有效、0 错误；[候选索引](inspect.json)为 8 个待复核来源组，组内 dHash 近重复候选 0；[覆盖报告](coverage.json)给出 8 个大本营候选框、19 个防空建筑候选框，但**可用于原型或正式数据冻结的图均为 0**。`complete=false`，没有任何完整确认的类别；没有框的地方不构成负例。[工具验证器](../../../../training/vision/data.py)要求 `buildings.complete_classes` 非空才可填，因此元数据用 `declared_complete_classes=[]` 明示“零个完整类别”，建筑标注不填写该字段；不放宽验证器。
