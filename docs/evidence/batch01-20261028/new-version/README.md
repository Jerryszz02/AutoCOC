# 18.647.5 建筑候选试标（2026-10-28）

本包包含 **36 张新版原图、36 份部分试标**，来自 5 个已回村并完成费用核账的窗口。每张保留原文件/像素 SHA、客户端版本、采集会话、原始路径与复制链。均为 `agent_visual/pending`、`complete=false`、`split=unassigned`；没有穷尽确认的类别，未标区域不是负例，没有正式数据冻结或训练。

原图及工具可直接读取的完整目录为 `E:/projects/AutoCOC-training/candidates/building-th-ad-186475-trial-20261028-v1`。本目录保存同字节的 [manifest](manifest.jsonl)、[annotations](annotations.jsonl)、[复制/来源回执](receipt.json)和脱去玩家名区域的[缩览](contactsheet.jpg)。原图保持原样，缩览不能替代原图复核。[打包脚本](package_candidates.py)是本轮一次性证据工具，路径和窗口固定，拒绝覆盖目标；原始采集窗不被改写。

| 检查 | 结果及边界 |
| --- | --- |
| 源文件与像素/schema | [36/36 有效](validation.json)，0 错误、0 警告 |
| 试标 | 35 个大本营候选框、128 个防空候选框、15 个未知区域 |
| 缩放 | `{'as_observed': 34, 'minimum_stable': 2}`；按原 manifest 记录，未用后补推测提升为最小缩放 |
| 来源组 | 5 个采集会话关联组，阵型组仍 provisional；不是 36 个已确认独立阵型 |
| 近重复 | [4 对 dHash 候选](inspect.json)已逐对原图做[结构初查](near-duplicate-visual-review.json)，墙体连接、核心和塔位置均有可见差异；仅 agent_visual，不将其升级为所有阵型独立的正式证明 |
| 数据资格 | [原型 eligible=0、正式 eligible=0](coverage.json)；完整性和复核门保留 |

`coverage.json` 中 `definite_independent_groups=5` 是工具按已提供的分组键计数；本包分组仍临时，因此本轮不将该字段解释成已经确认的独立阵型数量。近重复结构初查不代替完整布局组核对；正式拆分和训练仍未放行。

试标规则沿用[标签指南](../historical/LABEL_GUIDE.md)，新版新增混淆包括蓝金/红金/绿金防空外观、英雄或宠物遮住大本营屋顶、底部卡栏截断基座。无法判定的发射器放入 unknown，不按对称布局或预期数量补框。特别是窗 02 第 2 张大本营在底部遮挡区，没有强行给正框；部分防空只给可见主体提案，仍需精确边界复核。

第 1 批尚未满足正式出口：本轮 36 张未达到约 50 个阵型的诊断目标；三个窗口因准备倒计时短或读数不可用安全提前结束，达到窗口上限后停止，没有补凑数量。继续前先处理上述视角、完整标注、近重复与复核缺口；本轮的 8 张旧 TRAIN 试标另存[历史包](../historical/README.md)，不能补充新版独立数量。

复验用现有 `scripts/vision_data.py validate`、`inspect --near-duplicates` 和 `scripts/vision_candidate_coverage.py --task building --classes town_hall air_defense`，将 `--root` 指向上述外部原图目录并使用其 manifest/annotations。仅校验候选，不执行 split/export/train。
