# 本地视觉训练工具

为另行验证黄色 HP，按原顺序离线检查 101600 和 113248 两个既定 TRAIN 全组 **279 帧**（272 程序帧 + 7 raw），零读取错误、原图/源码哈希不变。发现 113248 的 162/165/166/167/168 五张像素候选具备同帧黄 HP、闭卡、身份及 ready 成对锚；Root 已原尺寸复看 162/168，实际仍在战斗中。当前生产仍拒绝 yellow-ready，拟议修复尚未集成；该诊断不是人工真值、模型训练或输入授权，不使用或重评 120916/旧 VAL/TEST。全帧回执 `diagnostics/warden-yellow-train-audit-run-v1/receipt.json`（`77b7c28d7a782e9b4dcdadf39659adc753d177fc5ad287c96f40c84e910a575e`）。

新增独立 VAL 诊断快照 `r-card-dense-agent-val-20261017-120916-v1`，与旧 v5 快照并列保存，**2 帧、1 组、1 正窗、0 负窗**。整场采集前已声明 VAL；三张 pre 原图逐图 Agent 复核后，前两张间隔 0.118909 秒且严格相机证明通过，第三张位移 3.394 像素超门限而隔离，未采到 post。外部 raw 适配器保留全部原字节及 admit/quarantine 原因，既有 `import_reviewed_r.py` 完成权威校验；旧45帧和模型文件不变，90对哈希无重复或近重复冲突。冻结 v5 仅评估一次，**not_established、0 TP / 0 FP / 1 FN、2 帧部署未验证、0 可评估正窗、recall=null**：现有 `dynamic_tracking` 不接受黄血条部署证据，未绕过该门。模型仍禁用，本组没有训练或调参；单组正窗不构成正式质量估计，`event_at` 是首次观察可用而非真实起始时刻。证据 `reports/completion-20261015/diagnostics/r-dense-120916-pipeline-v1/`；评估报告 SHA `cf4a7b660fe4803fa2cce950e9dbe85c00a437a8d3c8e7e373c585ef4b121d44`。

已完成的只读 picker 采样可导入独立的待复核候选队列：

```powershell
python scripts/vision_data.py collect-picker --config C:\work\AutoCOC\config.toml --output C:\data\new-picker-run --max-troop-pages 3 --limit-sec 150
python scripts/vision_data.py import-picker-candidates --source-run C:\data\picker-run --output C:\data\picker-candidates
```

`collect-picker` 需明确实机启动，输出必须是项目检出目录外的新目录；当前客户端限定 18.647.2，兵种最多采 5 页、法术只采首页。它会打开/关闭选择器并最多滑动兵种 4 次，输入前核验前台、当前帧及军队证明。GUI 的“数据工具”页提供同一独立动作。校准仅打包小型箭头模板和历史来源哈希；来源帧/事件引用是审计凭据，不宣称完整旧图已随项目提供。

导入工具校验 run 摘要、事件、前后完整军队证明与各页 PNG，然后只复制兵种页和法术首页原图。新目录的 `queue.json` 与 `manifest.jsonl` 使用 `picker_candidate_queue_v1`，图像路径均相对该目录；运行时已选行仅是识别证据，不是下方单位身份标签。全部候选保持 `pending`、无人工复核、无身份标签、`training_ready=false`；此队列不供 scene/building 或 R 窗口训练入口直接使用，也不创建 split 或冻结快照。输出目录必须尚不存在。采集与导入是两个独立命令，采集不会自动导入。

R-card v5 已加入 101600 的六张 Agent 复核 TRAIN：**29 train / 4 组、16 val / 2 组**，旧 39 帧和拆分不变，post_1 闪光帧隔离。固定 v4 配方实际训练 13 epoch、best 5，ONNX parity 16/16、最大误差 3.81e−6；事件仍 0 TP / 0 FP / 2 FN，继续禁用。没有读取 TEST/heldout 或手工查看旧 VAL 语义。新快照 `r-card-warden-agent-prototype-20261017-101600-v5` 与对应 `-35e` run/package 位于独立训练目录；根审计 `diagnostics/r-card-101600-training-root-v1.json`。本次未调整验证规则或阈值，不能将训练/导出成功当作事件验收。

R-card v4 使用与 v3 相同的 **23 train / 3 组、16 val / 2 组**，实际完成 35 epoch，best epoch 28，验证 NLL **0.043395**；ONNX parity **16/16**，最大误差 **5.96e−6**。事件评估仍为 **0 TP / 0 FP / 2 FN**，原型保持 unvalidated / runtime disabled，selected 类仍无验证样本。无可接受 observation 的帧由 11 降至 6，不等于 argmax unknown 类数量；两版该类数量都是 3。第一个正例已有高置信 available，但相邻帧间隔均超过 1 秒；另一个置信度低于 .95 且缺少连续两帧部署证明。相同验证帧、来源、标签与分组保持不变，held-out 未读取；hero_state 源版本有变化，不能将整体比较称为纯单变量实验。原流水线 trace 失败收据保留，根代理只修外部报告助手并独立重跑 trace，9.48 秒完成，结果与原评估一致，没有重复训练。证据 `external/training/r-card-warden-agent-prototype-20261017-v4/evaluation-val.json` 与 `val-frame-trace-root-v1.json`（`e4d73660fb5ca8f571e25387a09e7cd90de254bb21ceda99828abd5159615264`）。

以下为保留的历史 v2 实录：R-card v2 已实际完成 **12 epoch**、ONNX 导出和扩展 val 评估，18 train / 2 组与 16 val / 2 组。导出一致性 **16/16**，最大误差 **1.05e-5**；ONNX SHA `5af0535e61c402d6b39c26e4b2c64de7ce3d2d95fcbaebd05ddd9a3d45d62ecd` 与 v1 相同，因此本轮是扩展验证覆盖，没有证明推理权重改进。帧 argmax 分数 1.0 未应用生产置信门、连续帧或英雄部署条件，且无样本的 selected 类不计入宏分数；运行事件仍为 **0 TP / 0 FP / 2 FN**。逐帧 CPU 诊断与原指标精确对账：新 val 的四张 available 帧概率仅 **0.532–0.684**，均未达 .95，中间两张英雄肖像置信也未达 .95；旧 val 的时间间隔不足。模型继续禁用，不改旧 FN 或阈值。流水线 `diagnostics/r-card-pipeline-20261016-235453-818523/pipeline.json`，独立复核 `diagnostics/r-card-pipeline-independent-review-v2.json`（SHA-256 `3b6f651fee17d8f4065a96659e1632828fa474abc79d1c788514a1783275135a`），逐帧诊断 `diagnostics/r-card-runtime-diagnostic-v1-20261016T230701516318Z.json`（SHA-256 `827de722d6efbabccf47a34b8169f77de502c8edb0c8313d95226b0adf0d7024`）；输入/源码前后不变，test 未解析、解码或推理。

R-card v2 另建 **34 帧 train/val-only Agent 原型快照**：18 train / 2 组、16 val / 2 组；8 个窗口，旧 6 窗口/25 帧字段不变，新增 233306 独立 val 的 9 帧。新正窗相邻间隔为 **0.350880 / 0.280604 / 0.306417 秒**；输入后首帧 unknown，其余 4 帧 exhausted。`event_at` 仅是首次观察到可用的时刻，不是真实可用转换时间。既有校验器会解析全部窗口，因此旧 test 的 2 行和 10 张图另作不透明归档，不进入训练器；没有声称全 split 去重或 test 评估通过。工作快照签名 **`68f4d111dda74bfc16eee3af98cf74d7c6dba80037d23ec45942109168ba5169`**，原型冻结收据 `diagnostics/r-card-warden-prototype-20261016-v2-freeze.json`，独立审查 `diagnostics/r-card-windows-v2-newval-independent-review.json`（SHA-256 `85afa7aab77c0fcb08cc7bfcd321525de4a70edf14df45fffd2c8f196af4f59e`）。该快照随后已完成实际训练、导出和 val 评估，结果见本页最新记录；human/formal/runtime 仍为 false，selected 仍无样本，旧 val 的时间不足与 FN 保留。

R-card 原型另发现事件窗口的语义缺陷：训练场 221530 的负窗包含技能输入前连续可用帧，按当前“稳定可用”规则可能形成事件；这是标签与规则的条件冲突，未声称现有模型实际产生了 FP。外部 v2 草案把输入前 available 移到同场已有正 episode，负窗只留输入后的 unknown/exhausted，保留 6 窗口、25 帧、全部帧字段及原分组/拆分。独立检查无 P1/P2；只有 221530 正窗具备小于 1 秒的连续对子，旧 070901/071728 仍时间采样不足，历史 FN 不改。原草案保留不改；新增独立验证场和新版原型快照进展见本文最新记录，原型权重继续禁用。证据 `tools/r-card-windows-v2-draft/`、`diagnostics/r-card-windows-v2-draft-independent-review.json`（SHA-256 `0b7db6b6f2a7f5305847667c1aceb277227d9db1a554873f0022dac31f4fc272`）；原 v1 和不透明隔离的 test 记录未改。

2026-10-16 已实际训练首个大守护者 `r_card` Agent 原型：35 帧、8 窗口、4 个独立战斗/阵型组，按组冻结为 18 train / 7 val / 10 test；test 组因历史近似关联单独隔离，本次没有语义查看、推理或调参。MobileNetV3-Small 在 RTX 3070 Ti 上完成 12 epoch 后按配方早停，CPU FP32 ONNX 的 7/7 val 帧通过数值一致性（最大绝对误差 1.05e-5）。每帧 argmax 分数虽为 1.0，但 `selected` 没有样本，只有一个 val 组；事件评估为 **0 TP / 0 FP / 1 FN**。两张 available 帧相隔 3.48 秒，超过双帧规则的 1 秒间隔，另有 3 帧低于 .95 置信门；原型继续禁用，不能称为正式事件准确率。包位于 `E:/projects/AutoCOC-training/packages/r-card-warden-agent-prototype-20261016-v1-35e`；流水线及独立核验见 `reports/completion-20261015/diagnostics/r-card-pipeline-20261016-224939-544479/pipeline.json`、`r-card-pipeline-independent-review-v1.json`。全部来源、快照、配置、权重及产物哈希一致，标签仍是 `agent_visual`，未转为人工真值。

2026-10-16 扩充后的大本营单类 Agent 原型为 17 张、16 组，14 train / 2 val / 1 test。新增 6 张来自 5 个会话，只加入 train；旧 11 张的分配和字节保留，命中历史 test 近似哈希的整个会话 3 图排除。固定原配方的 YOLO11n/YOLO26n 分别完成 47/38 epoch、ONNX 导出与 val parity 2/2；阈值 0.7 下分别为 1 TP/1 FN 与 0 TP/2 FN，AP50/AP50–95 为 1.0000/0.4015 与 0.5187/0.3080。两张 val 不能支持正式质量结论，模型继续禁用。首个 OOM/MemoryError 中断目录保留，新尝试分别使用独立目录，源码及输入前后不变。比较报告为 `E:/projects/AutoCOC-training/reports/building-current-th-expanded-postb11-20261016-v1/two-model-comparison-resume-a-root-v1.json`；哈希、schema 和格式转换之外没有 test 语义查看、推理或调参。

2026-10-16 新增当前 18.647.2 的大本营单类 Agent 原型，11 张/11 个来源组按 seed 42 冻结为 8 train / 2 val / 1 test。相同 640 输入、batch 2、最多 60 epoch、patience 20 下，YOLO11n 实际训练 57 epoch，YOLO26n 训练 60 epoch；两者完成 ONNX 导出，全部 2 张 val 的数值一致性通过。运行点阈值 0.7 下两者均 0 TP / 2 FN，不能启用；独立 AP50 / AP50–95 分别为 1.0000 / 0.5030 和 0.2525 / 0.1017，只是两张 val 的原型结果。test 仅用于工具的快照完整性/标签 schema 校验，未用于模型推理、指标、人工查看或调参。源码与输入绑定均固定于 `b8b427e`，比较报告位于 `E:/projects/AutoCOC-training/reports/building-current-th-only-prototype-20261016-v1/two-model-comparison-root-v1.json`；模型保持 `prototype_only/unvalidated`，不与下述旧范围实验直接比较。

S/R 训练配方 `trainer_recipe_version=3` 在头部训练时同时冻结主干参数和 BatchNorm 运行统计；进入微调时重置早停计数，保留全局最佳权重。checkpoint 先按宏 F1、最低类别召回的既有字典序排序；两项完全相同时，validation NLL 至少改善固定 `0.001` 才替换最佳权重并重置早停计数。连续 7 次复合评分无改善才早停。S/R history 每轮保存 validation NLL 和不依赖阈值的概率/类别间隔摘要，并分别标明 NLL 用作 `checkpoint_tiebreak`、scene 阈值接受率只作 `diagnostic_only`（R 的阈值字段为 `not_applicable`）。配方写入运行配置和恢复签名，旧配方 checkpoint 不能混用 resume；旧运行和导出产物保留用于对照。

新冻结拆分评估统一以 `evaluation_scope.image_count` 表示所选 split 的样本数，仅 test 同时输出兼容字段 `test_image_count`。发布验证支持旧字段；双字段并存时须为一致的严格整数，不能让 val 报告误看作已评估 test。历史原件保留其原字段，应结合 `evaluation_scope.split` 解释。

训练环境与日常助手 `.venv` 分开。先在隔离环境中安装匹配的 PyTorch/torchvision CUDA wheel，再安装 `requirements-training.txt` 的其余顶层依赖；程序不会下载预训练权重。当前 Windows/Python 3.13.12/RTX 3070 Ti 实测 torch 2.13.0+cu126 与 torchvision 0.28.0+cu126，完整已安装版本写入每次运行的 `environment.json`。其他平台需重新核对兼容性与冒烟，不把这份记录当成通用锁定环境。

Ultralytics 的设置在每次运行目录隔离；首次导入前即写入 `sync=false`、关闭集成日志，设置 `YOLO_OFFLINE`，并在本训练进程中阻断 socket 出站连接。离线依赖缺失时运行会失败，需先由操作者准备，不会边运行边安装。

输入必须是 `scripts/vision_data.py export` 产生的完整复核快照，带 `SHA256SUMS`、`train.txt`、`val.txt`、`test.txt`、`split.json` 和标签文件。训练和导出会核对全部文件哈希。32 图 2 epoch 冒烟用 `--prototype --max-epochs 2`。原型完成冒烟后，可用 `--prototype --development --max-epochs 20` 在同一冻结划分上使用全部训练及验证图探索；场景最多 35 epoch，建筑最多 100 epoch 且 batch 不超过 8，仍不可把测试集用于调参。建筑训练把原图和标签复制到运行目录，避免 Ultralytics 缓存写入冻结快照。原型数据和模型始终标 `prototype_only`，不能充当正式验收。

场景人工复核使用 `scripts/review_vision_scenes.py`。源目录须已有通过校验的 `manifest.jsonl`、`annotations.jsonl` 和原图；日志必须放在源目录外。启动后在本机浏览器打开终端给出的地址，逐张查看原图，由复核者填写自己的名字并明确确认建议、改标签或跳过。方向键只翻图，不会提交；跳过也可撤回该图之前的确认。工具不会替用户确认，也不会覆盖原有 Agent 标注。

```powershell
python scripts/review_vision_scenes.py serve --root C:\data\scene-source --journal C:\reviews\scene-verdicts.jsonl
python scripts/review_vision_scenes.py export --root C:\data\scene-source --journal C:\reviews\scene-verdicts.jsonl --output C:\data\scene-human-reviewed
```

服务仅监听 `127.0.0.1`，页面地址仅供本机复核。日志追加保存每次明确操作，绑定原图、来源清单和标注哈希；来源变动会拒绝提交或导出。导出只包含最新操作为确认或改标签的图，保留原始来源与全部复核历史；目标须是新目录，全部写入和校验成功后才发布。零确认时拒绝导出。导出结果还须经过既有分组、覆盖检查、`split` 和 `export` 才成为训练快照；它不自动满足正式数量门槛，也不会把原型模型改为正式模型。

未冻结候选池可先用 `scripts/vision_candidate_coverage.py` 检查每类的独立来源组、完整标注范围及预计拆分缺口。输入必须明确为 `split=unassigned`，工具拒绝已有快照或拆分目录；它只生成诊断报告，不写标签、split 或快照，也不启动训练。Agent 标注仅可计入原型资格，正式资格另列；近似重复待复核时不预测拆分。报告必须保存在候选根目录外的新文件中。

```powershell
python scripts/vision_candidate_coverage.py --root C:\data\building-candidates --manifest C:\data\building-candidates\manifest.jsonl --annotations C:\data\building-candidates\annotations.jsonl --task building --classes town_hall air_defense --output C:\reports\building-candidate-coverage.json
```

这里的预计 train/val/test 覆盖仅使用传入的未分配候选，不读取已封存测试集；类别缺口需要补样本，不能用拆散同组或降低验收门槛消除。

建筑可按明确的类别范围制作快照。逐张标注中的 `buildings.complete=false` 和 `buildings.complete_classes=["town_hall","air_defense"]` 表示审核者已在**整张图**穷尽这两类，包括该类确实为零实例的图；它不表示只框出了目前看见的几个目标。若遮挡、裁切或证据不足以确认某类已穷尽，不要把该类写进 `complete_classes`。该字段只接受非空、无重复的注册类别，且必须保留审核人、方法与证据。旧的 `complete=true` 仍表示全部建筑类已穷尽。

对类别子集，`split` 与 `export` 必须以相同顺序显式传 `--classes town_hall air_defense`。拆分仅选择每张图都对这两类完整复核的样本；快照保留原标注供核查，但训练标签只含启用类，且 split、snapshot、class mapping 三处固定类别顺序与稳定 ID。任何类别或 taxonomy ID 变化均拒绝导出或训练前验证。没有显式 `--classes` 的旧拆分仍只接受 `complete=true`，不会自动放宽到部分标注；正式训练仍要求人工复核和已知客户端版本，`--prototype-only` 不会升格为正式数据。

需要补充实机候选图时，可由独占模拟器的操作者运行 `python scripts/collect_vision_samples.py --config config.toml --output C:\data\new-collection --max-samples 10 --max-search-cost-gold 20000`。输出目录须事先不存在，包含原图、`unreviewed-manifest.jsonl` 和 `collection.json` 搜索费用记录。同一采集会话的图必须归入同一泄漏组；这些图尚未逐张复核、标注，不能直接导出正式训练快照。

配置 JSON 必填 `task`、`code_version`、`data_version`、`client_versions`、`source_width_range`、`source_height_range`、`zoom`、`output`。场景还需按模型输出顺序给 `classes`；建筑类顺序从快照 `class_mapping.json` 读取，并要求 `architecture` 为 `yolo11n` 或 `yolo26n`。分辨率和版本应由已复核快照实际覆盖，`output` 阈值由验证集定稿。示例原型配置：

```json
{"task":"scene","code_version":"local-sha","data_version":"reviewed-v1","client_versions":["actual-reviewed-version"],"source_width_range":[2560,2560],"source_height_range":[1440,1440],"zoom":"minimum","classes":["enemy_village","battle"],"output":{"format":"logits","min_confidence":0.8,"min_margin":0.15},"prototype_only":true}
```

建筑 `output` 应为 `{"format":"yolo_xywh_classes","coordinates":"input_pixels","min_confidence":0.7,"nms_iou":0.5,"max_candidates":1000,"max_detections":300}`。只传本地 `--pretrained` 路径；正式场景训练要求 torchvision ImageNet state dict，建筑训练要求本地 YOLO `.pt`。训练与导出命令：

```powershell
python scripts/train_vision.py scene --snapshot C:\data\scene-v1 --config C:\data\scene-config.json --output C:\runs\scene-smoke --device cpu --prototype --max-epochs 2 --deadline-hours 1
python scripts/export_vision_model.py --training-dir C:\runs\scene-smoke --snapshot C:\data\scene-v1 --package C:\models\prototype\scene
```

导出生成 CPU FP32 ONNX、manifest、同输入数值对比报告，并通过真实运行时 loader。2 epoch 冒烟只测一张验证图，原型开发训练核对全部验证图，正式导出最多核对 100 张；正式包少于 100 张则报告 `parity_sample_sufficient=false`。建筑包仅在 ONNX 图检查未见过滤或 TopK/NMS 算子且确认 raw dense 输出时声明独立 AP 可用的原始输出来源。parity 只验证兼容性，独立测试精度和实战仍须另行验收。`--resume` 使用同配置、同快照和同总 epoch 上限，场景恢复模型、优化器、调度器及随机状态；建筑恢复 Ultralytics 的 `last.pt`。超时后保留最后 checkpoint。

2026-10-15 的本机原型记录：S 冻结快照 `E:\projects\AutoCOC-training\datasets\scene-prototype-20261015-v2` 有 59 train / 13 val / 13 test；32 图 2 epoch 冒烟成功，随后 `--prototype --development --max-epochs 20` 使用全部 59/13 图，在第 16 epoch 早停，最佳验证 macro-F1 0.75385、minimum recall 0。B 冻结快照 `E:\projects\AutoCOC-training\datasets\building-prototype-20261015-v2` 有 35 train / 8 val / 7 test；32 图 2 epoch 冒烟 mAP50 0.0011。两条 35/8 开发基线的命令仅替换配置、预训练权重和输出目录：

```powershell
python scripts/train_vision.py building --snapshot E:\projects\AutoCOC-training\datasets\building-prototype-20261015-v2 --config E:\projects\AutoCOC-training\configs\building-prototype-20261015-v2.json --output E:\projects\AutoCOC-training\runs\building-prototype-20261015-v2-yolo11n-dev100 --device 0 --pretrained E:\projects\AutoCOC-training\weights\yolo11n.pt --prototype --development --max-epochs 100 --deadline-hours 1
python scripts/train_vision.py building --snapshot E:\projects\AutoCOC-training\datasets\building-prototype-20261015-v2 --config E:\projects\AutoCOC-training\configs\building-prototype-20261015-v2-yolo26n.json --output E:\projects\AutoCOC-training\runs\building-prototype-20261015-v2-yolo26n-dev100 --device 0 --pretrained E:\projects\AutoCOC-training\weights\yolo26n.pt --prototype --development --max-epochs 100 --deadline-hours 1
```

YOLO11n 在第 31 epoch 早停，YOLO26n 在第 33 epoch 早停，均为 640 输入、AdamW、batch 2、workers 0、patience 20。两者最佳权重导出的 CPU FP32 ONNX 均通过真实 loader 和 8/8 验证图同输入 parity，保留 8400 原始候选，图内无 NMS/TopK。固定独立验证协议（8 图、6 组、未看 test）的 YOLO11n AP50/AP50–95 为 0.67186/0.23163，YOLO26n 为 0.60792/0.21426，结果在 `E:\projects\AutoCOC-training\reports\building-yolo{11,26}n-dev100-val.json`。YOLO26n 原始低分背景锚框有 23 个无效几何，最大分数 5.89e-5；导出报告明确记录，运行时和独立 AP 按无效框过滤。首次过严检查造成的导出失败目录保留，新包以 `-retry` 后缀输出。这些全部是 `prototype_only/unvalidated`：标注为 agent_visual，混有未知旧版本，验证集过小，测试集从未用于调参，也没有战斗可用性证据。

当前版 S 原型另用 63 张、7 个保守分组，固定 43 train / 9 val / 11 test。旧配方以 20 epoch 为上限，在第 10 轮早停；按 argmax macro-F1 选出的 `best.pt` 在验证集为 0.58974，但运行阈值 0.8 / margin 0.15 下 9/9 拒识。`last.pt` 的验证 argmax macro-F1 更低（0.51515），虽接受 4/9 图，却把一张 battle 高置信识别为 enemy_village；没有据此换用最后权重或降低阈值。旧包的 CPU FP32 同图 parity 为 9/9。旧报告在本地 `reports/completion-20261015/current-scene-checkpoint-diagnostic.json`，旧包为 `E:/projects/AutoCOC-training/packages/current-scene-prototype-20261015-v1`。

修复头部 BatchNorm 冻结与微调早停计数后，以相同的 43 train / 9 val / 11 test 冻结划分、seed 42、运行阈值 0.8 / margin 0.15 重新训练。新配方运行了 12 轮（5 轮头部、7 轮微调），全局最佳为从零计数的 epoch 4；验证集 argmax macro-F1 和已出现类别的最低 recall 均为 1.0，但验证集只出现 3 类，且各来自 1 个来源组。按生产运行阈值离线判定时只接受 2/9（1 张 battle、1 张 settlement），其余 7 张为 unknown，3 张 enemy_village 全部拒识，coverage 22.22%。CPU FP32 ONNX 对同一验证图的 parity 为 9/9，最大绝对差 5.7220459e-6。训练源码提交为 `8038ee391f600b26182edfbeca3a59656f785aac`；新运行与包分别为 `E:/projects/AutoCOC-training/runs/current-scene-prototype-20261015-v1-recipe2-20epoch` 和 `E:/projects/AutoCOC-training/packages/current-scene-prototype-20261015-v1-recipe2-20epoch`。两次诊断均只使用冻结 val，11 张 test 未用于选型，也没有新增实机通过证据；新旧包都保持 `prototype_only/unvalidated`，不发布。

`trainer_recipe_version=2` 的历史诊断实验只记录逐轮验证交叉熵及原运行时阈值下的按类接受/拒识，不改变当时的最佳权重排序。`3c575ee` 上实际重跑同一 63 图快照、seed 42 和原配方，训练 53.11 秒；最佳 epoch 4、最后 epoch 11 各 244 个模型张量与旧配方 2 逐项完全一致。新包 `current-scene-prototype-20261016-validation-diagnostics` 的 CPU FP32 parity 9/9、验证接受 2/9，仍未发布。早期 epoch 3 接受 9/9，但 argmax macro-F1 只有 0.515，不能按总接受率换用它。独立记录为 `scene-validation-diagnostics-experiment.json` 和 `scene-validation-diagnostics-checkpoint-verification.json`；未使用 test 选型。当前配方 v3 才将固定 NLL 最小改进作为同分 tie-break；历史 v2 结果不回写。

R 的建筑状态和技能卡状态使用独立的四类空间与独立模型包，不能把 `truth_event` 当逐帧状态标签。`training.vision.dynamic_snapshot.freeze_dynamic_snapshot` 先把审核过的动态窗口冻结为带 `windows.jsonl`、`snapshot.json`、`split.json` 和 `SHA256SUMS` 的不可覆盖目录；训练只消费 `verify_dynamic_snapshot` 全面复核后的副本。每帧必须有 `state_label`、审核人与证据、目标身份、源图哈希和原图 ROI；同一事件/会话/阵型连通组不可跨 train/val/test。R 训练用 224×224 原图 ROI、ImageNet 预训练 MobileNetV3-Small、按类与事件连通组采样，最多 35 epoch、5 epoch 头部训练；早停沿用宏 F1/最低召回优先级，仅用固定 NLL 改善打破精确平分，连续 7 次复合评分无改善后停止。签名绑定快照、拆分、权重、种子与 v3 配方，可用同一运行目录 `last.pt` 恢复；v2 checkpoint 会因签名不同而拒绝恢复。CPU FP32 ONNX 导出通过真实 `DynamicModel` 加载与 ROI 推理，并比较最多 100 张验证帧的同输入 logits；报告仅证明导出兼容。

离线策略候选可使用 `autococ.dynamic_strategy.propose_configured_hero_ready_wait_from_sources`：传入两个原图路径、帧元数据、同一英雄卡 ROI、已加载 `DynamicModel` 及显式等待/技能配置，由入口内部读取字节、计算 SHA、解码、核验当前英雄，再推理并更新同一 tracker；不接受外部预测步骤。英雄识别使用同一字节的私有临时快照，模型只获得独立只读像素副本，输入被修改会拒绝候选。旧 `propose_configured_hero_ready_wait` 只核对外部步骤声明，不能证明推理来源。两者均保持 `execution_authorized=false`，不绕过正式评估元数据门；采集时间/相机声明、真实模型质量及运行时发布仍须另行验证。

`r_card` 只训练英雄技能卡，逐帧 `target_id` 必须为 `hero_skill:<已登记英雄ID>`，不能使用普通兵、攻城或未明确身份的卡。四类 `available/selected/exhausted/unknown` 仍是单帧状态；复合 `skill_available` 另要求连续两帧在各自**同一原图**上确认该英雄已部署。每个 R-card 标注帧须有单独复核的 `hero_deployment_truth`：`hero_id`、`card_identity_verified=true`（已确认原图 ROI 是该英雄的己方技能卡）、布尔 `deployed`、审核人/方法/证据、原图 SHA、ROI、frame/session/camera/capture/client 绑定；负例可明确标 `deployed=false`，正事件至少两帧 `available` 且 `deployed=true`。离线评估不会把此真值当模型读数，而是从哈希核对过的原图调用生产英雄卡头像与封闭 HP 识别；读不到、身份或版本不符、低置信、过期或未部署均打断事件连续性。连续证据不能跨客户端版本、模型权重或推理合同拼接。没有可独立确认的正例时技能事件指标为 `not_established`。合成测试只验证工具链；当前 Agent 复核原型含真实技能正窗口，val 的一个正事件仍漏检，正式人工正例与正式事件验收尚未完成。

已复核的 R 窗口可通过 `scripts/import_reviewed_r.py` 从 `vision_review_queue.py` 候选队列导入。输入是现有 `dynamic_windows_v1` JSONL；每帧另用 `queue_session`、`queue_frame_id` 绑定队列里的原图，保留明确的状态、事件、相机、ROI、审核和 split。路径、SHA、版本和采集时间只取自经哈希核验的队列，不接受审核文件覆盖；工具不代填标签或分组。

```powershell
python -m scripts.import_reviewed_r --queue C:\reviews\queue.json --reviews C:\reviews\r-windows.jsonl --source-root C:\data\live --kind r_card --output C:\data\r-card-frozen --prototype-only
```

不传 `--prototype-only` 时要求人工复核。技能卡必须已有同源、同 ROI、同版本的英雄身份与部署复核，才允许从队列 `hero-card:<id>` 映射为 `hero_skill:<id>`；不能仅改前缀。导入的建筑窗口须保持同一目标；共享验证器也拒绝正建筑事件跨目标，同时允许直接构造的跨目标负例用于跟踪反例。输出仅在来源、窗口、分组及快照全部验证成功后发布到新目录，拒绝覆盖。当前真实已复核 R 窗口仍为零，不能对 pending 队列直接运行后宣称训练就绪。

```powershell
python scripts/train_dynamic_state.py --snapshot C:\data\r-building-frozen --config C:\data\r-building-config.json --output C:\runs\r-building --device 0 --pretrained C:\weights\mobilenet_v3_small-047dcff4.pth --max-epochs 35 --deadline-hours 4
python scripts/export_dynamic_state.py --training-dir C:\runs\r-building --snapshot C:\data\r-building-frozen --package C:\models\prototype\r-building
```

配置必须明确 `kind`（`r_building` 或 `r_card`）、与冻结快照一致的 `classes/client_versions/prototype_only`、`architecture=mobilenet_v3_small`、固定 `preprocess`、`applicability/output/event_rules/max_frame_age_seconds/code_version`；卡片状态使用单独配置与包。当前已有上方记录的真实图像、Agent 复核 R-card 原型及权重；val 事件漏检、类别和独立组不足，正式数据及发布仍未通过。`tests/test_dynamic_train_export.py` 的合成窗口只验证接口、checkpoint、CPU loader 和 ONNX parity。动态包仍默认禁用，之后还需充分的独立事件评估与实机响应证据。

U 位置检测独立于 R 状态分类，只接收已冻结的 `u_position` 窗口。每帧须明确 `boxes_complete=true`，注明框审核人、方法、证据，并将审核绑定到原图 SHA 和 ROI；空 `truth_boxes` 才是已复核的负例。英雄类用 `hero:<identity>`，主力兵群用 `troop_group`，框必须全部在原图 ROI 内。训练器逐个验证冻结窗口，将原图 ROI 拉伸为 640×640，精确转换原图坐标框到 YOLO 标签，保留 train/val/test 连通组隔离；测试 split 不参与调参。U 仅支持本地 `yolo26n.pt`，固定 RGB `/255`、AdamW、workers 0、batch 1–8，最多 100 epoch，有限时长和签名绑定的 `resume.pt`。导出选择 [YOLO26 one-to-one `nms=False` 输出](https://docs.ultralytics.com/guides/end2end-detection) `[1,300,6]`（ROI 像素 `xyxy,confidence,class_id`），经真实 `DynamicModel` CPU loader 与最多 100 张验证帧同输入 parity 核对。兼容性校验不代表定位准确率或战斗响应通过。

```powershell
python scripts/train_u_position.py --snapshot C:\data\u-frozen --config C:\data\u-config.json --output C:\runs\u-position --device 0 --pretrained C:\weights\yolo26n.pt --max-epochs 35 --deadline-hours 4
python scripts/export_u_position.py --training-dir C:\runs\u-position --snapshot C:\data\u-frozen --package C:\models\prototype\u-position
```

配置须声明 `kind=u_position`、冻结快照给出的有序 `classes/client_versions/prototype_only`、`architecture=yolo26n`、`code_version`、`batch`、`max_frame_age_seconds`、`event_rules` 与 `applicability`（`roi_kind=u_position`、分辨率范围、`requires_stable_camera=true`、明确 `requires_verified_zoom`）。`preprocess` 固定为 `{"width":640,"height":640,"color":"RGB","resize":"stretch","scale":0.00392156862745098,"mean":[0,0,0],"std":[1,1,1]}`；`output` 固定 `format=xyxy_score_class`、`max_detections=300`，另由验证集定稿 `min_confidence`。目前没有任何真实已复核 U 窗口或 U 权重；`tests/test_u_train_export.py` 的合成一轮 CPU 训练、导出与 loader 测试只证明工具链可运行，动态功能仍默认禁用。

## 免费导航场景候选采集

`scripts/collect_scene_samples.py` 用当前 18.647.2 客户端采集固定路线：两帧 HOME → 战斗菜单 → 军队页 → 两帧军队读数 → 返回菜单和 HOME → 部落聊天 → 返回并再拍两帧 HOME。仅六次导航输入，程序核对当前版本、前台、画面来源和唯一按钮；军队只在军队页比较，资源和宝石在前后 HOME 比较。任一状态、库存、停止请求或时间门不满足就保留失败原图并停止，不追加恢复操作。

从已推送且干净的工作区根目录运行，使用实机配置，模板目录须指向该工作区 `assets/templates`，游戏须已正常载入 HOME。`--output` 必须是仓库外的新目录，默认总预算 180 秒（允许 90–240 秒）；账号冲突或闲置重载应先另行处理。

```powershell
python -B scripts/collect_scene_samples.py --config C:\captures\live.toml --output C:\captures\scene-candidates\session-001 --split TRAIN --session-id session-001 --group-id candidate-group-001 --limit-sec 180
```

拍图前写入 `prospective.json` 并在输出父目录预留 session/group，拒绝同一父目录中的重复身份；结束写入原图和事件哈希及源码/配置指纹。`TRAIN`/`VAL` 仅是预声明候选归属，`labels=null`、`leakage_independence_verified=false`、`admitted_to_training=false`。先做真实图复核、版本核对和近重复/关联组审计，再由既有数据工具导入；不同采集会话不能自动算独立组，发现跨拆分关联时隔离，不能移动候选以改善分数。Agent 复核不会变成人工真值。

2026-10-17 的 `20261017-0927-train-d39dabf` 已实际完成 18 帧、6 次导航、55.764 秒，军队和前后 HOME 库存相同；审计 `reports/completion-20261015/diagnostics/scene-route-d39dabf-0927-root-v1.json`。原始候选保持未标注，运行时场景输出不能当作标签；后续 Agent 逐图复核、旧拆分保持和实际增量训练单独记录。

## 0927 场景原型增量快照与训练

Root 随后逐张查看上述 18 张原 PNG 并单独批准 Agent 视觉标签：8 张 `village`、4 张 `search`、4 张 `training` 纳入 TRAIN；第 14/15 张因聊天侧栏和部落战卡的遮挡状态不能仅凭静态图确定，保留原图并隔离、标签为 null。旧 TRAIN 村庄参考和军队页近似图也经原图复核，两个旧 TRAIN 组保守关联为同一泄漏组；不宣称相同对手或配兵。首次构建在两对跨组近重复检查处拒绝，失败目录保留；取得独立来源绑定的关联收据后才重新构建。没有调用自动 `split`，旧 train/val/test 的 63 条分配及其 63 个图像、63 个注释文件字节不变。

新快照 `E:/projects/AutoCOC-training/datasets/current-scene-prototype-20261017-0927-v1` 为 **59 train / 9 val / 11 test**，其中新增 16 张只进 train、2 张隔离；SHA256SUMS 的 SHA-256 为 `bd0219f64eb20d75ea5ce2ec3b2286bf8ce724d0f12895ebe0d54e4c83a27f0e`。在已推送且干净的 `55ee3dc335920ab1e76d2db573d863e80691fe2c` 源码上，用本地权重、seed 42、batch 8、MobileNetV3-Small、320×192 和固定 recipe3 最多 20 epoch 新训，实际完成 16 epoch，最佳为 epoch 8，validation NLL `0.1608283817`。导出的 ONNX 在全部 9 张 VAL 图上同输入 parity 9/9，最大绝对误差 `5.245e-6`。

固定运行阈值下 VAL 接受 **5/9**、拒绝 **4/9**，相同旧 VAL 上旧包为 2/9；这只说明此原型在该验证集上的覆盖变化。VAL 仍仅含 `battle`、`enemy_village`、`settlement` 三类，均来自一个旧来源组，新增的 `village/search/training` 没有 VAL 覆盖。因此质量准入未通过，包保持 `prototype_only`、`runtime_enabled=false`，没有实机模型验收。根审计见 `reports/completion-20261015/diagnostics/s-scene-incremental-0927-root-v1.json`（SHA-256 `ef1e2f373caf6cc8506ea9d351c0b645b6e1fcc0a4df4b93e0f559a96143650b`），构建/训练原始回执在 `reports/completion-20261015/tools/scene_incremental_0927_v1/runs/`。

旧 TEST 只用于标准完整性哈希、schema 和近重复检查，没有作为模型输入、推理、评估或选模依据；但准备脚本时曾误打印 `CS-174504-00001` 的一条旧 TEST manifest/annotation 记录，故**不能称为零 TEST 语义接触**。没有查看 TEST 像素，也没有据此改标签、拆分或阈值；该事件保留在外部执行报告中。上述复核均为 `agent_visual`，不冒充真人逐图确认或正式数据。
