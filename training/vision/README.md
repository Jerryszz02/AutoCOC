# 本地视觉训练工具

S/R 训练配方 `trainer_recipe_version=2` 在头部训练时同时冻结主干参数和 BatchNorm 运行统计；进入微调时重置早停计数，保留全局最佳权重，连续 7 次微调验证无改善才早停。配方写入运行配置和恢复签名，旧配方 checkpoint 不能混用 resume；旧运行和导出产物保留用于对照。针对性验证为 17 passed / 1 skipped（未启用本次不涉及的 B 训练集成），包含真实张量训练、阶段边界恢复和 S/R 合成训练导出；这些结果只证明程序行为。

训练环境与日常助手 `.venv` 分开。先在隔离环境中安装匹配的 PyTorch/torchvision CUDA wheel，再安装 `requirements-training.txt` 的其余顶层依赖；程序不会下载预训练权重。当前 Windows/Python 3.13.12/RTX 3070 Ti 实测 torch 2.13.0+cu126 与 torchvision 0.28.0+cu126，完整已安装版本写入每次运行的 `environment.json`。其他平台需重新核对兼容性与冒烟，不把这份记录当成通用锁定环境。

Ultralytics 的设置在每次运行目录隔离；首次导入前即写入 `sync=false`、关闭集成日志，设置 `YOLO_OFFLINE`，并在本训练进程中阻断 socket 出站连接。离线依赖缺失时运行会失败，需先由操作者准备，不会边运行边安装。

输入必须是 `scripts/vision_data.py export` 产生的完整复核快照，带 `SHA256SUMS`、`train.txt`、`val.txt`、`test.txt`、`split.json` 和标签文件。训练和导出会核对全部文件哈希。32 图 2 epoch 冒烟用 `--prototype --max-epochs 2`。原型完成冒烟后，可用 `--prototype --development --max-epochs 20` 在同一冻结划分上使用全部训练及验证图探索；场景最多 35 epoch，建筑最多 100 epoch 且 batch 不超过 8，仍不可把测试集用于调参。建筑训练把原图和标签复制到运行目录，避免 Ultralytics 缓存写入冻结快照。原型数据和模型始终标 `prototype_only`，不能充当正式验收。

场景人工复核使用 `scripts/review_vision_scenes.py`。源目录须已有通过校验的 `manifest.jsonl`、`annotations.jsonl` 和原图；日志必须放在源目录外。启动后在本机浏览器打开终端给出的地址，逐张查看原图，由复核者填写自己的名字并明确确认建议、改标签或跳过。方向键只翻图，不会提交；跳过也可撤回该图之前的确认。工具不会替用户确认，也不会覆盖原有 Agent 标注。

```powershell
python scripts/review_vision_scenes.py serve --root C:\data\scene-source --journal C:\reviews\scene-verdicts.jsonl
python scripts/review_vision_scenes.py export --root C:\data\scene-source --journal C:\reviews\scene-verdicts.jsonl --output C:\data\scene-human-reviewed
```

服务仅监听 `127.0.0.1`，页面地址仅供本机复核。日志追加保存每次明确操作，绑定原图、来源清单和标注哈希；来源变动会拒绝提交或导出。导出只包含最新操作为确认或改标签的图，保留原始来源与全部复核历史；目标须是新目录，全部写入和校验成功后才发布。零确认时拒绝导出。导出结果还须经过既有分组、覆盖检查、`split` 和 `export` 才成为训练快照；它不自动满足正式数量门槛，也不会把原型模型改为正式模型。

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

R 的建筑状态和技能卡状态使用独立的四类空间与独立模型包，不能把 `truth_event` 当逐帧状态标签。`training.vision.dynamic_snapshot.freeze_dynamic_snapshot` 先把审核过的动态窗口冻结为带 `windows.jsonl`、`snapshot.json`、`split.json` 和 `SHA256SUMS` 的不可覆盖目录；训练只消费 `verify_dynamic_snapshot` 全面复核后的副本。每帧必须有 `state_label`、审核人与证据、目标身份、源图哈希和原图 ROI；同一事件/会话/阵型连通组不可跨 train/val/test。R 训练用 224×224 原图 ROI、ImageNet 预训练 MobileNetV3-Small、按类与事件连通组采样，最多 35 epoch、5 epoch 头部训练和 7 epoch 验证停滞早停，签名绑定快照、拆分、权重、种子与配置，可用同一运行目录 `last.pt` 恢复。CPU FP32 ONNX 导出通过真实 `DynamicModel` 加载与 ROI 推理，并比较最多 100 张验证帧的同输入 logits；报告仅证明导出兼容。

`r_card` 只训练英雄技能卡，逐帧 `target_id` 必须为 `hero_skill:<已登记英雄ID>`，不能使用普通兵、攻城或未明确身份的卡。四类 `available/selected/exhausted/unknown` 仍是单帧状态；复合 `skill_available` 另要求连续两帧在各自**同一原图**上确认该英雄已部署。每个 R-card 标注帧须有单独复核的 `hero_deployment_truth`：`hero_id`、`card_identity_verified=true`（已确认原图 ROI 是该英雄的己方技能卡）、布尔 `deployed`、审核人/方法/证据、原图 SHA、ROI、frame/session/camera/capture/client 绑定；负例可明确标 `deployed=false`，正事件至少两帧 `available` 且 `deployed=true`。离线评估不会把此真值当模型读数，而是从哈希核对过的原图调用生产英雄卡头像与封闭 HP 识别；读不到、身份或版本不符、低置信、过期或未部署均打断事件连续性。连续证据不能跨客户端版本、模型权重或推理合同拼接。没有可独立确认的正例时技能事件指标为 `not_established`。合成测试只验证工具链；目前真实技能正例为零。

```powershell
python scripts/train_dynamic_state.py --snapshot C:\data\r-building-frozen --config C:\data\r-building-config.json --output C:\runs\r-building --device 0 --pretrained C:\weights\mobilenet_v3_small-047dcff4.pth --max-epochs 35 --deadline-hours 4
python scripts/export_dynamic_state.py --training-dir C:\runs\r-building --snapshot C:\data\r-building-frozen --package C:\models\prototype\r-building
```

配置必须明确 `kind`（`r_building` 或 `r_card`）、与冻结快照一致的 `classes/client_versions/prototype_only`、`architecture=mobilenet_v3_small`、固定 `preprocess`、`applicability/output/event_rules/max_frame_age_seconds/code_version`；卡片状态使用单独配置与包。当前尚无已复核的真实 R 逐帧状态窗口，所以**没有真实 R 训练权重或事件准确率**；`tests/test_dynamic_train_export.py` 的合成窗口只验证接口、checkpoint、CPU loader 和 ONNX parity。动态包仍默认禁用，之后还需独立事件评估与实机响应证据。

U 位置检测独立于 R 状态分类，只接收已冻结的 `u_position` 窗口。每帧须明确 `boxes_complete=true`，注明框审核人、方法、证据，并将审核绑定到原图 SHA 和 ROI；空 `truth_boxes` 才是已复核的负例。英雄类用 `hero:<identity>`，主力兵群用 `troop_group`，框必须全部在原图 ROI 内。训练器逐个验证冻结窗口，将原图 ROI 拉伸为 640×640，精确转换原图坐标框到 YOLO 标签，保留 train/val/test 连通组隔离；测试 split 不参与调参。U 仅支持本地 `yolo26n.pt`，固定 RGB `/255`、AdamW、workers 0、batch 1–8，最多 100 epoch，有限时长和签名绑定的 `resume.pt`。导出选择 [YOLO26 one-to-one `nms=False` 输出](https://docs.ultralytics.com/guides/end2end-detection) `[1,300,6]`（ROI 像素 `xyxy,confidence,class_id`），经真实 `DynamicModel` CPU loader 与最多 100 张验证帧同输入 parity 核对。兼容性校验不代表定位准确率或战斗响应通过。

```powershell
python scripts/train_u_position.py --snapshot C:\data\u-frozen --config C:\data\u-config.json --output C:\runs\u-position --device 0 --pretrained C:\weights\yolo26n.pt --max-epochs 35 --deadline-hours 4
python scripts/export_u_position.py --training-dir C:\runs\u-position --snapshot C:\data\u-frozen --package C:\models\prototype\u-position
```

配置须声明 `kind=u_position`、冻结快照给出的有序 `classes/client_versions/prototype_only`、`architecture=yolo26n`、`code_version`、`batch`、`max_frame_age_seconds`、`event_rules` 与 `applicability`（`roi_kind=u_position`、分辨率范围、`requires_stable_camera=true`、明确 `requires_verified_zoom`）。`preprocess` 固定为 `{"width":640,"height":640,"color":"RGB","resize":"stretch","scale":0.00392156862745098,"mean":[0,0,0],"std":[1,1,1]}`；`output` 固定 `format=xyxy_score_class`、`max_detections=300`，另由验证集定稿 `min_confidence`。目前没有任何真实已复核 U 窗口或 U 权重；`tests/test_u_train_export.py` 的合成一轮 CPU 训练、导出与 loader 测试只证明工具链可运行，动态功能仍默认禁用。
