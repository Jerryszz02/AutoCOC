# 本地视觉训练工具

训练环境与日常助手 `.venv` 分开。先在隔离环境中安装相互兼容的 PyTorch、torchvision、Ultralytics、ONNX、ONNX Runtime、OpenCV、NumPy；程序不会下载预训练权重。实际已安装版本会写入每次运行的 `environment.json`，不要把未经冒烟的版本表当成锁定环境。

Ultralytics 的设置在每次运行目录隔离；首次导入前即写入 `sync=false`、关闭集成日志，设置 `YOLO_OFFLINE`，并在本训练进程中阻断 socket 出站连接。离线依赖缺失时运行会失败，需先由操作者准备，不会边运行边安装。

输入必须是 `scripts/vision_data.py export` 产生的完整复核快照，带 `SHA256SUMS`、`train.txt`、`val.txt`、`test.txt`、`split.json` 和标签文件。训练和导出会核对全部文件哈希。32 图 2 epoch 冒烟用 `--prototype --max-epochs 2`。原型完成冒烟后，可用 `--prototype --development --max-epochs 20` 在同一冻结划分上使用全部训练及验证图探索；场景最多 35 epoch，建筑最多 100 epoch 且 batch 不超过 8，仍不可把测试集用于调参。建筑训练把原图和标签复制到运行目录，避免 Ultralytics 缓存写入冻结快照。原型数据和模型始终标 `prototype_only`，不能充当正式验收。

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
