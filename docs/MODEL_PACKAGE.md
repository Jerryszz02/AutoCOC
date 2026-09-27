# 本地视觉模型包接口

代码已支持场景分类和建筑检测的 ONNX Runtime CPU 推理。仓库不带权重；缺少模型包、哈希不符、游戏版本不符或输入帧过期时，能力明确为不可用。模型推理可供战前准备使用；真正允许其控制实战部署，还须独立数据验收和首套配兵的布局验收。

统一模型包根目录下按用途分开存放，避免两种模型争用同一个 `manifest.json`。运行时的 `model_dir` 指向包根目录；独立评估 CLI 则接收单个模型子目录。

```text
model_dir/
  scene/
    manifest.json
    model.onnx
  building/
    manifest.json
    model.onnx
```

每个子目录的元信息包含 `manifest.json` 和其引用的 `.onnx` 文件。建筑模型示例：

```json
{
  "schema_version": 1,
  "kind": "building",
  "client_versions": ["已验证的游戏版本"],
  "code_version": "训练和导出代码版本",
  "data_version": "审核数据版本",
  "validation": {"status": "unvalidated"},
  "applicability": {
    "screen_scope": "battlefield",
    "source_width_range": [1280, 1280],
    "source_height_range": [720, 720],
    "zoom": "minimum"
  },
  "classes": ["town_hall", "air_defense"],
  "model_file": "model.onnx",
  "sha256": "权重文件的64位SHA256十六进制值",
  "max_frame_age_seconds": 2.0,
  "preprocess": {
    "width": 640,
    "height": 640,
    "color": "RGB",
    "resize": "letterbox",
    "scale": 0.00392156862745098,
    "mean": [0, 0, 0],
    "std": [1, 1, 1],
    "pad_value": 114
  },
  "output": {
    "format": "yolo_xywh_classes",
    "coordinates": "input_pixels",
    "min_confidence": 0.7,
    "class_min_confidence": {"town_hall": 0.8},
    "nms_iou": 0.5,
    "max_candidates": 1000,
    "max_detections": 300
  }
}
```

上述类别和阈值仅展示格式，尚非游戏标签或准入门槛的定稿。场景模型的 `screen_scope` 应为 `full_ui`，建筑模型应为 `battlefield`；实际分辨率必须处于声明范围内。`validation.status` 必须显式设为 `unvalidated` 或 `evaluated`，后者还需要 `report_id`；这只是可追溯记录，不会让模型自动通过实战准入。`preprocess.resize` 可为 `letterbox` 或 `stretch`；颜色可为 `RGB` 或 `BGR`，均从 OpenCV 读取的 BGR 原图开始转换。场景分类的 `output` 用 `{"format":"logits","min_confidence":0.8,"min_margin":0.15}`，也支持 `probabilities`。场景输出形状为 `[1, 类别数]`。建筑输出可为标准 YOLO 原始 `[1, 4+类别数, 目标数]`（中心 x/y、宽/高、各类置信度），或经后处理的 `[1, 目标数, 6]`，格式名分别是 `yolo_xywh_classes`、`xyxy_score_class`。后一种每行为 `x0,y0,x1,y1,score,class_id`。两种坐标均为预处理后输入图像的像素坐标，程序将补边和缩放还原到原截图坐标，随后按类别做 NMS。`max_candidates` 和 `max_detections` 有代码硬上限，防止异常模型输出导致无界后处理。

调用方必须传入**取帧开始时的单调时钟时间** `captured_at`，不可在推理时重置时间。`SceneModel(...).classify(image, captured_at=...)` 返回场景、置信度和拒识原因；`BuildingModel(...).detect(image, captured_at=...)` 返回可见建筑框。若要聚合核心区，向 `infer_core_geometry` 提供经审定 profile 的 `core_classes`（类别到最低置信度的映射）、`min_core_targets` 和可选的 `core_padding_pixels`；没有这些条件时核心区为未知。输出中心与范围位于调用方指定的基准坐标系。建筑漏检不会被解释为摧毁。

`ScreenshotRecognizer` 可选择注入两个模型，并在 `recognize(..., captured_at=...)` 中使用。场景 OCR 与模型给出两个互相冲突的已知类别时，结果为未知；建筑模型结果与现有模板跟踪结果分别保存，避免把可见目标混成已核验的摧毁状态。

已有人工复核的 JSONL 可用以下命令离线评估；省略 `--reviewed-jsonl` 时只校验包和加载权重：

```powershell
python scripts/evaluate_model_package.py C:\models\AutoCOC\scene --kind scene --client-version 已验证的游戏版本 --reviewed-jsonl C:\reviewed\scenes.jsonl
python scripts/evaluate_model_package.py C:\models\AutoCOC\building --kind building --client-version 已验证的游戏版本 --reviewed-jsonl C:\reviewed\buildings.jsonl
```

场景行格式为 `{"image":"frame.png","label":"enemy_village"}`；建筑行格式为 `{"image":"frame.png","boxes":[{"class":"town_hall","bbox":[100,120,180,200]}]}`。图片路径相对于 JSONL 文件目录。工具报告场景每类 precision/recall/F1、混淆矩阵、覆盖率、拒识率、未知误接受数；建筑每类 precision/recall、AP50、中心和框角平均定位误差，以及 mAP50。有效真值遇到读图或推理失败时，场景结果记 `inference_error`、建筑计入对应类别 FN 与 AP50 真值分母；无效标签只报告错误，不凭空增加类别。覆盖率以全部尝试为分母，并另报有效标注样本分母。

每条评估尝试还记录首张有效标注图片的 `cold` / 后续 `warm`、设备、执行提供器、模型输入尺寸、成功或失败及耗时；无效标签只有 `preflight`。汇总包含 P50/P95/最大值与失败次数。模型加载单独报告 `model_load_seconds`；`cold` 指模型已加载后的首张离线图片。`offline_elapsed_seconds` 从本地文件读取前开始，包含离线解码、模型推理和指标匹配；`inference_elapsed_seconds` 只覆盖模型调用。离线帧时效也从读取文件前计时，不在解码结束后重置。**这些数值不包含实时取帧、决策和游戏输入，不能作为战中 0.5 秒目标验收。**加载成功只报告 `package_loadable`，不会写成模型已验收。模型训练、标注整理、阈值定稿和真实权重验收仍待后续工作。
