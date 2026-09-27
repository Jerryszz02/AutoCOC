# 战前视觉规划与连续投放

本次交付程序实现及离线验证：本地 ONNX 推理、核心区聚合、准备期限、候选选择、冻结计划、连续输入、投放后核验、结算回村、GUI 配置与报告。**不包含模型训练、素材采集/整理/标注、权重、配兵与阵型实战验收。** 战中增强模式保留计时接口并拒绝启用；没有宣称达到实机 0.5 秒。

## 配置与入口

旧配置默认 `vision_agent.enabled = false`，保持已有打法。GUI 的「配置 → 战前视觉规划」可保存同样的字段，打开仍默认离线预演。路径相对于主配置文件；GUI 方案保存在本机 `.desktop.json`，不改写原 TOML。

```toml
[runtime]
png_compression_level = 0

[vision_agent]
enabled = false
model_dir = "models/current"
layout_profile = "profiles/accepted-layout.json"
preparation_reserve_sec = 5.0
mode = "continuous"
evidence_limit_mb = 256
jev_enabled = false
jev_model = "jev-1.13.0"
jev_timeout_sec = 2.0
guide_file = ""
```

这是配置格式示例，不是已经验收的模型或布局。启用前需要本地模型与对应的布局验收文件；程序在搜敌前加载/检查模型，缺失时报告不可用。模型目录、ONNX 输出格式见[模型包接口](MODEL_PACKAGE.md)，配兵与布局要求见[运行时契约](VISION_AGENT_RUNTIME.md)。

```powershell
python -m autococ.gui --config config.toml
python -m autococ.cli run --config config.toml --profile battle-only --once --dry-run
```

离线预演检查已有任务编排，不产生真实识别或投放证据。新规划链路的模拟测试见 `tests/test_prepared_battle.py` 及 `tests/test_combat.py`，它们使用构造的当前帧状态和模拟输入，不代表模型精度或实际战果。

## 准备与投放

准备倒计时只接受当前敌方预览中顶部标签及唯一相邻数字，未知或冲突时不能假设仍有 30 秒。截止点使用取帧开始的单调时间减去已花时间及预留余量。模型、攻略预先加载，准备阶段只消费当前帧兵栏、合法边界、核心建筑及已经验收的配兵规则。

冻结计划将输入序列、数量、卡位、模型/布局哈希、显示器和时限绑定。执行期间不截图、OCR、检测或等待 Jev；必要选卡和落点均计入输入数 `N`，整批时间上限是 `min(15, 1 + 0.25 * N)` 秒。超时、停止或传输不确定后，剩余队列终止，已尝试输入不重放。失败也保留回执。

投放之后再读取新帧核验消耗，接着复用原结算和回村流程。GUI/报告分别标明「计划就绪」「输入已发出」「消费已核验」「战斗完成」。传输返回成功、全部点击完成、消费核验、战果是不同证据。

PNG 0 仅控制 MuMu 或 ADB raw 的本地编码；Android 自带 PNG 编码不受此参数控制。原生 SDK 仍隔离在可超时结束的工作进程，输入异常不切换通道重放。新模式对本次 `frames` 目录设置容量上限，并在投放前为结果帧检查余量。等待结算等轮询只循环保留最新中间帧，结束或停止时将最后一帧转为保留证据；只回收本次等待过程登记的临时帧，已保留的截图不删除。额度不足停止，一张新捕获的暂存帧可能占用额外空间，超额帧不会发布。事件日志标明临时帧及最终保留帧。暂未引入共享内存；需完整链路基准证明确有必要后再做。

## Jev 与本地候选

Jev 使用固定版本的 Choice HTTP 协议，只选择程序已经校验过的候选 ID。请求只包含本局必要的结构化状态和匹配的短攻略，不上传截图。凭据从 `TYPESAFE_API_KEY` 环境变量读取；配置样例、日志和报告不保存密钥。

`jev_enabled = false` 时没有外部调用；启用但未配置凭据会报告 `jev_credentials_unavailable` 并用首个已校验的本地候选。网络错误、低置信、错误候选或模型版本、非法概率、超时都留下相应原因。一次准备最多发一次请求，不自动重试。超时结果留在请求私有队列，不能改写已冻结计划；总准备期限已过时也不使用本地回退开始进攻。

可选攻略文件为 JSON 数组，每项格式如下；这只是接口示例，未提供或制作训练素材：

```json
[
  {
    "id": "reviewed-rule-id",
    "units": ["已验收兵种ID"],
    "reason": "适用局面与选择理由",
    "order": "动作顺序依据",
    "failure_conditions": "不适用条件"
  }
]
```

只有所列兵种本场全部存在的条目才参与决策，最多传入 8 条。攻略文本不能替代代码中的数量、坐标及期限校验。协议依据：[TypeSafe API](https://docs.typesafe.ai/api)、[固定模型版本与输入类型](https://docs.typesafe.ai/models)。本次未配置凭据或调用真实 Jev 服务。

## 验证与尚待完成的验收

本地单测覆盖截止时间、零观察投放、重复执行阻止、异常/停止、卡位和配兵限制、模型元信息/坐标变换/拒识、有限 Jev 响应、证据额度及旧配置兼容，实际结果见[验证记录](VISION_AGENT_VALIDATION.md)。`vision_timing.ActionTiming` 保存从截图请求到完整动作及效果核验的阶段时间，报告按动作类型和冷/热启动统计，失败和未知不从尝试分母剔除。

独立模型评估入口见[模型包接口](MODEL_PACKAGE.md)。未提供权重时，只能运行模拟契约测试；训练、独立数据精度、Windows 真实推理性能、Jev 连通、稳定兵栏和完整战斗都仍待后续验收。`models/`、`datasets/` 及权重文件已忽略，不加入源码仓库。
