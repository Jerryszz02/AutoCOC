# 战前连续投放运行契约

在 `config.toml` 的 `[vision_agent]` 将 `enabled` 设为 `true`、`mode` 设为 `"continuous"`，配置本地 `model_dir` 和 `layout_profile`。`model_dir` 是模型包根目录，下设 `scene/` 和 `building/` 两个各自带清单与权重的目录。两个模型都必须记录 `validation.status="evaluated"` 和非空 `report_id`；建筑模型的报告 ID 还要与布局档案一致。战斗任务沿用原有搜敌、资源限制、结算与回村流程；搜敌前加载并预热模型与攻略，敌方预览中读取真实倒计时、卡位、核心与合法边界，冻结后一次性发送完整输入序列。模型包缺失或未通过校验时，新模式明确不可用，旧策略不会自动改用它。

布局档案是针对已核验客户端、显示器、传输、配兵和固定卡位的验收记录。以下是**格式示意**，不是可直接投入实机的验收档案：

```json
{
  "accepted": false,
  "cards_stable_after_deployment": false,
  "baseline_resolution": [1280, 720],
  "client_version": "填写实测版本",
  "transport": "mumu_native",
  "display_id": 2,
  "zoom": "minimum_stable",
  "model_sha256": "填写已验收建筑模型哈希",
  "model_validation_report_id": "填写独立评估报告ID",
  "cards": [
    {"kind": "troop", "unit_id": "填写单位ID", "source": "army", "count": 1, "point": [110, 630]}
  ],
  "candidate_orders": [["填写单位ID"]],
  "funnel_units": [],
  "excluded_kinds": ["hero", "siege"],
  "core_classes": {"填写核心建筑ID": 0.9},
  "min_core_targets": 1,
  "core_min_confidence": 0.9,
  "spell_targets": {}
}
```

`cards` 必须逐项等于本模式使用的当前兵栏身份、来源、数量和中心点；`candidate_orders` 每条必须恰好包含全部卡片，每个单位出现一次，最多四条顺序。每条顺序与核心位置决定的两个已检测边界入口组合成有限候选。`funnel_units` 可指定整张兵卡作为清边队，生成清边和主力从不同入口投放的候选；不会把一张卡拆成未核验的数量。法术仅支持在 `spell_targets` 中为该单位明确写 `"core"` 的固定核心落点；需要等待战中状态的法术/技能不属于此模式。当前生产识别的英雄卡通常没有可靠数值，只有在档案显式列入 `excluded_kinds` 时才排除英雄或攻城器，且不报告它们已投放。兵种/法术未知、滚动、来源混淆、未核验数量或布局变化会拒绝计划。部落援军卡当前拒绝，不根据容量推断是否已收到。

多核心候选默认关闭：省略 `multi_core` 或写 `{"enabled":false}` 时，仍使用上述单核心聚合、候选 ID 和选择规则。只有单独验收的布局档案才能显式加入下列**示意字段**；现行可运行档案未启用此模式：

```json
"multi_core": {
  "enabled": true,
  "link_distance_pixels": 60,
  "separation_margin_pixels": 20,
  "min_targets_per_core": 1,
  "max_cores": 3,
  "max_selected_detections": 12,
  "max_candidates": 12
}
```

启用时全部七个字段必须显式填写、不得附加其他字段：连边距离为 20–400 基准像素、组间余量为 1–100、每组最低可见目标数为 1–4、组数上限为 2–3、选中检测数上限为 2–24、候选总数上限为 1–24；布尔值、非有限数、越界值都拒绝。`core_classes` 同时必须给出有限且大于 0 的类别阈值。输入检测总数最多 300，只使用同一当前帧、同一建筑模型、达到类别阈值且框完全位于画面内的可见目标；任一传入检测的帧、模型或框无效就拒绝。距离内的目标先形成连通分量，但组内任意两目标也必须保持在连边距离内，避免桥接串链；不同组最近距离必须超过连边距离与余量之和，边界或灰区拒绝。启用后只见 1 组也拒绝，不暗中退回单核；漏检绝不说明建筑已摧毁或防御空缺。

每组按中心从左到右稳定编号，结合两个已核验入口、整卡清边选项和 `candidate_orders` 生成完整候选，超出档案候选上限就整次拒绝。候选 `core_index`、`core_center`、`core_bbox`、入口、清边入口和顺序固定；法术的 `"core"` 落点取该候选所选组中心。本地兜底按入口到组中心的距离选择，等距时用稳定编号和顺序打破平局；Jev 仍只能选择已编译的候选 ID。全部组的几何、来源检测和 `selected_core_index` 写入 `FrozenBattlePlan.core_geometries` / `selected_core_index`、事件及计划 SHA，执行前篡改会拒绝。此实现目前只有合成离线测试，仍需模型发布、对应布局档案验收和多核心实战核验，不能视为通用多核心战术已可用。

准备阶段使用取帧开始时间和本帧倒计时计算截止点，并留出 `preparation_reserve_sec`。连续阶段单次许可绑定当前帧、显示器、档案和完整计划哈希，只可尝试一次；输入数 `N` 包含选卡与每次落点，整批限时 `min(15, 1 + 0.25N)` 秒。投放窗口内只发送输入并检查停止、期限和传输，不截图或推理。任何超时或不确定输入停止剩余队列，随后用新画面核验消耗；无法确认的单位保持未确认，结算与回村另记证据。`evidence_limit_mb` 限制本次帧目录，额度不足时在开打前拒绝，不删除旧证据。

增强模式保留配置入口但未通过按动作的正确性和完整 0.5 秒延迟验收，运行时明确拒绝。此代码的模拟测试只证明软件流程；具体配兵、布局、模型精度和实机投放速度仍需完成验收后才可把档案标为 `accepted`。
