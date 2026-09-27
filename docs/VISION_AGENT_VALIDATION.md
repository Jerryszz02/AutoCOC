# 非训练程序验证记录

日期：2026-09-27。基础版本：`2ea13c118c1a5aa855be97b8b1fe9bb0c26e26e6`。实现分支：`agent/vision-agent`。

## 已执行

- 全量回归：`python -m pytest -q --tb=short --maxfail=5`，918 passed、129 skipped、495 subtests passed，95.00 秒。
- 最后补改后的运行时、停止回执与证据留存回归：`python -m pytest tests/test_prepared_battle.py tests/test_combat.py tests/test_session.py tests/test_flow.py tests/test_daily.py tests/test_reporting.py tests/test_evidence.py -q --tb=short -rs`，215 passed、1 skipped、100 subtests passed，4.21 秒。该组跳过项为独立工作树中未提供的可选 delayed-search 截图。
- `git diff --check` 通过。

在 Windows 独立工作树中使用现有 `.venv`，设置 `PYTHONPATH` 指向该工作树的 `src`。未改动主目录配置和历史截图；测试期间使用本任务独立的临时电源请求。

## 测试覆盖边界

模拟集成经过现有 `run_battle`：模型预载、敌方预览、冻结计划、连续输入、投放后消费核验、结算与回村。覆盖期限耗尽、无效模型及布局、当前帧倒计时、配兵和卡位不符、单次计划许可、输入超时/不确定不重放、停止后部分回执、模型坐标还原、Jev 响应校验与本地回退、PNG 像素一致性、证据容量及旧配置兼容。

模型测试注入固定推理输出；模拟游戏输入和构造观测不证明真实模型精度、卡位稳定性或战果。跳过的历史图片测试不计为通过。GUI 配置及持久化有程序测试，本次未进行人工视觉验收。

## 未执行及交付边界

- 按用户范围，不采集、整理或标注训练素材，不编写训练/导出工具，不提供模型权重。
- 未调用真实 Jev 服务，未开启模拟器实机对战，未改变本机生效配置以启用新模式。
- ONNX 加载和评估工具已实现；真实权重精度、Windows 推理性能、完整投放速度、布局/配兵验收及战果仍需后续验证。
- 新功能默认关闭；启用后要求模型包及对应已验收布局。战中增强仅交付计时接口，执行模式保持禁用，不宣称满足 0.5 秒目标。

配置与操作见 [VISION_AGENT.md](VISION_AGENT.md)，模型接口见 [MODEL_PACKAGE.md](MODEL_PACKAGE.md)，运行约束见 [VISION_AGENT_RUNTIME.md](VISION_AGENT_RUNTIME.md)。
