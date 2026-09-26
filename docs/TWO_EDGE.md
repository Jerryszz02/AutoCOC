# 两边一字划：实现与实测

日期：2026-09-25。目标是先把一字划、等待结算和回村跑通，不以胜率、英雄技能或收益最大化作为本轮目标。

## 实施计划与最终行为

1. 分开测截图与识别：增加会话耗时日志和只读基准脚本。
2. 简化部署：对战开始后双指缩到最小，再做一次缩小并比较地图特征，确认比例/位置稳定；固定两条边，不平移寻找边界。
3. 按兵栏实际 `xN` 分配，每个兵种尽量平分到两边，单数余数放第一边。40 个兵对应 20+20。每张卡只选一次，连续发送沿线触点，放完后核对耗尽；延迟读数只补拍，不重放落兵。
4. 英雄、攻城器和法术不参与本模式。战斗结束后确认结算画面并回村，输掉战斗也可算流程成功；不宣称未核对的库存收益。
5. 配置、CLI、GUI 共用同一策略；停止信号在后续每次输入前检查。等待结算约每 5 秒发起一次新观察，跳过无用的部队/英雄详细识别。

## 实测结果

| 项目 | 结果与证据 |
| --- | --- |
| 初始三帧基准中位数 | 截图 0.443 秒，识别 4.727 秒，整次观察 5.170 秒；[原始结果](../reports/benchmark-20260925-154721-861217/summary.json) |
| 主要瓶颈 | 识别及重复观察，比持续截图本身更耗时。后续热运行的识别约 2～3 秒，不能把不同场景/系统负载间的差异全归因于代码优化。 |
| 完整实战 | [运行报告](../reports/run-20260925-161014-749222-452411e4.md)：战斗任务 238.54 秒，12/12 个兵投放核验，40% / 0 星，自然结束并自动回村。 |
| 雷龙投放 | 10 条，固定两边各 5 条；连续点击含间隔约 1.82 秒，数量由 x10 核对至 x0。 |
| 全部投放 | 10 雷龙 + 1 超级炸弹人 + 1 法师；选卡、截图、数量核验合计 26.23 秒，不含前面的缩放。 |
| 缩放与落点 | [固定视角](../reports/20260925-161014-749222-452411e4/frames/00013-line-zoom-verified.png)、[雷龙已耗尽](../reports/20260925-161014-749222-452411e4/frames/00015-line-consumption.png)。 |
| 结果与回村 | [结算](../reports/20260925-161014-749222-452411e4/frames/00043-battle-settlement.png)、[回村](../reports/20260925-161014-749222-452411e4/frames/00045-battle-return-home.png)。 |
| 活动兵 | 第一场调试画面额外出现灰色 x40 卡，不属于军队编辑页的配兵；现已支持该头像与数量的独立识别。两次定位调试未投兵且自然结束，不计成功；最终完整实战没有这张活动卡。40 兵投放仍只有回放/单元测试，未实机验证。 |

单场成功不能证明持续运行稳定性，也不能证明胜率改善。固定落点按本机中文常规战、1280×720 基准和已验证的最小缩放标定；不同场景/布局需要再验证。原资源模式和历史失败报告不改写。

最终全量离线回归：**577 passed, 423 subtests passed**（75.52 秒）。覆盖 40 兵 20+20 分配、连续投放、延迟读数不重投、部分投放失败保留已核验数量、停止传播、双指异常释放、活动灰卡及数量缺失、败局/满仓不误判流程失败、原资源模式回归。额外检查通过：CLI 两任务离线预演；GUI 本机策略为 `two_edge`，1060×870 最小窗口中最后一个控件底部为 737，显示完整，打开界面未连接设备或启动任务。

## 使用与复现

GUI：选择“仅对战”→“两边一字划（刷场数）”→“实机执行”。基础 TOML 需要已有有效的 `[mumu]` 配置。界面仍默认离线，打开时不自动运行。

```powershell
.\.venv\Scripts\python.exe -m autococ.cli run --config config.toml --profile battle-only --strategy two_edge --once
.\.venv\Scripts\python.exe scripts/benchmark_observation.py --config config.toml --frames 3
.\.venv\Scripts\python.exe -m pytest -q
```

代码保留 `verified` 原策略；已有未设置 strategy 的配置仍采用原策略。本机 `config.toml` 已设为 `two_edge`，GUI 的保存方案可以另行覆盖。

## 改动文件

- 运行逻辑：`src/autococ/deployment.py`、`combat.py`、`battle.py`、`mumu.py`、`session.py`、`vision.py`。
- 配置与入口：`src/autococ/config.py`、`cli.py`、`desktop.py`、`gui.py`、`config.example.toml`；本机不入库的 `config.toml`。
- 测量：`scripts/benchmark_observation.py`。
- 识别素材：`assets/templates/battle_event_super_pekka.png`、`assets/templates/README.md`、`tests/fixtures/event-battle-bar.png` 和 `.json`。
- 回归测试：`tests/test_deployment.py`、`test_combat.py`、`test_config.py`、`test_desktop.py`、`test_event_troops.py`、`test_gui.py`、`test_mumu.py`、`test_session.py`。
- 使用说明：`README.md`、`docs/DESKTOP.md`、本文。
