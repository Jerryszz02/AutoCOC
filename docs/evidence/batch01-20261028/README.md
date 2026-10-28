# 第 0–1 批证据索引

实现范围和预算见[试点记录](../../planning/PILOT_BATCH_0_1.md)。客户端升级经用户明确授权，现为 **18.647.5 / 180647006**。旧版基线失败、新版校准、自动路线结果和历史试标分开记录。

## 代码交付

- [`capture_pilot_baseline.py`](../../../scripts/capture_pilot_baseline.py)：固定设备/精确版本检查，只允许四次免费导航，记录输入尝试、原图、源码/配置/素材前后哈希，双帧完整军队及回村资源核对；停止/不确定输入不重放。
- 新版 HOME 图标、军队卡、英雄皮肤和数量模板独立采样；来源、裁剪框、哈希与人审边界保存在 `assets/catalogs/*18_647_5*`。只支持已见军队页素材，未添加新版战斗卡模板。
- [固定对照配置](../../../configs/pilot-batch01-control.toml)默认 dry-run，冻结普通兵单边打法。正式对战仍需后续批次验证。
- [36 张新版候选试标](new-version/README.md)：35 个大本营、128 个防空候选框；完整标注、人审和冻结仍未完成。
- [8 张旧 TRAIN 试标](historical/README.md)：候选原图/schema 校验、来源组审计、部分框样张、类别指南与缺口。原型/正式 eligible 均为 0，没有启动训练。

## 实机证据与边界

原始窗口在 `E:/projects/AutoCOC-training/candidates/`；开发校准原图在 `E:/projects/AutoCOC-batch01/reports/batch01-20261028/`。本目录保存摘要与哈希索引，原图不挪动、不覆盖。来源中引用外部原图的条目依赖这些保留目录；可提交的脱敏军队 fixture 只负责离线回归，不替代原始证据。

| 窗口 | 结果 | 费用/输入边界 |
| --- | --- | --- |
| `pilot-baseline-20261028-v1` | 旧版强制更新，主村不可达 | 0 输入，0 搜索 |
| `pilot-baseline-20261028-v2` | 新版两次军队读取中第二帧英雄动画拒识 | 2 次免费导航，0 搜索、0 部署；人工辅助返回单列 |
| `pilot-baseline-20261028-v3` | 亡灵王子动画仍拒识，停止继续补单帧模板重跑 | 2 次免费导航，0 搜索、0 部署；失败原样保留 |
| [`pilot-baseline-20261028-v4`](pilot-baseline-v4.json) | 63.944 秒，连续两帧完整一致军队，自动回村 | 4 次免费导航，四项资源不变；源码/配置/素材哈希一致 |

v2/v3 表明：从个别同源样张得到的回归通过，不代表英雄动画覆盖充分。后续改进采用有限纯观察等待，仍要求连续两张完整一致的真实军队帧；不降低身份阈值，也不从卡位或容量猜身份。v4 的两张军队帧直接通过，未实际触发等待重读分支；该分支的证据为真实失败帧离线回放与测试，不是全动画姿势实机验收。OCR 限边实现后的 [v5](pilot-baseline-v5.json) 使用 `ocr.max_side_len=1280`，30.030 秒完成同一免费路线，哈希不变；四项资源前后为 6,057,482 / 11,166,206 / 52,897 / 1,349。v4 到 v5 之间资源变化来源未确认，不归为程序收益。最终补齐版本缺失/类型拒绝守卫后，[v7](pilot-baseline-v7.json) 用默认 OCR 尺寸在 58.669 秒内再次完成 4 次免费导航、完整一致军队与自动回村，四项资源前后为 5,996,282 / 11,166,206 / 52,897 / 1,349；源码/配置/素材哈希一致。最终运行及测试结论以 [verification.json](verification.json) 为准。

本轮安装更新、模拟器恢复、详情页取名和辅助回村是开发校准操作，不算自动路线成功。未购买、升级单位、修改编队、投兵或启动正式训练。第一个采集窗在 OCR `ConvTranspose.0: bad allocation` 后失败，0 张正式采集，原报告保持 failed；一次人工结束战斗返回主村，未投兵，保存图像的金币差为 1,700，自动费用核销仍未完成，见[独立恢复记录](collection-window01-recovery.json)。

后续增加可选 OCR 预处理长边限制，默认沿用 RapidOCR 原行为；本轮显式 1280，原始截图和输出坐标系保持不变。这不是任意图像或内存条件下的无 OOM 保证。失败原图的[离线复验](ocr-bounded-offline.json)和[识别集成结果](ocr-bounded-integration.json)通过，新基线 v5 与采集窗 02 实机通过。窗 02 获得 10 张候选、自动回村并核销 17,000 金币，见[原始报告](collection-window02.json)。最终五个成功窗共 **36 张新版候选**，自动费用核销 **61,200 金币**；首次失败搜索以人工恢复图的 1,700 差额计入预算，合计 **62,900 / 85,000**。三窗因倒计时短/不可读提前安全回村，达到窗口上限后停止；不声称完成约 50 个独立阵型。见[费用总表](collection-totals.json)和[新版候选试标包](new-version/README.md)。

低内存 OCR 有识别取舍：[v6](pilot-baseline-v6.json) 在主村把黑油读成非数字，因此零输入拒绝；原失败保留，同原图[默认尺寸离线复验](v6-home-default-ocr-replay.json)读取正确。没有降低资源阈值，也没有在运行中静默切换 OCR 尺寸。v7 默认配置随后通过。

最终回归 **256 passed、257 subtests passed、1 skipped**，跳过的是缺少可选旧军队样张的测试。版本元数据为空时的拒绝测试先复现了 3 个失败，再收紧为白名单版本与整型版本码匹配；原失败测试记录及修复见[回执](version-guard-regression.json)。

## 复验

```powershell
$env:PYTHONPATH = 'E:\projects\AutoCOC-batch01\src'
& 'E:\projects\AutoCOC\.venv\Scripts\python.exe' scripts/capture_pilot_baseline.py `
  --config reports/batch01-20261028/device.toml `
  --output 'E:\projects\AutoCOC-training\candidates\pilot-baseline-NEW' --limit-sec 150
```

输出必须是新目录，游戏应已在目标设备前台并停在主村，模型/Jev 关闭；入口不会自行启动游戏。本机配置不含凭据，未提交到仓库。迁移机器时重新配置并核验设备、分辨率、模板路径和当前版本。历史候选的数据校验命令见其[说明](historical/README.md)。
