# 多视角拼接视频 vs 单前视 Front120 压缩视频 Caption 对比报告

日期：2026-06-24

## 目标

比较同一批 clip 在两种输入形式下的 caption 效果差异：

1. 多视角拼接视频 `*_multicam.mp4`
2. 单前视 `Front120` 压缩视频 `*_Front120.mp4`

重点关注：

- 驾驶相关信息完整性
- 对本车交互者的刻画是否准确
- 是否容易出现视角外信息缺失或事件误判
- 推理耗时与 token 开销

## 测试口径

### 公平对比口径

`multicam` 视频来自拼接脚本产物，时长固定为中间 10 秒。为避免把“时长不同”混入结论，单前视 `Front120` 也统一裁成中间 10 秒后再测试。

- 多视角 scene: [multicam_caption.yaml](/home/huajiang.sun/llmtagger/prompt_pipeline/scenes/multicam_caption.yaml)
- 单前视 scene: [front120_caption.yaml](/home/huajiang.sun/llmtagger/prompt_pipeline/scenes/front120_caption.yaml)
- 模型: `qwen3.7-plus`
- 接口: DashScope
- 输出: 单段纯中文 caption

### 样本范围

我重新筛了当前机器上同时具备 `multicam` 和 `Front120` 压缩视频的 UUID。交集只有 3 条，因此本次 `Front120` 主报告基于这 3 条样本。

用于主对比的 UUID：

- `0a15e23c-7e7d-4875-9cdb-9d4a9bbdf858`
- `0c069783-25d7-48de-b4be-b7ed500bf0c2`
- `2e33d932-6c4c-4dae-bed2-100db78466d0`

多视角输出目录：
[ab_multicam_front120set_3clips](/home/huajiang.sun/llmtagger/prompt_pipeline/captions/ab_multicam_front120set_3clips)

单前视 Front120 输出目录：
[ab_front120trim_3clips](/home/huajiang.sun/llmtagger/prompt_pipeline/captions/ab_front120trim_3clips)

## 输入差异

以 UUID `2e33d932-6c4c-4dae-bed2-100db78466d0` 为例：

- `multicam`: `1920x810`，`10.0s`
- `Front120` 原始压缩版: `1280x720`，`23.0s`
- `Front120` 裁剪后对比版: `1280x720`，`10.0s`

因此，主结论基于 `10s multicam vs 10s Front120`。

## 定量结果

### 公平对比：10 秒 `multicam` vs 10 秒 `Front120`

| 输入形式 | 样本数 | 平均 tokens | 平均耗时 | 平均字符数 |
|---|---:|---:|---:|---:|
| `multicam` | 3 | 8014.3 | 26.1s | 234.7 |
| `Front120` 裁剪 10s | 3 | 4963.0 | 22.4s | 220.0 |

观察：

- `Front120` 平均 token 明显更低。
- `Front120` 平均更快，约快 `3.7s`。
- `multicam` 的额外成本，主要换来的是侧后方与周边上下文信息。

补充说明：

- 这 3 条里有 2 条是园区/通道类简单场景，`Front120` 在这类场景下输出更短、更收敛，因此平均 token 降得比较明显。
- 在更复杂的路口和交互场景里，`Front120` 与 `multicam` 的 token 差距会缩小。

## 逐样本分析

### 1. `0a15e23c-7e7d-4875-9cdb-9d4a9bbdf858`

结论：`multicam` 更完整，`Front120` 可用但偏保守。

原因：

- `multicam` 描述到蓝色建筑出口、二维码立牌、抬杆道闸、减速带、斑马线、导向箭头，以及后方一辆白色轿车缓慢跟随。
- `Front120` 也抓住了主线，但只保留了出口车道、减速带、斑马线、导向箭头和向前靠近路口过程，写成“全程无其他车辆或行人交互”。
- 这里 `Front120` 并不是明显错误，而是天然拿不到后视关系，所以少写了后方跟随信息。

判断：当前向主线清晰、侧后方只是补充信息时，`Front120` 已经足以描述本车行为；如果希望把周边跟车关系也纳入 caption，`multicam` 更优。

### 2. `0c069783-25d7-48de-b4be-b7ed500bf0c2`

结论：两者都正确，`multicam` 略优。

原因：

- `multicam` 更完整地交代了“本车先在园区道路减速并向左转，再驶入带黄色门框的半封闭通道入口”，并保留了入口前导向箭头、蓝色指示牌、石墩、右侧静止行人等信息。
- `Front120` 也准确写出了单向通道、顶部横梁、右侧红衣人员和出口明暗变化，但对“进入通道前的道路几何变化”和“左转进入”交代较弱。

判断：如果场景涉及入口选择、转向进入、通道几何等连续动作，`multicam` 更容易把进入前后的关系写完整。

### 3. `2e33d932-6c4c-4dae-bed2-100db78466d0`

结论：两者都很好，`multicam` 略优但优势不大。

原因：

- `multicam` 和 `Front120` 都稳定识别出：阴雨、湿滑、积水反光、红灯、两辆白色车减速停车、限速 30、本车平稳减速靠近停止线。
- `multicam` 多写了“侧视和后视画面显示后方及侧方无紧急逼近车辆”，这是多视角补充信息。
- `Front120` 则多写了“右侧可见树木与建筑”，这类信息对驾驶决策价值较低。

判断：前方主线信息高度集中时，`Front120` 已经非常接近 `multicam`；多视角提升主要体现在把“周边无风险”也交代出来。

## 综合结论

### 哪种效果更好

主结论：如果目标是自动驾驶 caption 的完整性，`multicam` 仍然更好；如果目标是在更低成本下保住前向主线质量，`Front120` 已经是一个很强的单前视基线。

### `multicam` 的优势

- 能补足侧后方上下文，减少对本车交互关系的漏写
- 更容易把“进入前后的轨迹变化”和“周边是否有跟车/逼近车辆”写完整
- 在需要描述本车与周边交通单位相对关系时更稳

### `Front120` 的优势

- 成本更低，平均 token 更少，平均耗时更短
- 当前向主线清晰时，结果已经非常接近 `multicam`
- 对简单场景足以产出可用 caption

## 结论排序

如果只看这 3 条同源样本，我的判断是：

1. `multicam` 最适合生产高完整度 caption
2. `Front120` 是最合理的单前视替代方案

## 建议

1. 生产 caption 若强调交互完整性，优先使用 `multicam`。
2. 如果需要在成本和效果之间折中，优先选择 `Front120`。
3. 当前机器上能做同源 `multicam vs Front120` 的交集样本只有 3 条；如果后续要做更稳的结论，建议继续补齐更多同时存在 `multicam` 和 `Front120` 的 UUID，再扩大评测集。
