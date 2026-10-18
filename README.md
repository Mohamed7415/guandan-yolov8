# 天天爱掼蛋 · YOLOv8 牌面识别

基于微信小程序《天天爱掼蛋》的 **2000 张真实截图**，训练 54 类 YOLOv8n 检测模型并导出 ONNX。首批 200 张试验、正式标注、训练和导出已完成。

| 交付 | 内容 |
|---|---|
| 正式模型 | [`artifacts/card_model.onnx`](artifacts/card_model.onnx)，FP32 / opset 12 |
| 训练权重 | [`artifacts/card_model.pt`](artifacts/card_model.pt) |
| 类别与模型参数 | [`classes.txt`](artifacts/classes.txt)、[`model_info.json`](artifacts/model_info.json) |
| 2000 张原图和完整标注 | [Releases 下载页](https://github.com/Mohamed7415/guandan-yolov8/releases/tag/v1.0.0)，原图分卷与标注包 |
| 评测与修订说明 | [`reports/FULL_2000_REPORT.md`](reports/FULL_2000_REPORT.md) |

## 已验证的结果

按整局划分训练 **1392**、验证 **400**、测试 **208** 张，三部分均有全部 54 类。修订后标注共 **38122** 个目标，29 张无牌负样本。全部源于真实小程序截图，未使用网络视频或生成图片。

修订测试标注后，mAP50 **99.47%**、mAP50–95 **99.35%**。置信度 0.25 时，3641 个测试目标检出 3641 个，另有 **24 个误报**；**190/208 张（91.35%）**整图完全匹配。原冻结测试 mAP50 为 99.15%。测试后按原图纠正了 10 个红桃/方块误标和 13 个漏框，模型与阈值未再调整；原版指标及修订记录保留。

标注由助手检查全图和类别图集，并在 X-AnyLabeling 打开验证，尚无独立人工二审。数据集中于当前小程序主题，以 1620×1000 为主；其他分辨率、皮肤和独立扇形手牌布局未验证。测试包含 5 个组，同局图片有相关性，不能据此保证未来画面零漏检。

**后续复核（2026-10-19）：** 又确认三张图有可见 8♠ 漏标。在保留 v1 发布数据的前提下，新回归版本统计为 **TP 3643、误报 22、漏检 1，190/208 张整图全对**；正式模型没有变化。上面的零漏检属于旧标签版本。[追加审计说明](docs/REGRESSION_V2_AUDIT.md) · [持续优化目标](docs/CONTINUOUS_GOAL.md)。

## 识别一张截图

建议 Python 3.12。仅做 ONNX 推理无需安装训练框架：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r artifacts/requirements-inference.txt
.venv/bin/python scripts/infer_onnx.py /绝对路径/游戏截图.png \
  --model artifacts/card_model.onnx --output predictions
```

输出识别 JSON 和画框 JPG。输入 RGB float32 `[1,3,640,640]`，保持比例缩放、114 填边后除以 255。输出 `[1,58,8400]`，前四通道是输入像素单位的 `cx,cy,w,h`，其余为 54 类分数。没有额外 objectness 或内置 NMS；需类别感知 NMS 和原图坐标还原，脚本已实现。起始置信度 0.25、NMS IoU 0.45。

## 标注、训练和数据

下载数据后，依照 [`data/README.md`](data/README.md) 将全部原图分卷与 `dataset-metadata.zip` 解压到同一目录。数据包包含原生图像、X-AnyLabeling JSON、YOLO 标签、固定划分和 SHA-256 清单。

- [训练与导出](README_training.md)
- [54 类标注规范](docs/ANNOTATION_GUIDE.md)
- [截图与自动续局](docs/AUTO_NEXT.md)，仅适用于已经开始的普通虚拟金蛋场
- [规格约定](specs/SPEC_NOTES.md)
- [第三方组件及游戏素材说明](THIRD_PARTY_NOTICES.md)

当前本机仍可双击 `X-AnyLabeling Project.app` 查看已整理的 2000 张工程；换机器后先安装 `requirements-label.txt` 并加载解压后的图像/标注目录。自动截图和续局进程已停止。

## 仓库维护

仓库保留通用采集/转换/训练/推理工具、模板、回归样本、正式模型和经路径清理的评测摘要。Python 环境、候选截图、未审核草稿、完整训练缓存和临时调试图片均不进入 Git。历史原件保留在本地 `.local_archive` 或未跟踪的数据目录，正式模型与冻结数据未删除。

```sh
.venv/bin/python -m unittest discover -s tests -v
```

原图与标注作为 Release 附件分开下载，避免把约 3 GB 数据直接放入 Git 历史。正式模型文件始终为 `card_model.onnx`；旧的 `card_model_pilot.onnx` 是低质量试验模型，不作为本次正式交付。
