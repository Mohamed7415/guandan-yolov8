# 新原生开发回归样本

`extra_kclub_on_aceclub.png` 是 2026-10-19 直接从微信小程序采集的完整原生 PNG，未裁剪或重编码。JSON 包含逐图目视复核的 27 张牌；审核者为助手，没有独立人工二审。

v1 ONNX 在固定置信度 0.25、类别感知 NMS IoU 0.45 下，检出全部 27 张牌，但在 A♣ 的可见条带额外输出一个 K♣：置信度 0.256013，坐标 `[337.96,686.08,469.25,744.40]`。完整预测、模型及文件哈希见 `baseline_v1.json`。

```sh
.venv/bin/python scripts/infer_onnx.py \
  examples/native_regression/extra_kclub_on_aceclub.png \
  --model artifacts/card_model.onnx --confidence 0.25 --iou 0.45 \
  --output reports/native_regression_check
```

本图已经用于诊断和补样，属于开发回归，不能计入今后的新独立盲测。保留失败结果供后续候选比较；不能提高阈值或删除该图来声称模型已全对。
