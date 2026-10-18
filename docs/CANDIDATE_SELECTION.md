# 固定验证规则与候选选择

`native2098_960_v1` 必须遵守训练前保存的[实验协议](../configs/experiments/native2098_960_v1.json)：固定 960 输入、置信度 0.25、类别感知 NMS IoU 0.45、匹配 IoU 0.5，在同一 400 张验证图上比较。只有误报减少、整图全对增加，且每张图每个类别的漏检都不增加，才提名新模型进入旧回归。否则保留原 960 候选。原实验及阈值不因本工具变更。

[验证门槛](../scripts/validation_gate.py)核对全部 54 类的 TP/FP/FN、逐图错误、图数和全对数是否与真实标签相符；会拒绝把同样数量的漏检转移到另一张图或另一个类别，也拒绝靠少输出获得零误报、修改阈值/NMS、换数据或套用其他模型的报告。它固定原实验、原基线报告及这 400 张图和标签的字节哈希，并将候选报告与实际 ONNX、导出和数值一致性验证记录关联。

现有基线报告在当前 400 张验证图的 7803 张真实牌上通过一致性核对；相同报告自比被正确拒绝为“无改进”。这只是历史报告审计，没有重新运行检测器，也不是新模型成绩。见[审计证据](../reports/published/native2098_validation_gate_baseline_audit.json)。8 项新增检查覆盖漏检转移、类别转移、零输出、阈值/NMS变化、报告计数矛盾、模型替换、数据漂移及决策不可覆盖；全套 78 项本地检查通过。

训练实际完成并写出 `training_record.json` 后，先用现有安全导出器导出新路径，不能恢复已完成的旧训练或覆盖旧模型：

```sh
.venv/bin/python scripts/export_onnx.py \
  --checkpoint runs/cards/native2098_960_v1/weights/best.pt \
  --output artifacts/candidates/native2098_960_v1.onnx

.venv/bin/python scripts/evaluate_onnx_point.py \
  --data data/prepared_full2098_native_v1 \
  --model artifacts/candidates/native2098_960_v1.onnx \
  --split val --confidence 0.25 \
  --output reports/continuous_goal/native2098_960_v1_val.json

.venv/bin/python scripts/validation_gate.py \
  --config configs/validation_gates/native2098_960_v1.json \
  --candidate-report reports/continuous_goal/native2098_960_v1_val.json \
  --model artifacts/candidates/native2098_960_v1.onnx \
  --verification artifacts/candidates/native2098_960_v1.verification.json \
  --output reports/continuous_goal/native2098_960_v1_validation_decision.json
```

评测器只增加模型、清单和实际输入哈希，不改变原匹配或推理算法。选择决定单独保存且不可覆盖；仅 `accepted_for_old_regression=true` 时才写入该决定内的冻结候选配置，随后可做旧 208 张回归。通过这一门槛也不代表最终目标完成。

不读取、不推理、不训练使用已封存的 `batch002` 新盲测候选。新增 98 张已进入本轮训练，只能作训练/回归数据。最终新盲测仍需达到预定规模并在最终模型和后处理冻结后评测。
