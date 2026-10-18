# 训练、评测与 ONNX 导出

正式模型已完成；完整指标、18 张测试标签修订和实际训练轮数见 [评测报告](reports/FULL_2000_REPORT.md)。训练初始冻结数据为 38109 框；Release 包包含修订后的 38122 框。测试标签修订未改变本次模型权重，不能把修订版指纹写回历史训练记录。

## 环境与数据

使用 Python 3.12：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-training.txt
```

Ultralytics / PyTorch 等版本固定在 `requirements-training.txt`，本机完整依赖另见 `requirements-training.lock.txt`。本项目固定官方 YOLOv8n 架构；没有本地初始化权重时会下载官方 `yolov8n.pt`，不使用最新架构的默认模型。官方 COCO 初始化权重不能直接识别本项目的 54 类牌。

按 [数据说明](data/README.md) 下载并将元数据及全部原图 ZIP 解压到 `data/guandan2000/`，然后：

```sh
.venv/bin/python scripts/validate_dataset.py \
  --data data/guandan2000/data.yaml --full \
  --report reports/restored_dataset_validation.json
```

三个集合固定为 1392 / 400 / 208 张，按整局隔离；检查器验证类别顺序、54 类覆盖、坐标、框重复、图像完整性和跨集合泄漏。无牌负样本须显式复核，不能把未标注图当背景。

## 继续标注或加入新数据

X-AnyLabeling 安装在独立环境中，避免 Qt / OpenCV 与训练依赖冲突：

```sh
python3.12 -m venv .venv-label
.venv-label/bin/python -m pip install -r requirements-label.txt
.venv-label/bin/xanylabeling --config configs/xanylabeling.yaml \
  --filename data/guandan2000/images/train \
  --output data/guandan2000/annotations --no-auto-update-check
```

训练、验证、测试图像目录分别打开；现有 JSON 为可移植相对 `imagePath`。打开后先核对已有框被正确加载，再编辑和保存。完整标注规则见 [标注规范](docs/ANNOTATION_GUIDE.md)。本机旧工程的 GUI 修改保存在 `data/xanylabeling_full2000/annotations`，其专用输入清单是 `data/manifest_xanylabeling_full2000.csv`；该目录未作为 Git 运行前提。

新增真实数据时，清单必须填写真实的 `group_id/session_id`、`scene` 和 `reviewed`，至少三个独立正样本组；同局所有帧应在同一集合。先转换 JSON，再另建新数据版本：

```sh
.venv/bin/python scripts/convert_annotations.py \
  --manifest data/guandan2000/manifest.csv --output data/converted_next
.venv/bin/python scripts/prepare_dataset.py \
  --manifest data/converted_next/manifest.csv --output data/prepared_next
.venv/bin/python scripts/validate_dataset.py \
  --data data/prepared_next/data.yaml --full
```

重新划分后不能将新旧测试指标视为同一测试条件；严格比较应保留原测试组。训练本次固定划分的数据可直接使用 Release 的 `data.yaml`，无需再次准备或打散。

## 训练

```sh
.venv/bin/python scripts/train.py \
  --data data/guandan2000/data.yaml --stage full \
  --name full_next --batch 8 --cache-ram
```

默认正式训练上限 150 轮、patience 25，试验模式 60 轮。设备自动选择 CUDA / Apple MPS / CPU；内存不足可减小 batch，MPS 不可用可加 `--device cpu`。MPS 使用经过验证的 CPU 目标分配兼容处理，网络与反向传播仍在 MPS。RAM 缓存保存缩放后的图像，不改原图。

禁用翻转、色相变化和马赛克，以保留点数、朝向、红黑花色和遮挡语义。训练前校验官方 nano 架构与数据；输出包含 `weights/best.pt`、`results.csv`、验证图及 `training_record.json`。模型选择和阈值调整使用验证集，选择冻结后才评估测试集。

本次历史运行：第一次完成 2 轮后在第 3 轮 MPS 报错；第二次从其保存权重初始化（重新建立优化器），完成 16 轮后验证收敛停止，选择第 13 轮。没有声称完成 150 轮或触发自然 early stopping。[路径清理后的原训练记录](reports/published/full2000_training_record.json) 仅作历史审计；重新训练生成的新记录才用于导出新模型。

## 导出与评测

```sh
.venv/bin/python scripts/export_onnx.py \
  --checkpoint runs/cards/full_next/weights/best.pt \
  --output artifacts/card_model_next.onnx --evaluate-test
```

导出要求与权重匹配的已完成训练记录，核对冻结数据及标签哈希，不接受任意 COCO 权重；同名输出不覆盖。验证 ONNX checker、输入输出、类别顺序及真实验证图的 PyTorch/ONNX 原始数值一致性。`--evaluate-test` 在模型冻结后评估独立测试，另存两种格式的结果。

复现当前 ONNX 的固定阈值评测：

```sh
.venv/bin/python scripts/evaluate_onnx_point.py \
  --model artifacts/card_model.onnx --data data/guandan2000 \
  --split test --confidence 0.25 --output reports/reproduced_test.json
```

## 推理协议

输入 `[1,3,640,640]`、RGB、NCHW、float32/255，保持比例缩放并以 114 填边。标准 54 类检测头输出 `[1,58,8400]`；58 = 4 + 54，84 通道属于 80 类 COCO 模型。

前四通道是 `cx,cy,w,h`，单位为填边后输入图的像素；其余是 54 类 sigmoid 分数，没有额外 objectness。外部执行置信度过滤和类别感知 NMS，再减去填边、除以缩放比。示例工具 `scripts/infer_onnx.py` 已实现这一流程。
