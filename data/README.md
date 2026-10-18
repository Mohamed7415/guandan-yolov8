# 2000 张真实截图数据

完整原生图像和标注在私有仓库的 [v1.0.0 Release](https://github.com/Mohamed7415/guandan-yolov8/releases/tag/v1.0.0)。下载：

- `dataset-metadata.zip`：2000 份 X-AnyLabeling JSON、YOLO 标签、54 类清单、固定数据划分和说明。
- 全部 `dataset-images-001.zip` 至 `dataset-images-005.zip`：5 个独立 ZIP 分卷，合计 2000 张原生截图。
- `dataset-inventory.json`、`SHA256SUMS`：文件和下载包校验清单。

把元数据包和全部 5 个原图包解压到同一个目录，例如 `data/guandan2000/`。这是 5 个可以分别解压的 ZIP，不是字节切片，不要拼接：

```sh
mkdir -p data/guandan2000
# 假设下载文件位于 downloads/
for asset in downloads/dataset-*.zip; do unzip -n "$asset" -d data/guandan2000; done
.venv/bin/python scripts/validate_dataset.py \
  --data data/guandan2000/data.yaml --full
```

目录结构：

```text
data/guandan2000/
  data.yaml
  images/{train,val,test}/
  labels/{train,val,test}/
  annotations/                 # X-AnyLabeling JSON
  split_manifest.csv          # 固定 1392 / 400 / 208 划分
  manifest.csv                # JSON 转换清单
  yolo_manifest.csv
```

JSON 中的 `imagePath` 已转换为可移植相对路径。`data.yaml` 未固定本机目录，脚本通过文件所在目录定位数据。原图字节与整局划分不变。发布版测试标签包含已审计的 10 个花色修订及 13 个漏框补标；模型权重没有再训练。完整说明见 [评测报告](../reports/FULL_2000_REPORT.md)。

仓库内的 `dataset_manifest.csv` 仅包含图片路径、SHA-256、组别和划分，便于核对数量，不包含整套图片。原始候选和未经审核的草稿不会发布为训练数据。

本机原有 `data/raw`、`data/annotations`、`data/prepared_full2000*` 仍保留，历史候选已移至 `.local_archive/20261018_publish_cleanup`。这些路径不是换机器使用的前提。
