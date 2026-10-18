# 《天天爱掼蛋》截图标注规范

本规范说明标注配置与转换流程；实际采集、审核和训练进度以清单及报告为准。第一轮以 **200 张真实、去重的截图**验证流程，核对质量后再扩到 2000 张。

## 类别与配置

`configs/classes.txt` 每行一个类别，行号减 1 即索引。顺序严格为点数 `3,4,5,6,7,8,9,10,J,Q,K,A,2`，每个点数依次 `♠,♥,♦,♣`，最后 `52=SJ` 小王、`53=BJ` 大王。共 54 类。`configs/classes.json` 保存同一顺序。

启动 `scripts/start_labeling.sh`，当前默认打开 `data/xanylabeling_full2000/images` 的 2000 张工程，将 JSON 保存到 `data/xanylabeling_full2000/annotations`；不存在完整工程时回退到 `data/raw` 和 `data/annotations`。完整工程的图像是原图链接，JSON 是可编辑副本。未来 GUI 编辑应通过 `data/manifest_xanylabeling_full2000.csv` 转换，主标注不会自动跟随副本编辑。配置使用官方支持的 `labels`、`sort_labels: false`、`validate_label: exact`，避免手误新增类别。原图保持原分辨率，训练时再进行 640×640 的缩放/填充。

本次已完成助手对全图和类别图集的复核，并实际在 X-AnyLabeling 打开确认；尚无独立人工二审。测试标注修订、原 JSON 备份和评测影响详见 `reports/test_annotation_revision/audit.json`。`reviewed=true` 记录本次助手审核状态，不能据此声称有人类二审。

## 框的统一规则

1. 使用矩形（`R`），**每张可独立确认点数和花色的牌标一个框**，使用牌的可见部分的最小轴对齐矩形。不要只圈点数/花色小图标；完整可见的牌圈完整牌面。
2. 重叠、扇形、出牌横向叠加时，只框该牌实际可见的部分，不推测被遮挡的完整边缘。若可见部分有多块，框包住属于该牌的所有可见部分。相邻牌的框允许重叠。将被其他牌遮挡的目标标记 `occluded`。
3. 图片边缘截断的牌，框止于图像边界并标 `truncated`。不允许框越界、零面积或重复标同一张牌。
4. 每张截图独立判断。点数/花色有一项无法确定时不猜类别，不根据上一帧、发牌顺序或同组其他牌补答案。若截图内存在看似可见目标但无法确定是否应标，保留为待复核，`reviewed=false`，不要当训练背景。
5. 两副牌出现相同牌面时仍逐张标注；例如可分辨出两张 `A♠` 就标两个 `A♠` 框。大王/小王按该游戏实际牌面文字和配色确认，不用普通牌颜色推测。
6. 菜单按钮、头像、筹码、牌背、装饰性花色图案不属于 54 类目标。游戏规则产生的特殊牌面仍按截图显示的物理牌面标注，不将“级牌/逢人配”等玩法角色当新类别。

## 场景与分组

`scene` 可以多选，以 `|` 分隔；同时可在 X-AnyLabeling 图像级 `flags` 中勾选这些名称：

| 值 | 含义 |
|---|---|
| `self_play` | 自己出牌的界面 |
| `others_play` | 其他玩家出牌 |
| `pass` | 含“不要/不出”动作或提示；可仍有手牌和出牌，须正常标注 |
| `fan` | 手牌铺开/扇形重叠 |
| `sorted` | 手牌整理后的排列 |
| `negative` | 经确认整图不存在可识别目标牌的背景界面 |

`negative` 必须经人工逐图审核，JSON `shapes` 为空；含可识别牌的图片即使有“不要”提示也不是负样本。空文件或漏标 JSON 不能算负样本。

采集清单 `data/capture_manifest.csv` 保存 `filename,session_id,group_id,scene,captured_at,sha256,width,height`。同一局统一 `session_id`，同一段连续画面统一 `group_id`。标注清单 `data/manifest.csv` 保存 `image,label,group_id,scene,reviewed,session_id`。路径相对清单目录或使用绝对路径；`label` 在转换前指向 X-AnyLabeling JSON。**同一局的所有帧必须放在同一训练/验证/测试子集**；优先按 `session_id` 划分，不能把临近画面随机打散。`shape.group_id` 是对象分组字段，不是数据集切分依据。

## 标注、审核与转换

从真实采集清单建立待标注清单，不会伪造 JSON 或负样本：

```sh
.venv-label/bin/python scripts/convert_annotations.py \
  --init-from-capture data/capture_manifest.csv \
  --manifest data/manifest.csv
scripts/start_labeling.sh
```

在 X-AnyLabeling 中逐图标注，检查类别、漏标、框边界和场景。完成独立复核后，再将对应 `data/manifest.csv` 行的 `reviewed` 改为 `true`。GUI 的 `checked` 可辅助记录复核进度，转换器以清单 `reviewed` 为准。未完成的图保持 `false`。

```sh
.venv-label/bin/python scripts/convert_annotations.py \
  --manifest data/manifest.csv --output data/converted
```

默认遇到任何未审核行、缺失 JSON、未知类别、尺寸不一致、非矩形或越界框时，整个批次在写训练标签前失败。只导出完成的子集可显式加 `--skip-unreviewed`；未审核行会记入报告，不能默认为背景。已存在的输出需换目录或加 `--overwrite`。原图和 JSON 不会被更改。

输出 `data/converted/labels/*.txt` 采用 YOLO 检测格式 `class_id cx cy w h`，后四项均按原图宽高归一化到 `[0,1]`；`manifest.csv` 供 `scripts/prepare_dataset.py` 按整局拆分；`conversion_report.json` 列出每类数量、缺失类别与少于 50 个实例的类别。这里不拆分训练集。规格中的每类至少 50 个是标注实例数目标，不是“200 张截图即自动满足”的保证。

## 官方格式依据

- [官方入门/命令行文档](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/get_started.md)：支持 `--config`、`--labels`、`--flags`、`--validatelabel exact`。
- [官方用户指南](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/user_guide.md)：JSON 顶层包含 `imagePath`、`imageWidth`、`imageHeight`、`flags`、`shapes`；矩形保存 `label`、`shape_type: rectangle` 与顶点 `points`。本机 4.1.3 GUI 实际保存四个轴对齐顶点（左上、右上、右下、左下）；转换器同时兼容旧的两个对角点格式。四点必须恰好覆盖轴对齐矩形的四个不同角，旋转框或斜四边形会被拒绝，不会直接取外接框。JSON 默认随图保存，也可更改输出目录。
- [官方默认配置](https://github.com/CVHub520/X-AnyLabeling/blob/main/anylabeling/configs/xanylabeling_config.yaml)：本项目仅覆盖需要的类别、校验、图像标记等字段。
- [官方包配置](https://github.com/CVHub520/X-AnyLabeling/blob/main/pyproject.toml)：PyPI 包为 `x-anylabeling-cvhub`，命令入口为 `xanylabeling`，支持 Python 3.12。独立 `.venv-label` 避免与训练环境的 Qt/OpenCV 依赖冲突；无需选用或下载自动标注模型即可手动标注。
