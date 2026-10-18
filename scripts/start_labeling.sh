#!/bin/sh
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ ! -x "$project_dir/.venv-label/bin/xanylabeling" ]; then
  echo "未找到 .venv-label 中的 X-AnyLabeling，请先完成独立标注环境安装。" >&2
  exit 2
fi
image_dir="$project_dir/data/raw"
annotation_dir="$project_dir/data/annotations"
if [ -d "$project_dir/data/xanylabeling_full2000/images" ]; then
  image_dir="$project_dir/data/xanylabeling_full2000/images"
  annotation_dir="$project_dir/data/xanylabeling_full2000/annotations"
fi
mkdir -p "$image_dir" "$annotation_dir"
exec "$project_dir/.venv-label/bin/xanylabeling" \
  --config "$project_dir/configs/xanylabeling.yaml" \
  --filename "$image_dir" \
  --output "$annotation_dir" \
  --no-auto-update-check
