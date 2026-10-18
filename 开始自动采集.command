#!/bin/zsh
cd -- "${0:A:h}" || exit 1
if [[ ! -x .venv/bin/python ]]; then
  print "未找到项目 Python 环境，请先完成环境安装。"
  read -r "?按回车关闭。"
  exit 1
fi
.venv/bin/python -u scripts/auto_capture.py --target 200 --interval 2 --max-minutes 60
capture_result=$?
print ""
if [[ $capture_result -eq 0 ]]; then
  print "候选截图已保存在项目 data/candidates 下。标注复核后才能计入训练数据。"
fi
read -r "?按回车关闭。"
exit $capture_result
