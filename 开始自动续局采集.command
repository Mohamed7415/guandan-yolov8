#!/bin/zsh
cd -- "${0:A:h}" || exit 1
if [[ ! -x .venv/bin/python ]]; then
  print "未找到项目 Python 环境。"
  read -r "?按回车关闭。"
  exit 1
fi
print "先手动进入普通金蛋场并开启游戏自带托管，再运行本脚本。"
print "只识别普通结算页续局；不从大厅开局，不充值、不报名赛事。"
print "保持小程序在前台。Ctrl+C 或将鼠标移到屏幕左上角可停止续局。"
run_folder="data/candidates/auto_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
mkdir -p -- "$run_folder" || exit 1
capture_pid=""
cleanup_capture() {
  if [[ -n "$capture_pid" ]]; then
    kill -TERM "$capture_pid" 2>/dev/null
    wait "$capture_pid" 2>/dev/null
  fi
}
trap cleanup_capture EXIT INT TERM
.venv/bin/python -u scripts/auto_capture.py --target 200 --interval 2 --max-minutes 60 --gameplay-only --run-dir "$run_folder" &
capture_pid=$!
.venv/bin/python -u scripts/auto_next.py --max-rounds 10 --max-minutes 60 --activate-on-start --capture-run-dir "$run_folder"
continue_result=$?
cleanup_capture
capture_pid=""
print "已结束，候选截图位于：$run_folder"
read -r "?按回车关闭。"
exit $continue_result
