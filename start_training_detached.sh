#!/bin/bash
set -euo pipefail

mkdir -p logs

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="logs/train_${timestamp}.log"
pid_file="logs/train.pid"

if [ -f "${pid_file}" ] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    echo "Training is already running with PID $(cat "${pid_file}")."
    echo "Log: $(readlink -f logs/latest_train.log 2>/dev/null || echo "${log_file}")"
    exit 1
fi

nohup python -u main.py --config configs/rs3dbench_depth.yaml > "${log_file}" 2>&1 &
pid="$!"
echo "${pid}" > "${pid_file}"
ln -sfn "$(basename "${log_file}")" logs/latest_train.log

echo "Started training in the background."
echo "PID: ${pid}"
echo "Log: ${log_file}"
echo "Follow logs with: tail -f logs/latest_train.log"
