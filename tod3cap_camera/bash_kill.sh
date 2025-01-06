#!/bin/bash

# 获取包含指定路径的进程 PID
pids=$(ps aux | grep '/data/zyp/miniconda3/envs/todc/bin/python' | awk '{print $2}')

# 杀死这些进程
for pid in $pids; do
    kill -9 $pid
done