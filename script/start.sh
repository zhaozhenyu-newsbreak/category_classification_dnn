#!/bin/sh
# Created Time : Fri 24 Aug 2018 07:26:11 PM CST
# Description: 启动脚本, NDP部署部署多个实例

date "+%Y-%m-%d %H:%M:%S"
echo 'starting server'

port=$1
PYTHON=/home/$USER/miniconda3/envs/pandian.env/bin/python
$PYTHON server.py -p $port -f config.ini
