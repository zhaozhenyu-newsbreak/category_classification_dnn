#!/bin/sh
# Created Time : Fri 24 Aug 2018 07:39:20 PM CST
# Description: 关闭所有进程

date "+%Y-%m-%d %H:%M:%S"
echo 'kill server'

port=$1
lsof -i:$port | awk 'NR>1{print $2}' | xargs kill -9
