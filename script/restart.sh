#!/bin/sh
# Created Time : Fri 24 Aug 2018 07:43:20 PM CST
# File Name: docsim_service_restart.sh
# Description: 所有进程关闭,自动重启脚本,NDP目前不需要

while [ 1 ]
do
    for i in $(seq 9003 9003)
    do
        pid=$(pgrep -f "python docsim_server.py")
        if [ "$pid" = "" ]
        then
            timestamp=`date +%F-%T`
            echo -n $timestamp" Restarting Server $i..."
            sh start.sh
            pid_new=$(pgrep -f "python server.py")
            echo - n" new pid: " $pid_new
        fi
    done
    sleep 60
done

