#!/bin/sh
# Created Time : Fri 24 Aug 2018 07:47:31 PM CST
# Description: NDP目前不需要
port=$1
pIDa=`lsof -i:$port |grep -v "PID" |awk '{print $2}'`
if [ "$pIDa" != "" ];
then
    return 0
else
    return 1
fi
