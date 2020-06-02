#!/usr/bin/env python
# coding=utf-8
"""
# Description: 日志模块
"""
import logging
import logging.handlers
from cloghandler import ConcurrentRotatingFileHandler

def getLoggers(loggerName , loggerLevel , loggerLocation ):
    """
    生成logger，进程非安全
    :param loggerName:logger名字
    :param loggerLevel: logger等级
    :param loggerLocation: logger文件位置
    :return:
    """
    logger = logging.getLogger(loggerName)
    logger.setLevel(loggerLevel)
    format = "%(asctime)s - %(levelname)s : %(message)s"
    formater = logging.Formatter(format)
    #handler = logging.handlers.TimedRotatingFileHandler(loggerLocation, "D", 0, 0)
    handler = logging.FileHandler(loggerLocation, mode='a', encoding=None, delay=False)
    handler.suffix = "%Y%m%d"
    handler.setFormatter(formater)
    logger.addHandler(handler)
    return logger

def getcLoggers(loggerName,loggerLevel,loggerLocation):
    '''
    生成进程安全的logger
    '''
    logger = logging.getLogger(loggerName)
    logger.setLevel(loggerLevel)
    rotateHandler = ConcurrentRotatingFileHandler(loggerLocation,'a',1073741824,5)
    logger.addHandler(rotateHandler)
    
    return logger

if __name__=='__main__':
    logger = getcLoggers('test',logging.INFO,'./test.log')
    logger.info('test')
'''
sample:

from logger import getLoggers

        logService = getLoggers('logService', logging.INFO, LOG_PATH + '/service.log')
        logArgs = getLoggers('logArgs', logging.INFO, LOG_PATH + '/args.log')
        logOutput = getLoggers('logOutput', logging.INFO, LOG_PATH + '/output.log')
        logError = getLoggers('logError', logging.INFO, LOG_PATH + '/error.log')
        logDebug = getLoggers('logDebug', logging.INFO, LOG_PATH + '/debug.log')


        logArgs.info('docsim server\t%s\t%s\t%s' % (docid, title, body))
        logError.error('docsim server\t%s\t%s' % (docid, str(msg)))
'''
