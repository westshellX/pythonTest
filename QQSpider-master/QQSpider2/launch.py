# coding=utf-8

import datetime
from init_messages import InitMessages
import spide_controller
import importlib,sys

importlib.reload(sys)


if __name__ == '__main__':
    """ QQSpider的启动文件 """
    try:
        my_messages = InitMessages()  # 读取本地的爬虫信息，并对爬虫进行参数初始化
        spider = spide_controller.SpideController(my_messages=my_messages)
        print("%s: Initial work completed! Now, We start:" % datetime.datetime.now())
        spider.beginer()  # 开始爬虫
    except Exception as e:
        print (e)
        print (datetime.datetime.now())
    finally:
        print ('%s: Done!' % datetime.datetime.now())
