# -*-coding:utf-8-*-

import logging, sys
import os
import time

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding('utf8')


log_time = time.strftime('%m-%d %H-%M-%S', time.localtime(time.time()))
class Logger():
    def __init__(self, file_path="tmp/log.log"):
        #if not os.path.exists(os.path.dirname(file_path)):
        #    os.makedirs(os.path.dirname(file_path))
        if os.path.isdir(file_path):
            file_path = os.path.join(file_path, 'tmp.log')

        self.logger = logging.getLogger(file_path)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fh = logging.FileHandler(file_path, encoding='utf-8')
            fh.setLevel(logging.INFO)

            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter(
                #fmt="%(asctime)s [%(levelname).1s|%(threadName)s|%(process)d] %(message)s",
                fmt="%(asctime)s [%(levelname).1s|%(process)d] %(message)s",
                datefmt='%m-%d %H:%M:%S')
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

    def error(self, msg, *args, **kwargs):
        if self.logger is not None:
            self.logger.error(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        if self.logger is not None:
            self.logger.info(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        if self.logger is not None:
            self.logger.warning(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if self.logger is not None:
            self.logger.warning(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        if self.logger is not None:
            self.logger.exception(msg, *args, exc_info=True, **kwargs)



def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s [%(levelname).1s] %(message)', datefmt='%m-%d %H:%M:%S')
    handler = logging.FileHandler('log.txt', mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger

