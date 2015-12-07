__author__ = 'qiuchusheng'
import sys, logging

logger = None

def _init_logger():
    # the root logger level is set to debug by default, i.e. info message will not be printed
    dl = logging.getLogger('proj-default')
    dl.setLevel(logging.DEBUG)

    fmt = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stdo = logging.StreamHandler(stream=sys.stdout)
    stdo.setLevel(logging.INFO)
    stdo.setFormatter(fmt)
    dl.addHandler(stdo)

    # rolling = logging.FileHandler('log/info.log')
    # rolling.setLevel(logging.INFO)
    # dl.addHandler(rolling)
    return dl


if logger == None:
    logger = _init_logger()