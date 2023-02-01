from functools import wraps
import logging
import time 
logging.basicConfig(filename='./log/log_learn.log',
                    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__file__)

def func_log(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        t_start = time.time()
        print("call %s():" % func.__name__)
        for i in args:
            #print(i)
            logger.info(i)
        func(*args, **kwargs)
        t_end = time.time()
        logger.info(f'time cost {t_end-t_start} sec.')
    return decorated
