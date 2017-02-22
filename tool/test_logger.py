import __init_path
from util import logger

if __name__ == '__main__':
    logger.set_file_handler()
    logger.info('it is a test')
    logger.debug('it is a test')
    logger.warning('it is a test')
    logger.error('it is a test')
