import logging
import torch.multiprocessing as mp
from logging import Formatter, FileHandler, StreamHandler
from logging.handlers import QueueHandler, QueueListener

class LogFilter(logging.Filter):
    def __init__(self, rank, distributed):
        super().__init__()
        self.rank = rank
        self.distributed = distributed

    def filter(self, record):
        if(self.distributed):
            record.msg = f"Rank {self.rank} | {record.msg}"
        return True

def set_logger(rank, logger, distributed = False):
    queue_handler = QueueHandler(logger)
    queue_handler.addFilter(LogFilter(rank, distributed))
    queue_handler.setLevel(logging.INFO)
    queue_handler.flush()

    logger = logging.getLogger()
    logger.addHandler(queue_handler)
    logger.setLevel(logging.INFO)

def get_logger(log_file_path):
    logger = mp.Queue(-1)

    formatter = Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt = "%Y-%m-%d,%H:%M:%S")
    
    file_handler = FileHandler(log_file_path, "w+")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    listener = QueueListener(logger, file_handler, stream_handler)

    return logger, listener
