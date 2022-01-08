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

def set_worker_logger(rank, logger, log_level, distributed):
    queue_handler = QueueHandler(logger)
    queue_handler.addFilter(LogFilter(rank, distributed))
    queue_handler.setLevel(log_level)
    queue_handler.flush()

    logger = logging.getLogger()
    logger.addHandler(queue_handler)
    logger.setLevel(log_level)

def get_root_logger(log_file, log_level):
    logger = mp.Queue(-1)

    formatter = Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt = "%Y-%m-%d,%H:%M:%S")
    
    file_handler = FileHandler(filename = log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    stream_handler = StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)

    listener = QueueListener(logger, file_handler, stream_handler)
    listener.start()

    return logger, listener
