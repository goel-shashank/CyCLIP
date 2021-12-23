import torch
import logging
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

def set_logger(rank, logger, log_level, distributed):
    queue_handler = QueueHandler(logger)
    queue_handler.addFilter(LogFilter(rank, distributed))
    queue_handler.setLevel(log_level)

    logger = logging.getLogger()
    logger.addHandler(queue_handler)
    logger.setLevel(log_level)

def get_logger(log_file, log_level):
    torch.multiprocessing.set_start_method("spawn")
    logger = torch.multiprocessing.Queue(-1)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt = "%Y-%m-%d,%H:%M:%S")
    
    file_handler = logging.FileHandler(filename = log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)

    queue_listener = QueueListener(logger, file_handler, stream_handler)
    queue_listener.start()

    return logger

