import os
import torch
import logging
import numpy as np
import pandas as pd
from PIL import Image
from imagenetv2_pytorch import ImageNetV2Dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class CSVDataset(Dataset):
    def __init__(self, path, image_key, caption_key, delimiter, processor):
        logging.debug(f"Loading csv data from {path}")

        df = pd.read_csv(path, sep = delimiter)

        self.dir = os.path.dirname(path)
        self.images = df[image_key].tolist()
        self.captions = processor.tokenizer(df[caption_key].tolist(), return_tensors = "pt", padding = True, truncation = True)
        self.processor = processor

        logging.debug("Loaded data")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        item["input_ids"] = self.captions["input_ids"][idx]
        item["attention_mask"] = self.captions["attention_mask"][idx]
        item["pixel_values"] = self.processor.feature_extractor(Image.open(os.path.join(self.dir, self.images[idx])).convert("RGB"), return_tensors = "pt").pixel_values[0]
        return item

def get_dataloader(options, processor, train):
    path = options.train_data if train else options.validation_data
    if(path is None): return

    dataset = CSVDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor)
    sampler = DistributedSampler(dataset) if(options.distributed and train) else None

    dataloader = DataLoader(dataset, batch_size = options.batch_size, shuffle = train and (sampler is None), num_workers = options.workers, pin_memory = True, sampler = sampler, drop_last = train)
    dataloader.num_samples = len(dataloader) * options.batch_size
    dataloader.num_batches = len(dataloader)

    return dataloader

def get_test_dataloader(options, processor):
    path = options.test_data
    if(path is None): return

    dataset = ImageNetV2Dataset(location = path, transform = processor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.batch_size, num_workers = options.workers, sampler = None)

    return dataloader

def get_data(options, processor):
    data = {}
    
    data["train"] = get_dataloader(options, processor, train = True)
    data["validation"] = get_dataloader(options, processor, train = False)
    data["test"] = get_test_dataloader(options, processor)

    return data
