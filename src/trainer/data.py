import os
import torch
import logging
import itertools
import torchvision
import pandas as pd
from PIL import Image, ImageFile
from imagenetv2_pytorch import ImageNetV2Dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# IMPORTANT
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CSVDataset(Dataset):
    def __init__(self, path, image_key, caption_key, delimiter, processor):
        logging.debug(f"Loading csv data from {path}")

        df = pd.read_csv(path, sep = delimiter)

        self.dir = os.path.dirname(path)
        self.images = df[image_key].tolist()
        self.captions = processor.text(df[caption_key].tolist())
        self.processor = processor

        logging.debug("Loaded data")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        item["input_ids"] = self.captions["input_ids"][idx]
        item["attention_mask"] = self.captions["attention_mask"][idx]
        item["pixel_values"] = self.processor.image(Image.open(os.path.join(self.dir, self.images[idx])))
        return item

def get_dataloader(options, processor, train):
    path = options.train_data if train else options.validation_data
    if(path is None): return

    batch_size = options.train_batch_size if train else options.eval_batch_size

    dataset = CSVDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor)
    sampler = DistributedSampler(dataset) if(options.distributed and train) else None

    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = train and (sampler is None), num_workers = options.workers, pin_memory = True, sampler = sampler, drop_last = train)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader

class ImageNetDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        image_file_paths = sorted(os.listdir(os.path.join(self.root, "images")))
        labels = list(map(lambda s: list(map(lambda e: int(e) if e else -1, s.strip().split(","))), open(os.path.join(root, "labels.txt")).read().split("\n")))
        labels = list(zip(*itertools.zip_longest(*labels, fillvalue = -1)))
        self.dataset = [(image_file_path, torch.tensor(label)) for image_file_path, label in zip(image_file_paths, labels) if any(i != -1 for i in label) > 0]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_file_path, label = self.dataset[idx]
        image = self.transform(Image.open(os.path.join(self.root, "images", image_file_path)))
        return image, label

def get_test_dataloader(options, processor):
    if(options.test_data_dir is None): return

    if(options.test_data_type == "Imagenet"):
        dataset = ImageNetDataset(root = options.test_data_dir, transform = processor.image)
    elif(options.test_data_type == "CIFAR10"):
        dataset = torchvision.datasets.CIFAR10(root = options.test_data_dir, download = True, train = False, transform = processor.image)
    elif(options.test_data_type == "CIFAR100"):
        dataset = torchvision.datasets.CIFAR100(root = options.test_data_dir, download = True, train = False, transform = processor.image)
    else:
        raise Exception("Test dataset type {options.test_data_type} is not supported")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.eval_batch_size, num_workers = options.workers, sampler = None)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader

def get_data(options, processor):
    data = {}
    
    data["train"] = get_dataloader(options, processor, train = True)
    data["validation"] = get_dataloader(options, processor, train = False)
    data["test"] = get_test_dataloader(options, processor)

    return data
