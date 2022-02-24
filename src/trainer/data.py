import os
import torch
import logging
import itertools
import torchvision
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

def seed_everything(seed):
    seed = 0
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
ImageFile.LOAD_TRUNCATED_IMAGES = True

class TextDataset(Dataset):
    def __init__(self, path, caption_key, delimiter, processor, augment = False):
        logging.debug(f"Loading unaligned text data from {path}")

        df = pd.read_csv(path, sep = delimiter)
        self.captions = processor.process_text(df[caption_key].tolist())

        self.augment = augment
        if(self.augment): 
            self.augmented_captions = processor.process_text(df["augmented_" + caption_key].tolist())

        logging.debug("Loaded unaligned text data")

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        item = {}
        item["input_ids"] = self.captions["input_ids"][idx]
        item["attention_mask"] = self.captions["attention_mask"][idx]
        if(self.augment):
            item["augmented_input_ids"] = self.augmented_captions["input_ids"][idx]
            item["augmented_attention_mask"] = self.augmented_captions["attention_mask"][idx]
        return item

class ImageDataset(Dataset):
    def __init__(self, path, image_key, delimiter, processor, augment = False):
        logging.debug(f"Loading unaligned image data from {path}")

        df = pd.read_csv(path, sep = delimiter)
        self.root = os.path.dirname(path)
        self.images = df[image_key].tolist()
        self.processor = processor

        self.augment = augment
        if(self.augment): 
            if(f"augmented_{image_key}" in df.columns):
                self.augment_transform = None
                self.augmented_images = df[f"augmented_{image_key}"].tolist()
            else:
                self.augment_transform = torchvision.transforms.AutoAugment()
                self.augmented_images = None

        logging.debug("Loaded unaligned image data")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        image = Image.open(os.path.join(self.root, self.images[idx]))
        item["pixel_values"] = self.processor.process_image(image)
        if(self.augment):
            if(self.augment_transform is None):
                augmented_image = Image.open(os.path.join(self.root, self.augmented_images[idx]))
                item["augmented_pixel_values"] = self.processor.process_image(augmented_image)
            else:
                item["augmented_pixel_values"] = self.processor.process_image(self.augment_transform(image))
        return item

class TextImageDataset(Dataset):
    def __init__(self, path, image_key, caption_key, delimiter, processor, noise = False):
        logging.debug(f"Loading aligned data from {path}")

        df = pd.read_csv(path, sep = delimiter)

        self.root = os.path.dirname(path)
        self.images = df[("augmented_" if noise else "") + image_key].tolist()
        self.captions = processor.process_text(df[("augmented_" if noise else "") + caption_key].tolist())
        self.processor = processor

        logging.debug("Loaded data")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        item["input_ids"] = self.captions["input_ids"][idx]
        item["attention_mask"] = self.captions["attention_mask"][idx]
        item["pixel_values"] = self.processor.process_image(Image.open(os.path.join(self.root, self.images[idx])))
        return item

class ImageNetDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        df = pd.read_csv(os.path.join(root, "labels.csv"))
        self.images = df["image"]
        self.labels = df["label"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.transform(Image.open(os.path.join(self.root, self.images[idx])))
        label = self.labels[idx]
        return image, label

def get_train_dataloader(options, processor):
    path = options.train_data
    if(path is None): return None, None, None, None

    path_supplement = options.train_supplement_data
    path_unaligned_text = options.train_unaligned_text_data
    path_unaligned_image = options.train_unaligned_image_data

    if(path_supplement is None):
        if(path_unaligned_text is None and path_unaligned_image is None):
            batch_size = options.batch_size

            dataset = TextImageDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor)
            sampler = DistributedSampler(dataset) if(options.distributed) else None

            dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False and (sampler is None), num_workers = options.num_workers, pin_memory = True, sampler = sampler, drop_last = True, worker_init_fn = seed_everything)
            dataloader.num_samples = len(dataloader) * batch_size 
            dataloader.num_batches = len(dataloader)

            return dataloader, DataLoader([]), DataLoader([]), DataLoader([])
        
        if(path_unaligned_text is not None and path_unaligned_image is None):
            batch_size = round(options.batch_size * (1 - options.fraction))
            batch_size_unaligned_text = options.batch_size - batch_size
            
            dataset = TextImageDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor)
            dataset_unaligned_text = TextDataset(path_unaligned_text, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor, augment = options.augment_text)
            
            sampler = DistributedSampler(dataset) if(options.distributed) else None
            sampler_unaligned_text = DistributedSampler(dataset_unaligned_text) if(options.distributed) else None

            dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = (sampler is None), num_workers = options.num_workers, pin_memory = True, sampler = sampler, drop_last = True)
            dataloader_unaligned_text = DataLoader(dataset_unaligned_text, batch_size = batch_size_unaligned_text, shuffle = (sampler_unaligned_text is None), num_workers = options.num_workers, pin_memory = True, sampler = sampler_unaligned_text, drop_last = True)
            
            dataloader.num_samples = len(dataloader) * batch_size 
            dataloader.num_batches = len(dataloader)
            
            return dataloader, DataLoader([]), dataloader_unaligned_text, DataLoader([])
        
        if(path_unaligned_text is None and path_unaligned_image is not None):
            batch_size = round(options.batch_size * (1 - options.fraction))
            batch_size_unaligned_image = options.batch_size - batch_size
            
            dataset = TextImageDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor)
            dataset_unaligned_image = ImageDataset(path_unaligned_image, image_key = options.image_key, delimiter = options.delimiter, processor = processor, augment = options.augment_image)
            
            sampler = DistributedSampler(dataset) if(options.distributed) else None
            sampler_unaligned_image = DistributedSampler(dataset_unaligned_image) if(options.distributed) else None

            dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = (sampler is None), num_workers = options.num_workers, pin_memory = True, sampler = sampler, drop_last = True)
            dataloader_unaligned_image = DataLoader(dataset_unaligned_image, batch_size = batch_size_unaligned_image, shuffle = (sampler_unaligned_image is None), num_workers = options.num_workers, pin_memory = True, sampler = sampler_unaligned_image, drop_last = True)
            
            dataloader.num_samples = len(dataloader) * batch_size 
            dataloader.num_batches = len(dataloader)
            
            return dataloader, DataLoader([]), DataLoader([]), dataloader_unaligned_image
        
        if(path_unaligned_text is not None and path_unaligned_image is not None):
            batch_size = (round(options.batch_size * (1 - options.fraction)) // 2) * 2
            batch_size_unaligned_text = (options.batch_size - batch_size) // 2
            batch_size_unaligned_image = (options.batch_size - batch_size) // 2
            
            dataset = TextImageDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor)
            dataset_unaligned_text = TextDataset(path_unaligned_text, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor, augment = options.augment_text)
            dataset_unaligned_image = ImageDataset(path_unaligned_image, image_key = options.image_key, delimiter = options.delimiter, processor = processor, augment = options.augment_image)
            
            sampler = DistributedSampler(dataset) if(options.distributed) else None
            sampler_unaligned_text = DistributedSampler(dataset_unaligned_text) if(options.distributed) else None
            sampler_unaligned_image = DistributedSampler(dataset_unaligned_image) if(options.distributed) else None

            dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = (sampler is None), num_workers = options.num_workers, pin_memory = True, sampler = sampler, drop_last = True)
            dataloader_unaligned_text = DataLoader(dataset_unaligned_text, batch_size = batch_size_unaligned_text, shuffle = (sampler_unaligned_text is None), num_workers = options.num_workers, pin_memory = True, sampler = sampler_unaligned_text, drop_last = True)
            dataloader_unaligned_image = DataLoader(dataset_unaligned_image, batch_size = batch_size_unaligned_image, shuffle = (sampler_unaligned_image is None), num_workers = options.num_workers, pin_memory = True, sampler = sampler_unaligned_image, drop_last = True)
            
            dataloader.num_samples = len(dataloader) * batch_size 
            dataloader.num_batches = len(dataloader)
            
            return dataloader, DataLoader([]), dataloader_unaligned_text, dataloader_unaligned_image
    else:
        batch_size = round(options.batch_size * (1 - options.fraction))
        batch_size_supplement = options.batch_size - batch_size
        
        dataset = TextImageDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor)
        dataset_supplement = TextImageDataset(path_supplement, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor, noise = options.noise)
        
        sampler = DistributedSampler(dataset) if(options.distributed) else None
        sampler_supplement = DistributedSampler(dataset_supplement) if(options.distributed) else None

        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = (sampler is None), num_workers = options.num_workers, pin_memory = True, sampler = sampler, drop_last = True)
        dataloader_supplement = DataLoader(dataset_supplement, batch_size = batch_size_supplement, shuffle = (sampler_supplement is None), num_workers = options.num_workers, pin_memory = True, sampler = sampler_supplement, drop_last = True)
        
        dataloader.num_samples = len(dataloader) * batch_size 
        dataloader.num_batches = len(dataloader)
        
        return dataloader, dataloader_supplement, DataLoader([]), DataLoader([])

def get_validation_dataloader(options, processor):
    path = options.validation_data
    if(path is None): return

    dataset = TextImageDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor)
    dataloader = DataLoader(dataset, batch_size = options.batch_size, shuffle = False, num_workers = options.num_workers, pin_memory = True, sampler = None, drop_last = False)
    dataloader.num_samples = len(dataset) 
    dataloader.num_batches = len(dataloader)

    return dataloader

def get_eval_test_dataloader(options, processor):
    if(options.eval_test_data_dir is None): return

    if(options.eval_data_type in ["Imagenet", "ImagenetV2", "ImagenetSketch"]):
        dataset = ImageNetDataset(root = options.eval_test_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR10"):
        dataset = torchvision.datasets.CIFAR10(root = options.eval_test_data_dir, download = True, train = False, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR100"):
        dataset = torchvision.datasets.CIFAR100(root = options.eval_test_data_dir, download = True, train = False, transform = processor.process_image)
    else:
        raise Exception("Eval dataset type {options.eval_data_type} is not supported")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.batch_size, num_workers = options.num_workers, sampler = None)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader

def get_eval_train_dataloader(options, processor):
    if(not options.linear_probe or options.eval_train_data_dir is None): return

    if(options.eval_data_type in ["Imagenet", "ImagenetV2", "ImagenetSketch"]):
        dataset = ImageNetDataset(root = options.eval_train_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR10"):
        dataset = torchvision.datasets.CIFAR10(root = options.eval_train_data_dir, download = True, train = True, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR100"):
        dataset = torchvision.datasets.CIFAR100(root = options.eval_train_data_dir, download = True, train = True, transform = processor.process_image)
    else:
        raise Exception("Test dataset type {options.eval_data_type} is not supported")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.linear_probe_batch_size, num_workers = options.num_workers, sampler = None)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader

def load(options, processor):
    data = {}
    
    data["train"], data["train_supplement"], data["train_unaligned_text"], data["train_unaligned_image"] = get_train_dataloader(options, processor)
    data["validation"] = get_validation_dataloader(options, processor)
    data["eval_test"] = get_eval_test_dataloader(options, processor)
    data["eval_train"] = get_eval_train_dataloader(options, processor)

    return data
