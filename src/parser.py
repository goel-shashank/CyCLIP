import os
import argparse
import utils.config as config
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm    
from .scheduler import cosine_scheduler

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", type = str, default = "default", help = "Experiment Name")
    parser.add_argument("--logs", type = str, default = os.path.join(config.root, "logs/"), help = "Logs directory path")
    parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")
    parser.add_argument("--train_data", type = str, default = None, help = "Path to train data csv/tsv file")
    parser.add_argument("--validation_data", type = str, default = None, help = "Path to validation data csv/tsv file")
    parser.add_argument("--eval_data_type", type = str, default = None, choices = ["Caltech101", "CIFAR10", "CIFAR100", "DTD", "FGVCAircraft", "Flowers102", "Food101", "GTSRB", "ImageNet1K", "OxfordIIITPet", "RenderedSST2", "StanfordCars", "STL10", "SVHN", "ImageNetSketch", "ImageNetV2", "ImageNet-A", "ImageNet-R"], help = "Test dataset type")
    parser.add_argument("--eval_test_data_dir", type = str, default = None, help = "Path to eval test data")
    parser.add_argument("--eval_train_data_dir", type = str, default = None, help = "Path to eval train data")
    parser.add_argument("--linear_probe", action = "store_true", default = False, help = "Linear Probe classification")
    parser.add_argument("--linear_probe_batch_size", type = int, default = 16, help = "Linear Probe batch size")
    parser.add_argument("--linear_probe_num_epochs", type = int, default = 32, help = "Linear Probe num epochs")
    parser.add_argument("--delimiter", type = str, default = ",", help = "For train/validation data csv file, the delimiter to use")
    parser.add_argument("--image_key", type = str, default = "image", help = "For train/validation data csv file, the column name for the image paths")
    parser.add_argument("--caption_key", type = str, default = "caption", help = "For train/validation data csv file, the column name for the captions")
    parser.add_argument("--device", type = str, default = None, choices = ["cpu", "gpu"], help = "Specify device type to use (default: gpu > cpu)")
    parser.add_argument("--device_id", type = int, default = 0, help = "Specify device id if using single gpu")
    parser.add_argument("--distributed", action = "store_true", default = False, help = "Use multiple gpus if available")
    parser.add_argument("--distributed_backend", type = str, default = "nccl", help = "Distributed backend")
    parser.add_argument("--distributed_init_method", type = str, default = "tcp://127.0.0.1:5432", help = "Distributed init method")
    parser.add_argument("--device_ids", nargs = "+", default = None, help = "Specify device ids if using multiple gpus")
    parser.add_argument("--wandb", action = "store_true", default = False, help = "Enable wandb logging")
    parser.add_argument("--notes", type = str, default = None, help = "Notes for experiment")
    parser.add_argument("--num_workers", type = int, default = 8, help = "Number of workers per gpu")
    parser.add_argument("--inmodal", action = "store_true", default = False, help = "Inmodality Training")
    parser.add_argument("--epochs", type = int, default = 64, help = "Number of train epochs")
    parser.add_argument("--batch_size", type = int, default = 128, help = "Batch size")
    parser.add_argument("--lr", type = float, default = 5e-4, help = "Learning rate")
    parser.add_argument("--beta1", type = float, default = 0.9, help = "Adam momentum factor (Beta 1)")
    parser.add_argument("--beta2", type = float, default = 0.999, help = "Adam rmsprop factor (Beta 2)")
    parser.add_argument("--eps", type = float, default = 1e-8, help = "Adam eps")
    parser.add_argument("--weight_decay", type = float, default = 0.1, help = "Adam weight decay")
    parser.add_argument("--num_warmup_steps", type = int, default = 10000, help = "Number of steps to warmup the learning rate")
    parser.add_argument("--cylambda1", type = float, default = 0, help = "Cyclic regularization lambda 1")
    parser.add_argument("--cylambda2", type = float, default = 0, help = "Cyclic regularization lambda 2")
    parser.add_argument("--checkpoint", default = None, type = str, help = "Path to checkpoint to resume training")
    parser.add_argument("--pretrained", default = False, action = "store_true", help = "Use the OpenAI pretrained models")

    options = parser.parse_args()
    return options