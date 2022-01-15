import os
import argparse
import utils.config as config

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", type = str, default = None, help = "Experiment Name (default: current timestamp)")
    parser.add_argument("--logs", type = str, default = os.path.join(config.root, "logs/"), help = "Logs directory path")
    parser.add_argument("--overwrite", action = "store_true", default = False, help = "Overwrite previous logs")
    parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")
    parser.add_argument("--train_data", type = str, default = None, help = "Path to train data csv/tsv file")
    parser.add_argument("--validation_data", type = str, default = None, help = "Path to validation data csv/tsv file")
    parser.add_argument("--test_data_dir", type = str, default = None, help = "Path to test data for conducting evaluation")
    parser.add_argument("--test_data_type", type = str, default = None, choices = ["Imagenet", "CIFAR10", "CIFAR100"], help = "Test dataset type")
    parser.add_argument("--delimiter", type = str, default = ",", help = "For train/validation data csv file, the delimiter to use")
    parser.add_argument("--image_key", type = str, default = "image", help = "For train/validation data csv file, the column name for the image paths")
    parser.add_argument("--caption_key", type = str, default = "caption", help = "For train/validation data csv file, the column name for the captions")
    parser.add_argument("--device", type = str, default = None, choices = ["cpu", "gpu"], help = "Specify device type to use (default: gpu > cpu)")
    parser.add_argument("--distributed", action = "store_true", default = False, help = "Use multiple gpus if available")
    parser.add_argument("--distributed_backend", type = str, default = "nccl", help = "Distributed backend")
    parser.add_argument("--distributed_init_method", type = str, default = "tcp://127.0.0.1:6100", help = "Distributed init method")
    parser.add_argument("--wandb", action = "store_true", default = False, help = "Enable wandb logging")
    parser.add_argument("--wandb_notes", type = str, default = None, help = "Notes for experiment")
    parser.add_argument("--workers", type = int, default = 1, help = "Number of workers per gpu")
    parser.add_argument("--epochs", type = int, default = 32, help = "Number of train epochs")
    parser.add_argument("--train_batch_size", type = int, default = 64, help = "Train Batch size per gpu")
    parser.add_argument("--eval_batch_size", type = int, default = 64, help = "Eval Batch size per gpu")
    parser.add_argument("--lr", type = float, default = 5e-4, help = "Learning rate")
    parser.add_argument("--beta1", type = float, default = 0.9, help = "Adam momentum factor (Beta 1)")
    parser.add_argument("--beta2", type = float, default = 0.999, help = "Adam rmsprop factor (Beta 2)")
    parser.add_argument("--eps", type = float, default = 1e-8, help = "Adam eps")
    parser.add_argument("--weight_decay", type = float, default = 0.1, help = "Adam weight decay")
    parser.add_argument("--warmup_steps", type = int, default = 10000, help = "Number of steps to warmup the learning rate")
    parser.add_argument("--checkpoint", default = None, type = str, help = "Path to checkpoint to resume training")
    parser.add_argument("--pretrained", default = False, action = "store_true", help = "Use the OpenAI pretrained models")
    parser.add_argument( "--debug", default = False, action = "store_true", help = "If true, more information is logged.")

    options = parser.parse_args()
    return options
