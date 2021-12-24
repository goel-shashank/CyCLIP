import os

os.environ["WANDB_SILENT"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import sys
import time
import wandb
import torch
import shutil
import logging
import numpy as np
import torch.nn as nn
import tensorflow as tf
import torch.optim as optim
import torch.backends.cudnn as cudnn

from .clip import load
from .train import train
from .evaluate import evaluate
from .data import get_data
from .parser import parse_args
from .logger import get_logger, set_logger
from .scheduler import cosine_scheduler

def worker(rank, options):
    # set the properties of the worker
    options.rank = rank
    options.master = options.rank == 0
    options.map_location = "cpu" if(options.device == "cpu") else f"cuda:{options.rank}"

    # set the prefix of the worker in logger
    set_logger(options.rank, options.logger, options.log_level, options.distributed)

    if(options.master):
        if(options.device == "cpu"):
            logging.info("Using cpu device")
        elif(options.device == "gpu"):
            if(not options.distributed):
                logging.info("Using gpu device")
            else:
                logging.info("Using gpu device with distributed backend")
            
        logging.info("Params:")
        params_file = os.path.join(options.logs, options.name, "params.txt")
        with open(params_file, "w") as file:
            for key in sorted(vars(options)):
                value = getattr(options, key)
                logging.info(f"  {key}: {value}")
                file.write(f"{key}: {value}\n")

    # set the communication backend in the distributed setting
    if(options.distributed):
        torch.distributed.init_process_group(backend = options.distributed_backend, init_method = options.distributed_init_method, world_size = options.ngpus, rank = options.rank)
    
    # load the clip model processor
    model, processor = load(pretrained = options.pretrained)

    # move the model on gpu device
    if(options.device == "gpu"):
        torch.cuda.set_device(options.rank)
        model.cuda(options.rank)
        if(options.distributed):
            model = nn.parallel.DistributedDataParallel(model, device_ids = [options.rank])

    # load the data
    data = get_data(options, processor)

    # set the optimizer and scheduler for training
    if(data["train"] is None):
        optimizer = None
        scheduler = None
    else:
        decay_parameters = []
        nodecay_parameters = []

        for name, parameter in model.named_parameters():
            if(all(key not in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
                decay_parameters.append(parameter)
            if(any(key in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
                nodecay_parameters.append(parameter)

        optimizer = optim.AdamW([{"params": nodecay_parameters, "weight_decay": 0}, {"params": decay_parameters, "weight_decay": options.weight_decay}], lr = options.lr, betas = (options.beta1, options.beta2), eps = options.eps)
        scheduler = cosine_scheduler(optimizer, options.lr, options.warmup_steps, data["train"].num_batches * options.epochs)

    # resume from a checkpoint if given
    start_epoch = 0
    if(options.checkpoint is not None):
        if(os.path.isfile(options.checkpoint)):
            checkpoint = torch.load(options.checkpoint, map_location = options.map_location)
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            if(optimizer is not None): optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"=> loaded checkpoint '{options.checkpoint}' (start epoch {checkpoint['epoch']})")
        else:
            logging.info(f"=> no checkpoint found at {options.checkpoint}")

    cudnn.benchmark = True
    cudnn.deterministic = False

    # setup wandb logging
    if(options.wandb and options.master):
        logging.debug("Starting wandb")
        wandb.init(project = "mint-multimodal", notes = options.wandb_notes, tags = [], config = vars(options))
        if(options.debug):
            wandb.watch(model, log = "all")
        wandb.save(params_file)
        logging.debug("Finished loading wandb")

    # evaluate in the beginning
    evaluate(start_epoch, model, processor, data, options)

    if(data["train"] is not None):
        # start training
        best_loss = np.inf
        for epoch in range(start_epoch + 1, options.epochs + 1):
            if(options.master):
                logging.info(f"Start Epoch {epoch}")

            train(epoch, model, data, optimizer, scheduler, options)
            metrics = evaluate(epoch, model, processor, data, options)

            # saving checkpoint
            if(options.master):
                checkpoint = {"epoch": epoch, "name": options.name, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                torch.save(checkpoint, os.path.join(options.checkpoint_path, f"epoch_{epoch}.pt"))
                if("loss" in metrics):
                    if(metrics["loss"] < best_loss):
                        best_loss = metrics["loss"]
                        torch.save(checkpoint, os.path.join(options.checkpoint_path, f"epoch_{epoch}.best.pt"))

    # finish wandb
    if(options.wandb and options.master):
        wandb.finish()

def main():
    options = parse_args()

    shutil.rmtree(options.logs, ignore_errors = True)

    if(options.name is None):
        options.name = time.strftime(f"date=%Y-%m-%d-%H-%M-%S", time.gmtime())

    os.makedirs(os.path.join(options.logs, options.name), exist_ok = True)

    options.log_path = os.path.join(options.logs, options.name, "output.log")
    options.log_level = logging.DEBUG if options.debug else logging.INFO

    if(os.path.exists(options.log_path)):
        print("Experiment already exists; use --name to specify an experiment identifier")
        sys.exit()

    options.logger, options.listener = get_logger(options.log_path, options.log_level)

    options.checkpoint_path = os.path.join(options.logs, options.name, "checkpoints")
    os.makedirs(options.checkpoint_path, exist_ok = True)

    options.ngpus = torch.cuda.device_count()
    print(options.ngpus)

    if(options.ngpus == 0 or options.device == "cpu"):
        options.device = "cpu"
        options.ndevices = 1
        options.distributed = False
        worker(0, options)
    else:
        if(options.ngpus == 1 or not options.distributed):
            options.device = "gpu"
            options.ndevices = 1
            options.distributed = False
            worker(0, options)
        else:
            options.device = "gpu"
            options.ndevices = options.ngpus
            options.distributed = True
            torch.multiprocessing.spawn(worker, nprocs = options.ngpus, args = (options))
    
    options.listener.stop()
    
if(__name__ == "__main__"):
    main()
