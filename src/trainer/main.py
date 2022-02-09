import os

os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_API_KEY"] = "b5a237c3bc440290f623cc2ba16bb43394072c0c" 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import sys
import time
import wandb
import torch
import logging
import warnings
import numpy as np
import tensorflow as tf
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

# from pkgs.huggingface.clip import load
from pkgs.openai.clip import load
from .train import train
from .evaluate import evaluate
from .data import get_data
from .parser import parse_args
from .scheduler import cosine_scheduler
from .logger import get_logger, set_logger
from torch.cuda.amp import GradScaler

mp.set_start_method("spawn", force = True)
warnings.filterwarnings("ignore")

def worker(rank, options, logger):
    # set the properties of the worker
    options.rank = rank
    options.master = options.rank == 0
    options.map_location = torch.device("cpu" if(options.device == "cpu") else f"cuda:{options.rank}")

    # set logging for worker
    set_logger(rank = options.rank, logger = logger, log_level = options.log_level, distributed = options.distributed)

    if(options.master):
        if(options.device == "cpu"):
            logging.info(f"Using cpu device")
        elif(options.device == "gpu"):
            if(not options.distributed):
                logging.info(f"Using gpu device")
            else:
                logging.info(f"Using multiple gpu devices")
                logging.info(f"Using multiple gpu devices ({options.world_size})")
            
        logging.info("Params:")
        params_file = os.path.join(options.logs, options.name, "params.txt")
        with open(params_file, "w") as file:
            pass
            for key in sorted(vars(options)):
                value = getattr(options, key)
                logging.info(f"  {key}: {value}")
                file.write(f"{key}: {value}\n")

    # set the communication backend in the distributed setting
    if(options.distributed):
        dist.init_process_group(backend = options.distributed_backend, init_method = options.distributed_init_method, world_size = options.world_size, rank = options.rank)
    
    # load the clip model processor
    model, processor = load(name = options.model_name, pretrained = options.pretrained)
    
    # move the model on gpu device
    if(options.device == "gpu"):
        torch.cuda.set_device(options.rank)
        model.to(options.map_location)
        if(options.distributed):
            model = DDP(model, device_ids = [options.rank])
    else:
        model.float()

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
            state_dict = checkpoint["state_dict"]
            if(not options.distributed and next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
            if(optimizer is not None): optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"Loaded checkpoint '{options.checkpoint}' (start epoch {checkpoint['epoch']})")
        else:
            logging.info(f"No checkpoint found at {options.checkpoint}")

    cudnn.benchmark = True
    cudnn.deterministic = False

    # setup wandb logging
    if(options.wandb and options.master):
        logging.debug("Starting wandb")
        wandb.init(project = "mrl", notes = options.wandb_notes, tags = [], config = vars(options))
        wandb.run.name = options.name
        if(options.debug):
            wandb.watch(model, log = "all")
        wandb.save(params_file)
        logging.debug("Finished loading wandb")

    # evaluate in the beginning
    evaluate(start_epoch, model, processor, data, options)

    if(data["train"] is not None):
        scaler = GradScaler()
        # start training
        best_loss = np.inf
        for epoch in range(start_epoch + 1, options.epochs + 1):
            if(options.master):
                logging.info(f"Starting Epoch {epoch}")

            start = time.time()
            train(epoch, model, data, optimizer, scheduler, scaler, options)
            end = time.time()
            if(options.master):
                logging.info(f"Finished Epoch {epoch}, Time Taken: {end - start:.3f}")

            metrics = evaluate(epoch, model, processor, data, options)

            # saving checkpoint
            if(options.master):
                checkpoint = {"epoch": epoch, "name": options.name, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                torch.save(checkpoint, os.path.join(options.checkpoint_path, f"epoch_{epoch}.pt"))
                if("loss" in metrics):
                    if(metrics["loss"] < best_loss):
                        best_loss = metrics["loss"]
                        torch.save(checkpoint, os.path.join(options.checkpoint_path, f"epoch.best.pt"))

    if(options.distributed):
        dist.destroy_process_group()

    # finish wandb
    if(options.wandb and options.master):
        wandb.finish()

def main():
    options = parse_args()

    if(options.name is None):
        options.name = time.strftime(f"%Y:%m:%d:%H:%M:%S", time.gmtime())

    if(os.path.exists(os.path.join(options.logs, options.name))):
        if(options.overwrite):
            if(os.path.exists(os.path.join(options.logs, options.name, "output.log"))):
                os.remove(os.path.join(options.logs, options.name, "output.log"))
        else:
            print("Experiment exists; use the --name argument to specify another experiment name")
            sys.exit()
    
    os.makedirs(os.path.join(options.logs, options.name), exist_ok = True)

    options.log_path = os.path.join(options.logs, options.name, "output.log")
    options.log_level = logging.DEBUG if options.debug else logging.INFO

    logger, listener = get_logger(options.log_path, options.log_level)

    options.checkpoint_path = os.path.join(options.logs, options.name, "checkpoints")
    os.makedirs(options.checkpoint_path, exist_ok = True)

    options.ngpus = torch.cuda.device_count()

    if(options.ngpus == 0 or options.device == "cpu"):
        options.device = "cpu"
        options.distributed = False
        options.world_size = 1
        worker(0, options, logger)
    else:
        if(options.ngpus == 1 or not options.distributed):
            options.device = "gpu"
            options.distributed = False
            options.world_size = 1
            worker(0, options, logger)
        else:
            options.device = "gpu"
            options.distributed = True
            options.world_size = options.ngpus
            mp.spawn(worker, nprocs = options.ngpus, args = (options, logger))
    
    listener.stop()
    
if(__name__ == "__main__"):
    main()