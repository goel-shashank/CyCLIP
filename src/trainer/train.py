import time
import wandb
import torch
import logging
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast

def get_loss(umodel, outputs, criterion, options):
    if(options.distributed):
        gathered_image_embeds = [torch.zeros_like(outputs.image_embeds) for _ in range(options.world_size)]
        gathered_text_embeds = [torch.zeros_like(outputs.text_embeds) for _ in range(options.world_size)]

        dist.all_gather(gathered_image_embeds, outputs.image_embeds)
        dist.all_gather(gathered_text_embeds, outputs.text_embeds)

        image_embeds = torch.cat(gathered_image_embeds[:options.rank] + [outputs.image_embeds] + gathered_image_embeds[options.rank + 1:])
        text_embeds = torch.cat(gathered_text_embeds[:options.rank]+ [outputs.text_embeds] + gathered_text_embeds[options.rank + 1:])

        logits_per_image = umodel.logit_scale.exp() * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()
    else:
        logits_per_image = umodel.logit_scale.exp() * outputs.image_embeds @ outputs.text_embeds.t()
        logits_per_text = logits_per_image.t()

    target = torch.arange(len(logits_per_image)).long().to(options.map_location, non_blocking = True)
    contrative_loss = (criterion(logits_per_image, target) + criterion(logits_per_text, target)) / 2
    symmetric_loss = options.symlambda * (logits_per_image - logits_per_text).square().mean() / (umodel.logit_scale.exp() * umodel.logit_scale.exp())

    loss = contrative_loss + symmetric_loss
    return contrative_loss, symmetric_loss, loss

def train(epoch, model, data, optimizer, scheduler, scaler, options):    
    dataloader = data["train"]
    if(options.distributed): dataloader.sampler.set_epoch(epoch)

    model.train()
    criterion = nn.CrossEntropyLoss().to(options.map_location)

    modulo = int(dataloader.num_samples / options.train_batch_size / options.world_size / 10)

    start = time.time()
    for index, batch in enumerate(dataloader):        
        step = dataloader.num_batches * epoch + index
        scheduler(step)

        optimizer.zero_grad()
        
        input_ids, attention_mask, pixel_values = batch["input_ids"].to(options.map_location, non_blocking = True), batch["attention_mask"].to(options.map_location, non_blocking = True), batch["pixel_values"].to(options.map_location, non_blocking = True)
        outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)

        umodel = model.module if(options.distributed) else model

        with autocast():
            contrative_loss, symmetric_loss, loss = get_loss(umodel, outputs, criterion, options)
            scaler.scale(loss).backward()
            scaler.step(optimizer)

        scaler.update()

        umodel.logit_scale.data = torch.clamp(umodel.logit_scale.data, 0, 4.6052)

        end = time.time()

        if(options.master and (((index + 1) % modulo == 0) or (index == dataloader.num_batches - 1))):
            num_samples = (index + 1) * len(input_ids) * options.world_size
            dataloader_num_samples = dataloader.num_samples

            logging.info(f"Train Epoch: {epoch:02d} [{num_samples}/{dataloader_num_samples} ({100.0 * (index + 1) / dataloader.num_batches:.0f}%)]\tLoss: {loss.item():.6f}\tTime taken {end - start:.3f}\tLearning Rate: {optimizer.param_groups[0]['lr']:.9f}")

            metrics = {"contrative_loss": contrative_loss.item(), "symmetric_loss": symmetric_loss.item(), "loss": loss.item(), "time": end - start, "lr": optimizer.param_groups[0]["lr"]}
            if(options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"train/{key}": value, "step": step})
        
            start = time.time()