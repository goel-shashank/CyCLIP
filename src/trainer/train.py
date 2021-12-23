import os
import time
import wandb
import torch
import logging
import torch.nn as nn

def get_loss(umodel, outputs, criterion, options):
    if(options.distributed):
        gathered_image_embeds = [torch.zeros_like(outputs.image_embeds) for _ in range(options.ndevices)]
        gathered_text_embeds = [torch.zeros_like(outputs.text_embeds) for _ in range(options.ndevices)]

        torch.distributed.all_gather(gathered_image_embeds, outputs.image_embeds)
        torch.distributed.all_gather(gathered_text_embeds, outputs.text_embeds)

        image_embeds = torch.cat([outputs.image_embeds] + gathered_image_embeds[:options.rank] + gathered_image_embeds[options.rank + 1:])
        text_embeds = torch.cat([outputs.text_embeds] + gathered_text_embeds[:options.rank]+ gathered_text_embeds[options.rank + 1:])

        logits_per_image = umodel.logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()
    else:
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

    target = torch.arange(len(logits_per_image)).long().to(options.map_location, non_blocking = True)
    loss = (criterion(logits_per_image, target) + criterion(logits_per_text, target)) / 2

    return loss

def train(epoch, model, data, optimizer, scheduler, options):    
    model.train()
    dataloader = data["train"]
    criterion = nn.CrossEntropyLoss().to(options.map_location)

    start = time.time()
    for index, batch in enumerate(dataloader):
        step = dataloader.num_batches * epoch + index
        scheduler(step)

        optimizer.zero_grad()
        
        input_ids, attention_mask, pixel_values = batch["input_ids"].to(options.map_location, non_blocking = True), batch["attention_mask"].to(options.map_location, non_blocking = True), batch["pixel_values"].to(options.map_location, non_blocking = True) 
        outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)

        umodel = model.module if(options.distributed) else model

        loss = get_loss(umodel, outputs, criterion, options)
        loss.backward()
        optimizer.step()

        # clamp model's logit scale to log(100) = 4.6052 (from the original paper)
        umodel.logit_scale.data = torch.clamp(umodel.logit_scale.data, 0, 4.6052)

        end = time.time()

        if(options.master and (((index + 1) % 20 == 0) or (index == dataloader.num_batches - 1))):
            num_samples = (index + 1) * len(input_ids) * options.ndevices
            dataloader_num_samples = dataloader.num_samples

            logging.info(f"Train Epoch: {epoch} [{num_samples}/{dataloader_num_samples} ({100.0 * (index + 1) / dataloader.num_batches:.0f}%)]\tLoss: {loss.item():.6f}\tTime taken {end - start:.3f}\tLearning Rate: {optimizer.param_groups[0]['lr']:5f}")

            metrics = {"loss": loss.item(), "time": end - start, "lr": optimizer.param_groups[0]["lr"]}
            if(options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"train/{key}": value, "step": step})
        
        start = time.time()
