import time
import wandb
import torch
import logging
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast

def get_loss(umodel, outputs, criterion, options, true_batch_size):
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    
    if(options.distributed):
        gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
        gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]
        
        dist.all_gather(gathered_image_embeds, image_embeds)
        dist.all_gather(gathered_text_embeds, text_embeds)
        
        image_embeds = torch.cat(gathered_image_embeds[:options.rank] + [image_embeds] + gathered_image_embeds[options.rank + 1:])
        text_embeds = torch.cat(gathered_text_embeds[:options.rank]+ [text_embeds] + gathered_text_embeds[options.rank + 1:])

    logits_per_image = umodel.logit_scale.exp() * image_embeds @ text_embeds.t()
    logits_per_text = logits_per_image.t()
    
    batch_size = len(logits_per_text)

    if(options.loss == "contrastive"):
        target = torch.arange(batch_size).long().to(options.device, non_blocking = True)
        loss = (criterion(logits_per_image, target) + criterion(logits_per_text, target)) / 2

    if(options.loss == "trace"):
        loss = -torch.mean(torch.diag(logits_per_image))

    if(options.loss == "ranking1"):
        eye = torch.eye(batch_size).to(options.device, non_blocking = True)
        a = torch.exp(logits_per_text - torch.diag(logits_per_text)) * (1 - eye)
        b = torch.exp(logits_per_image - torch.diag(logits_per_image)) * (1 - eye)
        loss = torch.mean(torch.log(1 + a) + torch.log(1 + b))

    if(options.loss == "ranking2"):
        eye = torch.eye(batch_size).to(options.device, non_blocking = True)
        a = torch.diag(logits_per_text) - torch.log(torch.sum(torch.exp(logits_per_text) * (1 - eye), dim = -1))
        b = torch.diag(logits_per_image) - torch.log(torch.sum(torch.exp(logits_per_image) * (1 - eye), dim = -1))
        loss = torch.mean(a) + torch.mean(b)

    alignloss = loss
    with torch.no_grad():
        if(true_batch_size < batch_size):
            target = torch.arange(true_batch_size).long().to(options.device, non_blocking = True)
            alignloss = (criterion(logits_per_image[:true_batch_size, ...], target) + criterion(logits_per_text[:true_batch_size, ...], target)) / 2
            
    symloss = options.symlambda * (logits_per_image - logits_per_text).square().mean() / (umodel.logit_scale.exp() * umodel.logit_scale.exp()) * batch_size
    loss += symloss

    return loss, symloss, alignloss 

def train(epoch, model, data, optimizer, scheduler, scaler, options):    
    dataloader, dataloader_supplement, dataloader_unaligned_text, dataloader_unaligned_image = data["train"], iter(data["train_supplement"]), iter(data["train_unaligned_text"]), iter(data["train_unaligned_image"])
    if(options.distributed): dataloader.sampler.set_epoch(epoch)

    model.train()
    criterion = nn.CrossEntropyLoss().to(options.device)

    modulo = int(dataloader.num_samples / options.batch_size / options.num_devices / 10)
    umodel = model.module if(options.distributed) else model

    start = time.time()

    for index, batch in enumerate(dataloader): 
        try:
            batch_supplement = next(dataloader_supplement)
        except StopIteration:
            dataloader_supplement = iter(data["train_supplement"])
            batch_supplement = next(dataloader_supplement, None)

        try:
            batch_unaligned_image = next(dataloader_unaligned_image)
        except StopIteration:
            dataloader_unaligned_image = iter(data["train_unaligned_image"])
            batch_unaligned_image = next(dataloader_unaligned_image, None)

        try:
            batch_unaligned_text = next(dataloader_unaligned_text)
        except StopIteration:
            dataloader_unaligned_text = iter(data["train_unaligned_text"])
            batch_unaligned_text = next(dataloader_unaligned_text, None)

        step = dataloader.num_batches * epoch + index
        scheduler(step)

        optimizer.zero_grad()
        
        input_ids, attention_mask, pixel_values = batch["input_ids"].to(options.device, non_blocking = True), batch["attention_mask"].to(options.device, non_blocking = True), batch["pixel_values"].to(options.device, non_blocking = True)
        outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)

        if(batch_supplement is not None):
            input_ids_supplement, attention_mask_supplement, pixel_values_supplement = batch_supplement["input_ids"].to(options.device, non_blocking = True), batch_supplement["attention_mask"].to(options.device, non_blocking = True), batch_supplement["pixel_values"].to(options.device, non_blocking = True)
            outputs_supplement = model(input_ids = input_ids_supplement, attention_mask = attention_mask_supplement, pixel_values = pixel_values_supplement)
            outputs.text_embeds = torch.cat([outputs.text_embeds, outputs_supplement.text_embeds], dim = 0)
            outputs.image_embeds = torch.cat([outputs.image_embeds, outputs_supplement.image_embeds], dim = 0)

        if(batch_unaligned_text is not None):
            input_ids_unaligned_text, attention_mask_unaligned_text = batch_unaligned_text["input_ids"].to(options.device, non_blocking = True), batch_unaligned_text["attention_mask"].to(options.device, non_blocking = True)
            text_features_unaligned_text = umodel.get_text_features(input_ids = input_ids_unaligned_text, attention_mask = attention_mask_unaligned_text)
            text_features_unaligned_text = text_features_unaligned_text / text_features_unaligned_text.norm(dim = -1, keepdim = True)
            outputs.text_embeds = torch.cat([outputs.text_embeds, text_features_unaligned_text], dim = 0)
            if(options.augment_text):
                augmented_input_ids_unaligned_text, augmented_attention_mask_unaligned_text = batch_unaligned_text["augmented_input_ids"].to(options.device, non_blocking = True), batch_unaligned_text["augmented_attention_mask"].to(options.device, non_blocking = True)
                augmented_text_features_unaligned_text = umodel.get_text_features(input_ids = augmented_input_ids_unaligned_text, attention_mask = augmented_attention_mask_unaligned_text)
                augmented_text_features_unaligned_text = augmented_text_features_unaligned_text / augmented_text_features_unaligned_text.norm(dim = -1, keepdim = True)
                outputs.image_embeds = torch.cat([outputs.image_embeds, augmented_text_features_unaligned_text], dim = 0)
            else:
                outputs.image_embeds = torch.cat([outputs.image_embeds, text_features_unaligned_text], dim = 0)

        if(batch_unaligned_image is not None):
            pixel_values_unaligned_image = batch_unaligned_image["pixel_values"].to(options.device, non_blocking = True)
            image_features_unaligned_image = umodel.get_image_features(pixel_values = pixel_values_unaligned_image)
            image_features_unaligned_image = image_features_unaligned_image / image_features_unaligned_image.norm(dim = -1, keepdim = True)
            outputs.image_embeds = torch.cat([outputs.image_embeds, image_features_unaligned_image], dim = 0)
            if(options.augment_image):
                augmented_pixel_values_unaligned_image = batch_unaligned_image["augmented_pixel_values"].to(options.device, non_blocking = True)
                augmented_image_features_unaligned_image = umodel.get_image_features(pixel_values = augmented_pixel_values_unaligned_image)
                augmented_image_features_unaligned_image = augmented_image_features_unaligned_image / augmented_image_features_unaligned_image.norm(dim = -1, keepdim = True)
                outputs.text_embeds = torch.cat([outputs.text_embeds, augmented_image_features_unaligned_image], dim = 0)
            else:
                outputs.text_embeds = torch.cat([outputs.text_embeds, image_features_unaligned_image], dim = 0)

        with autocast():
            loss, symloss, alignloss = get_loss(umodel, outputs, criterion, options, len(batch["input_ids"]))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        
        scaler.update()
        umodel.logit_scale.data = torch.clamp(umodel.logit_scale.data, 0, 4.6052)

        end = time.time()

        if(options.master and (((index + 1) % modulo == 0) or (index == dataloader.num_batches - 1))):
            num_samples = (index + 1) * len(input_ids) * options.num_devices
            dataloader_num_samples = dataloader.num_samples

            logging.info(f"Train Epoch: {epoch:02d} [{num_samples}/{dataloader_num_samples} ({100.0 * (index + 1) / dataloader.num_batches:.0f}%)]\tLoss: {loss.item():.6f}\tTime taken {end - start:.3f}\tLearning Rate: {optimizer.param_groups[0]['lr']:.9f}")

            metrics = {"loss": loss.item(), "symloss": symloss.item(), "alignloss": alignloss.item(), "time": end - start, "lr": optimizer.param_groups[0]["lr"]}
            if(options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"train/{key}": value, "step": step})
        
            start = time.time()