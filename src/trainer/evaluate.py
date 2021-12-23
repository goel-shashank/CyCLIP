import wandb
import torch
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm    

def get_validation_metrics(model, dataloader, options):
    logging.info("Starting validation")

    metrics = {}

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction = "sum").to(options.map_location)

    losses = []

    with torch.no_grad():
        # running the model on validation batches
        for batch in tqdm(dataloader):
            input_ids, attention_mask, pixel_values = batch["input_ids"].to(options.map_location, non_blocking = True), batch["attention_mask"].to(options.map_location, non_blocking = True), batch["pixel_values"].to(options.map_location, non_blocking = True) 
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)

            target = torch.arange(len(input_ids)).long().to(options.map_location, non_blocking = True)
            loss = (criterion(outputs.logits_per_image, target) + criterion(outputs.logits_per_text, target)) / 2

            losses.append(loss)

        loss = sum(losses) / dataloader.num_samples
        metrics["loss"] = loss

    logging.info("Finished validation")

    return metrics

def get_test_metrics(model, processor, dataloader, options):
    logging.info("Starting test")

    model.eval()

    # Generating the class embeddings using multiple templates
    config = eval(open(options.test_config, "r").read())
    classes, templates = config["classes"], config["templates"]
    with torch.no_grad():
        text_embeddings = []
        for c in tqdm(classes):
            texts = [template(c) for template in templates]
            text_tokens = processor.tokenizer(texts, return_tensors = "pt").to(options.map_location)
            text_embedding = model.module.encode_text(text_tokens) if(options.distributed) else model.encode_text(text_tokens)
            text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
            text_embedding = text_embedding.mean(dim = 0)
            text_embedding /= text_embedding.norm()
            text_embeddings.append(text_embedding)
        text_embeddings = torch.stack(text_embeddings, dim = 1).to(options.map_location)

    # Predictions using cosine similarity and calculating the metrics
    with torch.no_grad():
        topk = [1, 3, 5, 10]
        correct = {k: 0 for k in topk}

        for images, labels in tqdm(dataloader):
            images, labels = images.to(options.map_location), labels.to(options.map_location)
            image_embeddings = model.module.encode_image(images) if(options.distributed) else model.encode_image(images)
            
            logits = (image_embeddings @ text_embeddings)
            predictions = logits.topk(max(topk), 1)[1].T == labels

            for k in topk:
                correct[k] += predictions[:k].sum().item() 

    results = {"zero_shot_top{k}": correct[k] / dataloader.num_samples for k in topk}
    logging.info("Finished test")

    return results

def evaluate(epoch, model, processor, data, options):
    metrics = {}
    
    if(options.master):
        if(data["validation"] is not None): metrics.update(get_validation_metrics(model, data["validation"], options))
        if(data["test"] is not None): metrics.update(get_test_metrics(model, processor, data["test"], options))

        if(metrics):
            logging.info(f"Epoch {epoch} evaluation")
            for key, value in metrics.items():
                logging.info(f"{key}: {value:.4f}")

            if(options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"eval/{key}": value, "epoch": epoch})

    return metrics