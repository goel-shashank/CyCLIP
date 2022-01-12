import wandb
import torch
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm    

def get_validation_metrics(model, dataloader, options):
    logging.info("Started validating")

    metrics = {}

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction = "sum").to(options.map_location)

    losses = []

    with torch.no_grad():
        # running the model on validation batches
        for batch in tqdm(dataloader):
            input_ids, attention_mask, pixel_values = batch["input_ids"].to(options.map_location, non_blocking = True), batch["attention_mask"].to(options.map_location, non_blocking = True), batch["pixel_values"].to(options.map_location, non_blocking = True) 
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)
            
            umodel = model.module if(options.distributed) else model
            logits_per_image = umodel.logit_scale.exp() * outputs.image_embeds @ outputs.text_embeds.t()
            logits_per_text = logits_per_image.t()

            target = torch.arange(len(input_ids)).long().to(options.map_location, non_blocking = True)
            loss = (criterion(outputs.logits_per_image, target) + criterion(outputs.logits_per_text, target)) / 2

            losses.append(loss)

        loss = sum(losses) / dataloader.num_samples
        metrics["loss"] = loss

    logging.info("Finished validating")

    return metrics

def get_test_metrics(model, processor, dataloader, options):
    logging.info("Started testing")

    model.eval()
    umodel = model.module if(options.distributed) else model

    # Generating the class embeddings using multiple templates
    config = eval(open(f"{options.test_data_dir}/classes.py", "r").read())
    classes, templates = config["classes"], config["templates"]
    with torch.no_grad():
        text_embeddings = []
        for c in tqdm(classes):
            text = [template(c) for template in templates]
            text_tokens = processor.process_text(text)
            text_input_ids, text_attention_mask = text_tokens["input_ids"].to(options.map_location), text_tokens["attention_mask"].to(options.map_location) 
            text_embedding = umodel.get_text_features(input_ids = text_input_ids, attention_mask = text_attention_mask)
            text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
            text_embedding = text_embedding.mean(dim = 0)
            text_embedding /= text_embedding.norm()
            text_embeddings.append(text_embedding)
        text_embeddings = torch.stack(text_embeddings, dim = 1).to(options.map_location)

    # Predictions using cosine similarity and calculating the metrics
    with torch.no_grad():
        topk = [1, 3, 5, 10]
        correct = {k: 0 for k in topk}

        for image, label in tqdm(dataloader):
            image, label = image.to(options.map_location), label.to(options.map_location)
            image_embedding = umodel.get_image_features(image)

            logits = (image_embedding @ text_embeddings)
            ranks = logits.topk(max(topk), 1)[1].T

            if(len(label.shape) == 1):
                predictions = ranks == label
            else:
                predictions = torch.vstack([torch.any(e == label.T, dim = 0) for e in ranks])

            for k in topk:
                correct[k] += torch.sum(torch.any(predictions[:k], dim = 0)).item() 

    results = {f"top{k}": correct[k] / dataloader.num_samples for k in topk}
    logging.info("Finished testing")

    return results

def evaluate(epoch, model, processor, data, options):
    metrics = {}
    
    if(options.master):
        if(data["validation"] is not None or data["test"] is not None):
            if(epoch == 0):
                logging.info(f"Base evaluation")
            else:
                logging.info(f"Epoch {epoch} evaluation")

        if(data["validation"] is not None): metrics.update(get_validation_metrics(model, data["validation"], options))
        if(data["test"] is not None): metrics.update(get_test_metrics(model, processor, data["test"], options))

        if(metrics):
            logging.info("Results")
            for key, value in metrics.items():
                logging.info(f"{key}: {value:.4f}")

            if(options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"evaluation/{key}": value, "epoch": epoch})

    return metrics