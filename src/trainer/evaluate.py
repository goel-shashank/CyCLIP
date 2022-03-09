import wandb
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm    

def get_validation_metrics(model, dataloader, options):
    logging.info("Started validating")

    metrics = {}

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction = "sum").to(options.device)

    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask, pixel_values = batch["input_ids"].to(options.device, non_blocking = True), batch["attention_mask"].to(options.device, non_blocking = True), batch["pixel_values"].to(options.device, non_blocking = True) 
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)
            
            umodel = model.module if(options.distributed) else model

            logits_per_image = umodel.logit_scale.exp() * outputs.image_embeds @ outputs.text_embeds.t()
            logits_per_text = logits_per_image.t()

            target = torch.arange(len(input_ids)).long().to(options.device, non_blocking = True)
            loss = (criterion(logits_per_image, target) + criterion(logits_per_text, target)) / 2

            losses.append(loss)

        loss = sum(losses) / dataloader.num_samples
        metrics["loss"] = loss

    logging.info("Finished validating")

    return metrics

def get_zeroshot_metrics(model, processor, dataloader, options):
    logging.info("Started zeroshot testing")

    model.eval()
    umodel = model.module if(options.distributed) else model

    config = eval(open(f"{options.eval_test_data_dir}/classes.py", "r").read())
    classes, templates = config["classes"], config["templates"]
    with torch.no_grad():
        text_embeddings = []
        for c in tqdm(classes):
            text = [template(c) for template in templates]
            text_tokens = processor.process_text(text)
            text_input_ids, text_attention_mask = text_tokens["input_ids"].to(options.device), text_tokens["attention_mask"].to(options.device) 
            text_embedding = umodel.get_text_features(input_ids = text_input_ids, attention_mask = text_attention_mask)
            text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
            text_embedding = text_embedding.mean(dim = 0)
            text_embedding /= text_embedding.norm()
            text_embeddings.append(text_embedding)
        text_embeddings = torch.stack(text_embeddings, dim = 1).to(options.device)

    with torch.no_grad():
        topk = [1, 3, 5, 10]
        correct = {k: 0 for k in topk}

        for image, label in tqdm(dataloader):
            image, label = image.to(options.device), label.to(options.device)
            image_embedding = umodel.get_image_features(image)

            logits = (image_embedding @ text_embeddings)
            ranks = logits.topk(max(topk), 1)[1].T
            predictions = ranks == label

            for k in topk:
                correct[k] += torch.sum(torch.any(predictions[:k], dim = 0)).item() 

    results = {f"zeroshot_top{k}": correct[k] / dataloader.num_samples for k in topk}
    logging.info("Finished zeroshot testing")

    return results

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs
    
def get_linear_probe_metrics(model, train_dataloader, test_dataloader, options):
    logging.info("Started linear probe testing")

    model.eval()
    umodel = model.module if(options.distributed) else model
    
    input_dim = umodel.text_projection.shape[1]
    
    if(options.eval_data_type in ["CIFAR10", "STL10", "MNIST"]):
        output_dim = 10
    elif(options.eval_data_type == "CIFAR100"):
        output_dim = 100
    elif(options.eval_data_type == "Caltech101"):
        output_dim = 101
    elif(options.eval_data_type in ["Imagenet", "ImagenetV2", "ImagenetSketch"]):
        output_dim = 1000

    classifier = LogisticRegression(input_dim = input_dim, output_dim = output_dim).to(options.device)
    
    optimizer = optim.Adam(classifier.parameters())
    criterion = nn.CrossEntropyLoss().to(options.device)

    pbar = tqdm(range(options.linear_probe_num_epochs))
    for epoch in pbar:
        cbar = tqdm(train_dataloader, leave = False)
        for image, label in cbar:
            image, label = image.to(options.device), label.to(options.device)
            with torch.no_grad():
                image = umodel.get_image_features(image)
            logit = classifier(image)
            optimizer.zero_grad()
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()
            cbar.set_postfix({"loss": loss.item()})
        pbar.set_postfix({"loss": loss.item()})

    classifier.eval()
    
    with torch.no_grad():
        topk = [1, 3, 5, 10]
        correct = {k: 0 for k in topk}

        for image, label in tqdm(test_dataloader):
            image, label = image.to(options.device), label.to(options.device)
            logits = classifier(umodel.get_image_features(image))
            ranks = logits.topk(max(topk), 1)[1].T
            predictions = ranks == label

            for k in topk:
                correct[k] += torch.sum(torch.any(predictions[:k], dim = 0)).item() 

    results = {f"linear_probe_top{k}": correct[k] / test_dataloader.num_samples for k in topk}
    logging.info("Finished linear probe testing")

    return results

def evaluate(epoch, model, processor, data, options):
    metrics = {}
    
    if(options.master):
        if(data["validation"] is not None or data["eval_test"] is not None):
            if(epoch == 0):
                logging.info(f"Base evaluation")
            else:
                logging.info(f"Epoch {epoch} evaluation")

        if(data["validation"] is not None): 
            metrics.update(get_validation_metrics(model, data["validation"], options))
            
        if(data["eval_test"] is not None): 
            metrics.update(get_zeroshot_metrics(model, processor, data["eval_test"], options))
            if(data["eval_train"] is not None):
                metrics.update(get_linear_probe_metrics(model, data["eval_train"], data["eval_test"], options))
    
        if(metrics):
            logging.info("Results")
            for key, value in metrics.items():
                logging.info(f"{key}: {value:.4f}")

            if(options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"evaluation/{key}": value, "epoch": epoch})

    return metrics


class RegularizedWassersteinDistance(nn.Module):
    """
    We find that it is important to ensure that λ is large enough,
    otherwise the projection of the image is excessively blurred.
    In addition to qualitative changes, smaller λ seems to make
    it harder to find Wasserstein adversarial examples, making
    the   radius go up as λ gets smaller. In fact, for λ = (1, 10)
    and almost all of λ = 100, the blurring is so severe that no
    adversarial example can be found.

    In contrast, we find that increasing p for the Wasserstein
    distance used in the cost matrix C seems to make the images
    more “blocky”. Specifically, as p gets higher tested, more
    pixels seem to be moved in larger amounts. This seems to
    counteract the blurring observed for low λ to some degree.
    Naturally, the   radius also grows since the overall cost of
    the transport plan has gone up.

    epsilon = 1 / lambda
    """

    def __init__(self, d, p = 2, epsilon = 0.2, max_iter = 50, tolerance = 1e-6, cost_function = None, device = "cpu"):
        super(RegularizedWassersteinDistance, self).__init__()
        self.d = d
        self.n = d * d
        self.p = p
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.device = device

        self.cost = torch.zeros(self.n, self.n)
        for i, j, k, l in itertools.product(range(self.d), repeat = 4):
            self.cost[i * self.d + j, k * self.d + l] = ((i - k) ** 2 + (j - l) ** 2) ** (self.p / 2) if(cost_function is None) else cost_function((i, j), (k, l))

        self.kernel = torch.exp(-self.cost / self.epsilon) 

        self.cost = self.cost.to(self.device)
        self.kernel = self.kernel.to(self.device)

    def forward(self, x, y):
        b = x.shape[0]

        x = x.reshape(b, -1).to(self.device)
        y = y.reshape(b, -1).to(self.device)

        u = torch.ones((b, self.n)).to(self.device) / self.n
        v = torch.ones((b, self.n)).to(self.device) / self.n

        for _ in range(self.max_iter):
            pu, pv = u, v
          
            u = torch.div(x, v @ self.kernel.T + 1e-8)
            v = torch.div(y, u @ self.kernel + 1e-8) 

            if(((u - pu).abs() + (v - pv).abs()).sum(1).mean() < self.tolerance):
                break
        
        pi = (u.diag_embed() @ self.kernel @ v.diag_embed()) 
        distance = torch.sum(pi * self.cost)
        
        return distance