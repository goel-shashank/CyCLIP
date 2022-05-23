import torch
import pickle
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
    
def cosine_scheduler(optimizer, base_lr, num_warmup_steps, total_steps):
    def _scheduler(current_step):
        if(current_step < num_warmup_steps):
            lr = base_lr * (current_step + 1) / num_warmup_steps
        else:
            n = current_step - num_warmup_steps
            d = total_steps - num_warmup_steps
            lr = 0.5 * (1 + np.cos(np.pi * n / d)) * base_lr

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
    return _scheduler
    
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

def get_dataloader(options, train):
    data = pickle.load(open(options.train_embeddings if train else options.test_embeddings, "rb"))
    image_embeddings, labels = torch.tensor(data["un_image_embeddings"]), torch.tensor(data["labels"])  
    dataset = torch.utils.data.TensorDataset(image_embeddings, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.batch_size, shuffle = train, worker_init_fn = options.worker_init_fn, generator = options.generator)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)
    return dataloader

def run(options):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataloader = get_dataloader(options, train = True)
    test_dataloader = get_dataloader(options, train = False)
        
    input_dim = 1024
    if(options.data_type == "Caltech101"):
        output_dim = 102
    elif(options.data_type == "CIFAR10"):
        output_dim = 10
    elif(options.data_type == "CIFAR100"):
        output_dim = 100
    elif(options.data_type == "DTD"):
        output_dim = 47
    elif(options.data_type == "FGVCAircraft"):
        output_dim = 100
    elif(options.data_type == "Flowers102"):
        output_dim = 102
    elif(options.data_type == "Food101"):
        output_dim = 101
    elif(options.data_type == "GTSRB"):
        output_dim = 43
    elif(options.data_type == "ImageNet1K"):
        output_dim = 1000
    elif(options.data_type == "OxfordIIITPet"):
        output_dim = 37
    elif(options.data_type == "RenderedSST2"):
        output_dim = 2
    elif(options.data_type == "StanfordCars"):
        output_dim = 196
    elif(options.data_type == "STL10"):
        output_dim = 10
    elif(options.data_type == "SVHN"):
        output_dim = 10

    classifier = LogisticRegression(input_dim = input_dim, output_dim = output_dim).to(device)
    optimizer = optim.AdamW([{"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" in name) and parameter.requires_grad)], "weight_decay": 0}, {"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" not in name) and parameter.requires_grad)], "weight_decay": options.weight_decay}])
    scheduler = cosine_scheduler(optimizer, options.lr, 0, len(train_dataloader) * options.num_epochs)
    criterion = nn.CrossEntropyLoss().to(device)
    
    classifier.train()

    bar = tqdm(range(options.num_epochs), leave = True)
    for epoch in bar:
        for index, (image_embedding, label) in enumerate(train_dataloader):
            step = len(train_dataloader) * epoch + index
            scheduler(step)
            image_embedding, label = image_embedding.to(device), label.to(device)
            logits = classifier(image_embedding)
            optimizer.zero_grad()
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
        bar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})

    classifier.eval()
    
    with torch.no_grad():
        correct = 0
        for image_embedding, label in tqdm(test_dataloader, leave = True):
            image_embedding, label = image_embedding.to(device), label.to(device)
            logits = classifier(image_embedding)
            prediction = torch.argmax(logits, dim = 1)
            correct += torch.sum(prediction == label).item()
    
    print(correct / test_dataloader.num_samples * 100.0)
         
if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type = str, default = "CIFAR10", help = "Data type")
    parser.add_argument("--train_embeddings", type = str, default = "analysis/embeddings/clip/CIFAR10.train.pkl", help = "Input train embeddings file")
    parser.add_argument("--test_embeddings", type = str, default = "analysis/embeddings/clip/CIFAR10.test.pkl", help = "Input test embeddings file")
    parser.add_argument("--lr", type = float, default = 0.005, help = "Learning rate")
    parser.add_argument("--batch_size", type = int, default = 16, help = "Batch size")
    parser.add_argument("--num_epochs", type = int, default = 32, help = "Num epochs")
    parser.add_argument("--weight_decay", type = float, default = 0.01, help = "Weight decay")
    parser.add_argument("--seed", type = int, default = 0, help = "Seed")
    options = parser.parse_args()
    
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    torch.backends.cudnn.deterministic = True
    
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)
        
    generator = torch.Generator()
    generator.manual_seed(options.seed)
    
    options.worker_init_fn = worker_init_fn
    options.generator = generator
     
    run(options)