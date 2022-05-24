import torch
import pickle
import argparse
from tqdm import tqdm

def run(options):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        data = pickle.load(open(options.embeddings, "rb"))
        image_embeddings, text_embeddings, labels = torch.tensor(data["image_embeddings"]), torch.tensor(data["text_embeddings"]).to(device), torch.tensor(data["labels"])  
                
        dataset = torch.utils.data.TensorDataset(image_embeddings, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.batch_size)
        
        correct = {k: 0 for k in options.k}
                
        for image_embedding, label in tqdm(dataloader):
            image_embedding, label = image_embedding.to(device), label.to(device)
            logits = (image_embedding @ text_embeddings.t())
            ranks = logits.topk(max(options.k), 1)[1].T
            predictions = ranks == label
            for k in options.k:
                correct[k] += torch.sum(torch.any(predictions[:k], dim = 0)).item() 

        for k in options.k:
            print(f"Zeroshot top {k}: {correct[k] / len(dataset) * 100.0}")    
        
if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type = str, default = "analysis/embeddings/clip/ImageNet1K.validation.pkl", help = "Input test embeddings file")
    parser.add_argument("--batch_size", type = int, default = 32, help = "Batch size")
    parser.add_argument("--k", nargs = "+", default = [1, 3, 5])
    options = parser.parse_args()
    run(options)