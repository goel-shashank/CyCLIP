import torch
import pickle
import argparse
from tqdm import tqdm

def run(options):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        data = pickle.load(open(options.embeddings, "rb"))

        image_embeddings, text_embeddings, labels = torch.tensor(data["image_embeddings"]).to(device), torch.tensor(data["text_embeddings"]).to(device), torch.tensor(data["labels"]).to(device)
        text_embeddings = text_embeddings[labels]
        
        alignment = (image_embeddings * text_embeddings).sum(1).mean(0)

        batch_size = 32
        
        dataset = torch.utils.data.TensorDataset(image_embeddings)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)
        
        uniformity = torch.zeros([]).to(device)
        for index, batch in enumerate(dataloader):
            image_embedding = batch[0]
            cross = image_embedding @ text_embeddings.t()
            uniformity += (-cross).exp().sum() - (-cross.diag(index * batch_size)).exp().sum()
        uniformity /= (len(image_embeddings) * (len(image_embeddings) - 1))
        uniformity = uniformity.log()
        
        print(f"Alignment: {alignment.cpu().item()}")
        print(f"Uniformity: {uniformity.cpu().item()}")
        
if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e,--embeddings", dest = "embeddings", type = str, default = "analysis/embeddings/clip/ImageNet1K.validation.pkl", help = "Input file")
    options = parser.parse_args()
    run(options)