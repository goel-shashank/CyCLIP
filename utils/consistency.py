import torch
import pickle
import argparse

def run(options):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        train_data = pickle.load(open(options.train_embeddings, "rb"))
        test_data = pickle.load(open(options.test_embeddings, "rb"))

        train_image_embeddings, train_labels = torch.tensor(train_data["image_embeddings"]).to(device), torch.tensor(train_data["labels"]).to(device)        
        test_image_embeddings, test_text_embeddings = torch.tensor(test_data["image_embeddings"]), torch.tensor(test_data["text_embeddings"]).to(device)    
        
        batch_size = 32
        
        test_dataset = torch.utils.data.TensorDataset(test_image_embeddings)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size)
        
        consistency = 0

        for test_batch in test_dataloader:
            test_image_embedding = test_batch[0].to(device)
            predictions_text = torch.argmax(test_image_embedding @ test_text_embeddings.t(), 1)
            predictions_image = train_labels[torch.argmax(test_image_embedding @ train_image_embeddings.t(), 1)]
            consistency += torch.sum(predictions_text == predictions_image).cpu().item()
        
        consistency /= len(test_dataset)
        
        print(f"Consistency: {consistency}")
        
if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_embeddings", dest = "train_embeddings", type = str, default = "analysis/embeddings/cyclip/ImageNet1K.train50000.pkl", help = "Input train embeddings file")
    parser.add_argument("--test_embeddings", dest = "test_embeddings", type = str, default = "analysis/embeddings/cyclip/ImageNet1K.validation.pkl", help = "Input test embeddings file")
    options = parser.parse_args()
    run(options)