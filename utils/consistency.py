import torch
import pickle
import argparse
from tqdm import tqdm

def run(options):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        train_data = pickle.load(open(options.train_embeddings, "rb"))
        test_data = pickle.load(open(options.test_embeddings, "rb"))

        train_image_embeddings, train_labels = torch.tensor(train_data["image_embeddings"]).to(device), torch.tensor(train_data["labels"]).to(device)        
        test_image_embeddings, test_text_embeddings, test_labels = torch.tensor(test_data["image_embeddings"]), torch.tensor(test_data["text_embeddings"]).to(device), torch.tensor(test_data["labels"])  
                
        test_dataset = torch.utils.data.TensorDataset(test_image_embeddings, test_labels)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = options.batch_size)
        
        accuracy = {k: 0 for k in options.k}
        consistency = {k: 0 for k in options.k}
        
        for test_batch in tqdm(test_dataloader):
            test_image_embedding, test_label = test_batch[0].to(device), test_batch[1].to(device)
            zeroshot_labels = torch.argmax(test_image_embedding @ test_text_embeddings.t(), 1)
            predicted_labels = train_labels[torch.topk(test_image_embedding @ train_image_embeddings.t(), k = max(options.k), dim = 1)[1]]
            
            for i in range(len(test_image_embedding)):
                for k in options.k:
                    temp = k
                    while(1):
                        values, counts = predicted_labels[i, :temp].unique(return_counts = True)
                        if(torch.sum(counts == counts.max()) == 1):
                            break
                        temp -= 1
                        
                    predicted_label = values[counts.argmax()]
                    
                    accuracy[k] += int((predicted_label == test_label[i]).cpu().item())
                    consistency[k] += int((zeroshot_labels[i] == predicted_label).cpu().item())

        for k in options.k:
            accuracy[k] /= (len(test_dataset) * 0.01)
            consistency[k] /= (len(test_dataset) * 0.01)
        
        print(f"Accuracy: {accuracy}")
        print(f"Consistency: {consistency}")
        
if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_embeddings", type = str, default = "analysis/embeddings/clip/ImageNet1K.train50000.pkl", help = "Input train embeddings file")
    parser.add_argument("--test_embeddings", type = str, default = "analysis/embeddings/clip/ImageNet1K.validation.pkl", help = "Input test embeddings file")
    parser.add_argument("--batch_size", type = int, default = 32, help = "Batch size")
    parser.add_argument("--k", nargs = "+", default = [1, 3, 5, 10, 30])
    options = parser.parse_args()
    run(options)