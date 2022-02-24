import os
import argparse
import warnings
import torch
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from PIL import Image, ImageFile
from pkgs.openai.clip import load


ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("--input_train_dir", type = str, default = "./data/CC3M/validation", help = "Input data file directory")
parser.add_argument("--input_train_file", type = str, default = "validation.csv", help = "Input data file name")
parser.add_argument("--imagenet_dir", type = str, default = "./data/Imagenet/validation", help = "Imagenet directory")
parser.add_argument("--save_dir", type = str, default = "/u/home/h/hbansal/scratch", help = "directory where the PCA will be stored")

parser.add_argument("--embeddings_train_file", type = str, default = "saved_train_embeddings.pkl", help = "embeddings train file")
parser.add_argument("--use_saved_train_embeddings", action = "store_true", default = False, help = "Use saved train embeddings")
parser.add_argument("--embeddings_imagenet_file", type = str, default = "saved_imagenet_embeddings.pkl", help = "embeddings imagenet file")
parser.add_argument("--use_saved_imagenet_embeddings", action = "store_true", default = False, help = "Use saved imagenet embeddings")

parser.add_argument("--data_size_pca", type = int, default = 5120, help = "Number of samples on which pca will be applied")
parser.add_argument("--checkpoint", default = None, type = str, help = "Path to saved checkpoint")
parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")
parser.add_argument("--experiment_name", type = str, default = "baseline", help = "baseline or symmetric")
parser.add_argument("--pretrained", default = False, action = "store_true", help = "Use the OpenAI pretrained models")
parser.add_argument("--batch_size", type = int, default = 64, help = "Batch Size")

options = parser.parse_args()

class ImageNetDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        df = pd.read_csv(os.path.join(root, "labels.csv"))
        self.images = df["image"]
        self.labels = df["label"]
        indices = [i for i in range(len(self.labels)) if self.labels[i] in [224, 449, 452, 453, 618, 646]]
        self.images = [self.images[i] for i in range(len(self.images)) if i in indices]
        self.labels = [self.labels[i] for i in range(len(self.labels)) if i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.transform(Image.open(os.path.join(self.root, self.images[idx])))
        label = self.labels[idx]
        return image, label

def evaluate_imagenet_embeddings(model, processor, dataloader, imagenet_dir):
    
    labels = []
    embeddings = []

    with torch.no_grad():   
        config = eval(open(f"{imagenet_dir}/classes.py", "r").read())
        classes, templates = np.array(config["classes"]), config["templates"]
        text_embeddings = []
        for c in tqdm(classes):
            # text = [template(c) for template in templates]
            text = ['a photo of {c}']
            text_tokens = processor.process_text(text)
            text_input_ids, text_attention_mask = text_tokens["input_ids"].to(device), text_tokens["attention_mask"].to(device) 
            text_embedding = model.get_text_features(input_ids = text_input_ids, attention_mask = text_attention_mask)
            text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
            text_embedding = text_embedding.mean(dim = 0)
            text_embedding /= text_embedding.norm()
            text_embeddings.append(text_embedding)
        text_embeddings = torch.stack(text_embeddings, dim = 1).to(device).t()

        for image, label in tqdm(dataloader):
            image, label = image.to(device), label.to(device)
            
            text_embedding = text_embeddings[label]

            image_embedding = model.get_image_features(image)
            image_embedding /= image_embedding.norm(dim = -1, keepdim = True)

            labels.append(classes[label.cpu().detach().numpy()])
            embeddings.append(text_embedding)
            embeddings.append(image_embedding)

        labels = np.concatenate(labels)
        embeddings = torch.cat(embeddings)
        d = {'labels': labels, 'embeddings': embeddings}

        return d


def get_imagenet_embeddings(model, processor, imagenet_dir):

    dataset = ImageNetDataset(root = imagenet_dir, transform = processor.process_image)
    dataloader = DataLoader(dataset, batch_size = options.batch_size, shuffle = False, pin_memory = True, drop_last = True)
    embeddings_dict = evaluate_imagenet_embeddings(model, processor, dataloader, imagenet_dir)
    return embeddings_dict

def batch(iterable, n = 1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_embeddings_in_batch(model, all_texts, all_images, root, processor, batch_size = 1024, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose = False):
    encodings = []
    with torch.no_grad():
        score = 0

        dataloader_texts = list(batch(all_texts, batch_size))
        dataloader_images = list(batch(all_images, batch_size))

        bar = zip(dataloader_texts, dataloader_images)
        if(verbose): 
            print("Evaluating..")
            bar = tqdm(bar, total = len(dataloader_texts))
        
        for texts, images in bar:
            captions = processor.process_text(texts)
            input_ids = captions['input_ids'].to(device)
            attention_mask = captions['attention_mask'].to(device)
            pixel_values = torch.tensor(np.stack([processor.process_image(Image.open(os.path.join(root, image)).convert("RGB")) for image in images])).to(device)
            
            text_embedding = model.get_text_features(input_ids = input_ids, attention_mask = attention_mask)
            image_embedding = model.get_image_features(pixel_values)

            text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
            image_embedding /= image_embedding.norm(dim = -1, keepdim = True)
            
            encodings.append(text_embedding)
            encodings.append(image_embedding)

        return encodings

def get_embeddings(model, processor, root, images, captions, verbose = True):

    all_embeddings = get_embeddings_in_batch(model, captions, images, root = root, processor = processor, batch_size = options.batch_size, device = device, verbose = verbose)
    all_embeddings = torch.cat(all_embeddings)

    return all_embeddings

if __name__ == '__main__':

    root = options.input_train_dir
    fname = f'{root}/{options.input_train_file}'

    df = pd.read_csv(fname)
    df = df[:options.data_size_pca]

    images   = df['image'].tolist()
    captions = df['caption'].tolist()
    saved_embeddings_train_file = f'{options.save_dir}/{options.experiment_name}_{options.embeddings_train_file}'
    saved_embeddings_imagenet_file = f'{options.save_dir}/{options.experiment_name}_{options.embeddings_imagenet_file}'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, processor = load(name = options.model_name, pretrained = options.pretrained)
    model = model.to(device)
    if(options.checkpoint is not None):
        if(os.path.isfile(options.checkpoint)):
            checkpoint = torch.load(options.checkpoint, map_location = device)
            state_dict = checkpoint['state_dict']
            if(next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
            print(f'Loaded checkpoint {options.checkpoint}')
        else:
            print(f'No checkpoint found at {options.checkpoint}')

    model.eval()

    if options.use_saved_train_embeddings:
        with open(saved_embeddings_train_file, 'rb') as handle:
            d = pickle.load(handle)
            train_embeddings = d['embeddings']
    else:
        train_embeddings = get_embeddings(model, processor, root, images, captions)
        d = {'caption-images': list(zip(captions, images)), 'embeddings': train_embeddings}
        with open(saved_embeddings_train_file, 'wb') as handle:
            pickle.dump(d, handle)  
    print('Training Embeddings Loaded!')

    if options.use_saved_imagenet_embeddings:
        with open(saved_embeddings_imagenet_file, 'rb') as handle:
            imagenet_d = pickle.load(handle)
    else:
        imagenet_d = get_imagenet_embeddings(model, processor, options.imagenet_dir)
        with open(saved_embeddings_imagenet_file, 'wb') as handle:
            pickle.dump(imagenet_d, handle)  
    print('Imagenet Embeddings Loaded!')

    imagenet_embeddings = imagenet_d['embeddings'].cpu().detach().numpy()
    train_embeddings = train_embeddings.cpu().detach().numpy()
    sc = StandardScaler()
    train_embeddings = sc.fit_transform(train_embeddings)
    imagenet_embeddings = sc.transform(imagenet_embeddings)

    pca = PCA(n_components = 2)

    train_embeds = pca.fit_transform(train_embeddings)
    imagenet_embeds = pca.transform(imagenet_embeddings)

    with open(f'{options.save_dir}/pca_{options.experiment_name}_embeddings.pkl', 'wb') as handle:
        data = {'train_embeddings': train_embeds,
                'train_captions_images': d['caption-images'],
                'imagenet_embeddings': imagenet_embeds,
                'imagenet_labels': imagenet_d['labels']
                }
        pickle.dump(data, handle)
        print('PCA Saved!!')


