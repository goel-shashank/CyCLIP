import os
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageNetV2Dataset(Dataset):
    '''
        Dummy Run
            import clip
            model, preprocess = clip.load("ViT-B/32", download_root = '.cache')
            ds = ImageNetV2Dataset('/u/scratch/s/shashank/project/Multimodal-Representation-Learning/data/ImageNetV2-matched-frequency/ImageNetV2.csv', preprocess)
    '''
    def __init__(self, imagenetv2_file, transform):
        self.imagenetv2_file = imagenetv2_file 
        self.dir = os.path.dirname(self.imagenetv2_file)
        self.df = pd.read_csv(self.imagenetv2_file)
        self.images = self.df['image'].tolist()
        self.labels = self.df['classes'].tolist()
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.transform(Image.open(os.path.join(self.dir, self.images[idx])))
        label = self.labels[idx]
        return image, label

class ImageNetSketch(Dataset):
    '''
        Dummy Run
            import clip
            model, preprocess = clip.load("ViT-B/32", download_root = '.cache')
            ds = ImageNetSketch('/u/scratch/s/shashank/project/Multimodal-Representation-Learning/data/ImageNet-Sketch/ImageNet_Sketch.csv', preprocess)
    '''
    def __init__(self, imagenet_sketch_file, transform):
        self.imagenet_sketch_file = imagenet_sketch_file 
        self.dir = os.path.dirname(self.imagenet_sketch_file)
        self.df = pd.read_csv(self.imagenet_sketch_file)
        self.images = self.df['locations'].tolist()
        self.labels = self.df['labels'].tolist()
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.transform(Image.open(os.path.join(self.dir, self.images[idx])))
        label = self.labels[idx]
        return image, label
