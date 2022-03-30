import argparse
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from PIL import Image, ImageFile
from pkgs.openai.clip import load

ImageFile.LOAD_TRUNCATED_IMAGES = True

def batch(iterable, n = 1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

@torch.no_grad()
def itm_eval(text_embeddings, image_embeddings):

    # sim_matrix_i2t = image_embeddings @ text_embeddings.t()
    # sim_matrix_t2i = text_embeddings @ image_embeddings.t()

    ## Image -> Text
    # ranks = np.zeros(len(sim_matrix_i2t))
    ranks = np.zeros(len(image_embeddings))

    for index in range(0, len(image_embeddings), 5):
        scores = image_embeddings[index] @ text_embeddings.t()
        # scores = sim_matrix_i2t[index]
        li = np.argsort(scores.detach().cpu().numpy())[::-1]
        for i in range(len(li)):
            if index <= li[i] and li[i] <= index + 4:
                rank = i
                break
        ranks[index] = rank
    
        # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    ## Image -> Text
    ranks = np.zeros(len(text_embeddings))
    for index in range(len(text_embeddings)):
        scores = text_embeddings[index] @ image_embeddings.t()
    # for index, scores in tqdm(enumerate(sim_matrix_t2i)):
        scores = scores[::5]
        li = np.argsort(scores.detach().cpu().numpy())[::-1]
        for i in range(len(li)):
            if li[i] == index//5:
                rank = i
                break
        ranks[index] = rank
    
    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    
    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                'txt_r5': tr5,
                'txt_r10': tr10,
                'txt_r_mean': tr_mean,
                'img_r1': ir1,
                'img_r5': ir5,
                'img_r10': ir10,
                'img_r_mean': ir_mean,
                'r_mean': r_mean}

    return eval_result

def get_all_embeddings(model, all_texts, all_images, root, processor, batch_size = 1024, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose = False):
    text_embeddings = []
    image_embeddings = []

    with torch.no_grad():
        score = 0

        dataloader_texts = list(batch(all_texts, batch_size))
        dataloader_images = list(batch(all_images, batch_size))

        bar = zip(dataloader_texts, dataloader_images)
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

            text_embeddings.append(text_embedding)
            image_embeddings.append(image_embedding)

        text_embeddings = torch.cat(text_embeddings)
        image_embeddings = torch.cat(image_embeddings)
        return text_embeddings, image_embeddings

def evaluate(input_file):

    if options.use_saved_embeddings:
        with open(options.embeddings_file, 'rb') as f:
            text_embeds, image_embeds = pickle.load(f)
        print('Embeddings Loaded!')
    else:
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
        print(input_file)
        root = os.path.dirname(input_file)
        df = pd.read_csv(input_file, sep = options.delimiter)

        captions = df[options.caption_key].tolist()
        images = df[options.image_key].tolist()

        text_embeds, image_embeds = get_all_embeddings(model, captions, images, root = root, processor = processor, batch_size = options.batch_size, device = device)

        with open(options.embeddings_file, 'wb') as f:
            pickle.dump((text_embeds, image_embeds), f)
        print('Embedding dumped!')

    result = itm_eval(text_embeds, image_embeds)

    print(result)

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_file", type = str, default = None, help = "Input file")
    # parser.add_argument("-o,--output_file", dest = "output_file", type = str, default = None, help = "Output file")
    # parser.add_argument("-q,--quiet", dest = "quiet" , default = False, action = "store_true", help = "Silent output")
    parser.add_argument("--batch_size", type = int, default = 128, help = "Batch Size")
    parser.add_argument("--delimiter", type = str, default = ",", help = "Input file delimiter")
    parser.add_argument("--image_key", type = str, default = "image", help = "Image column name")
    parser.add_argument("--caption_key", type = str, default = "caption", help = "Caption column name")
    parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")
    
    parser.add_argument("--checkpoint", default = None, type = str, help = "Path to checkpoint to resume training")
    parser.add_argument("--pretrained", default = False, action = "store_true", help = "Use the OpenAI pretrained models")
    parser.add_argument("--use_saved_embeddings", action = "store_true", default = False, help = "Use saved embeddings")
    parser.add_argument("--embeddings_file", type = str, default = "embeddings.pkl", help = "embedding file")


    options = parser.parse_args()
    evaluate(options.input_file)
