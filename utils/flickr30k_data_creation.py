import os
import shutil
import json
import pandas as pd 
from tqdm import tqdm

def move_files(images, split):

    for image in tqdm(images):
        current_loc = os.path.join(root, 'images', image)
        dest_loc = os.path.join(root, split) 
        shutil.move(current_loc, dest_loc)

def get_data(split):
    
    split_images = []
    split_comments = []

    df = pd.read_csv(os.path.join(root, 'results.csv'), "|")
    imgs = os.listdir(os.path.join(root, split))
    for img in tqdm(imgs):
        df_ = df.where(df['image_name'] == img).dropna()
        locations = list(map(lambda x: f'{split}/{x}' ,df_['image_name'].tolist()))
        split_images = split_images + locations
        comments = df_[' comment'].tolist()
        comments = list(map(lambda x: x[1:], comments))
        split_comments = split_comments + comments
    return split_images, split_comments        

if __name__ == "__main__":

    root = './data/flickr30k'
    file_management = False
    file_creation = True

    if file_management:
        with open(os.path.join(root, 'dataset.json')) as f:
            dataset = json.load(f)
        train_images = list(filter(lambda x: x['split'] == 'train', dataset['images']))
        val_images = list(filter(lambda x: x['split'] == 'val', dataset['images']))
        test_images = list(filter(lambda x: x['split'] == 'test', dataset['images']))
        train_images, val_images, test_images = list(map(lambda li: list(map(lambda x: x['filename'], li)), [train_images, val_images, test_images]))
        assert(len(test_images) == 1000)
        list_of_all_images = os.listdir(os.path.join(root, 'images'))
        move_files(train_images, 'train')
        move_files(val_images, 'validation')
        move_files(test_images, 'test')
    
    if file_creation:
        # train_images, train_captions = get_data(os.path.join(root, 'train'))
        # validation_images, validation_captions = get_data(os.path.join(root, 'validation'))
        test_images, test_captions = get_data('test')

        # images = train_images + validation_images + test_images
        # caps = train_captions + validation_captions + test_captions

        data = {'image': test_images,
                'caption': test_captions}
        dt = pd.DataFrame(data)
        dt.to_csv(f'{root}/flickr30k.csv')