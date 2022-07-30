import os
import shutil
import json
import pandas as pd 
from tqdm import tqdm

def move_files(images, split):

    for image in tqdm(images):
        current_loc = os.path.join(root, 'val2014', image)
        dest_loc = os.path.join(root, split) 
        shutil.move(current_loc, dest_loc)    

if __name__ == "__main__":

    root = './data/MSCOCO'

    with open('./data/karpathy_splits/dataset_coco.json') as f:
        dataset = json.load(f)
    test = list(filter(lambda x: x['split'] == 'test', dataset['images']))
    test_images = list(map(lambda x: x['filename'], test))
    test_captions = list(map(lambda x: x['sentences'][0]['raw'], test))
    # list_of_all_images = os.listdir(os.path.join(root, 'val2014'))
    # move_files(test_images, 'test')

    test_images = list(map(lambda x: f'test/{x}', test_images))
    data = {'image': test_images,
            'caption': test_captions}
    df = pd.DataFrame(data)
    df.to_csv(f'{root}/mscoco_test.csv')


