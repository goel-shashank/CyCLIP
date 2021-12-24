import json
import os, shutil
import random
from PIL import Image
from tqdm import tqdm
import pandas as pd
import jax
from transformers import FlaxVisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from huggingface_hub import hf_hub_download

'''
Requirements Installation

pip install transformers datasets
pip install huggingface-hub
pip install flax

'''


model_dir = './i2t_models/'
os.makedirs(model_dir, exist_ok=True)
files_to_download = [
			    "config.json",
			    "flax_model.msgpack",
			    "merges.txt",
			    "special_tokens_map.json",
			    "tokenizer.json",
			    "tokenizer_config.json",
			    "vocab.json",
			    "preprocessor_config.json",
			    ]

for fn in files_to_download:
	if not os.path.exists(f'{model_dir}{fn}'):
	    file_path = hf_hub_download("ydshieh/vit-gpt2-coco-en-ckpts", f"ckpt_epoch_3_step_6900/{fn}")
	    shutil.copyfile(file_path, os.path.join(model_dir, fn))

model = FlaxVisionEncoderDecoderModel.from_pretrained(model_dir)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
gen_kwargs = {"max_length": 16, "num_beams": 4}

@jax.jit
def generate(pixel_values):
	output_ids = model.generate(pixel_values, **gen_kwargs).sequences
	return output_ids

def predict(image):
	if image.mode != 'RGB':
		image = image.convert(mode = 'RGB')

	pixel_values = feature_extractor(images = image, return_tensors = 'np').pixel_values
	output_ids = generate(pixel_values)
	preds = tokenizer.batch_decode(output_ids, skip_special_tokens = True)
	preds = [pred.strip() for pred in preds]
	return preds[0]


def generate_and_save(root, image_locs):
	'''
		INPUT
			image_locs: [loc_1, loc_2, ..., loc_n] where loc_i is the location of an image
		OUTPUT
			csv w two cols: image_locs | captions
	'''
	dir_path = f'{root}/generated_captions/'
	os.makedirs(dir_path, exist_ok = True)

	captions = []
	for index, image_loc in tqdm(enumerate(image_locs)):
		image = Image.open(image_loc)
		captions.append(predict(image))

	data = {'image_locs': image_locs,
			'captions': captions
			}

	df = pd.DataFrame(data)
	df.to_csv(f'{dir_path}fake_captions.csv')

	print('Caption generation done!')



