import argparse
import torch
from datasets import load_dataset
from pkgs.openai.clip import load as load_model

parser = argparse.ArgumentParser()

parser.add_argument("--checkpoint", default = None, type = str, help = "Path to checkpoint to resume training")

options = parser.parse_args()

device = 'cuda:1'

token = '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'
winoground = load_dataset("facebook/winoground", use_auth_token=token)["test"]

def get_inputs(index, processor, image_number = 0, text_number = 0):
        captions = processor.process_text(winoground[index][f'caption_{text_number}'])
        pixel_values = processor.process_image(winoground[index][f"image_{image_number}"].convert("RGB"))
        return captions['input_ids'].to(device).unsqueeze(0), captions['attention_mask'].to(device).unsqueeze(0), pixel_values.to(device).unsqueeze(0)

model, processor = load_model(name = 'RN50', pretrained = False)
model.to(device)

state_dict = torch.load(options.checkpoint, map_location = device)["state_dict"]
if(next(iter(state_dict.items()))[0].startswith("module")):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()  

input_c0_i0 = get_inputs(155, processor, image_number = 0, text_number = 0)
input_c1_i0 = get_inputs(155, processor, image_number = 0, text_number = 1)
input_c0_i1 = get_inputs(155, processor, image_number = 1, text_number = 0)
input_c1_i1 = get_inputs(155, processor, image_number = 1, text_number = 1)
output_c0_i0 = model(input_ids = input_c0_i0[0], attention_mask = input_c0_i0[1], pixel_values = input_c0_i0[2])
output_c0_i1 = model(input_ids = input_c0_i1[0], attention_mask = input_c0_i1[1], pixel_values = input_c0_i1[2])
output_c1_i0 = model(input_ids = input_c1_i0[0], attention_mask = input_c1_i0[1], pixel_values = input_c1_i0[2])
output_c1_i1 = model(input_ids = input_c1_i1[0], attention_mask = input_c1_i1[1], pixel_values = input_c1_i1[2])

clip_score_c0_i0 = output_c0_i0.logits_per_image.item()
clip_score_c1_i0 = output_c1_i0.logits_per_image.item()
clip_score_c0_i1 = output_c0_i1.logits_per_image.item()
clip_score_c1_i1 = output_c1_i1.logits_per_image.item()

print("CLIP image-text match scores:")
print("image_0, caption_0:", clip_score_c0_i0)
print("image_0, caption_1:", clip_score_c1_i0)
print("image_1, caption_0:", clip_score_c0_i1)
print("image_1, caption_1:", clip_score_c1_i1)
