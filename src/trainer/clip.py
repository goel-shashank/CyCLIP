from transformers import CLIPTextConfig, CLIPVisionConfig, CLIPConfig, CLIPModel, CLIPProcessor

def load(pretrained = False):
    # use pretrained model from openai (only supports clip-vit-base-patch32 architecture)
    if(pretrained):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    else:
        text_config_dict = vars(CLIPTextConfig())
        vision_config_dict = vars(CLIPVisionConfig())
        config = CLIPConfig(text_config_dict = text_config_dict, vision_config_dict = vision_config_dict)
        model = CLIPModel(config = config)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
    return model, processor