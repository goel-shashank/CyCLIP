from transformers import CLIPTextConfig, CLIPVisionConfig, CLIPConfig, CLIPModel, CLIPProcessor

class Processor:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def text(self, text):
        return self.processor.tokenizer(text, return_tensors = "pt", padding = True, truncation = True)
    
    def image(self, image):
        return self.processor.feature_extractor(image.convert("RGB"), return_tensors = "pt").pixel_values[0]

def load(pretrained = False):
    # use pretrained model from openai (only supports clip-vit-base-patch32 architecture)
    if(pretrained):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    else:
        text_config_dict = vars(CLIPTextConfig())
        vision_config_dict = vars(CLIPVisionConfig())
        config = CLIPConfig(text_config_dict = text_config_dict, vision_config_dict = vision_config_dict)
        model = CLIPModel(config = config)

    processor = Processor()
        
    return model, processor