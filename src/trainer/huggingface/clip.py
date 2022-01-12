from transformers import CLIPTextConfig, CLIPVisionConfig, CLIPConfig, CLIPModel, CLIPProcessor

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

class Processor:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def process_text(self, text):
        return self.processor.tokenizer(text, return_tensors = "pt", padding = True, truncation = True)
    
    def process_image(self, image):
        return self.processor.feature_extractor(image.convert("RGB"), return_tensors = "pt").pixel_values[0]

def load(pretrained = False, **kwargs):
    # use pretrained model from OpenAI (only supports clip-vit-base-patch32 architecture): by huggingface
    if(pretrained):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    else:
        text_config_dict = vars(CLIPTextConfig())
        vision_config_dict = vars(CLIPVisionConfig())
        config = CLIPConfig(text_config_dict = text_config_dict, vision_config_dict = vision_config_dict)
        model = CLIPModel(config = config)

    convert_models_to_fp32(model)
    processor = Processor()
        
    return model, processor