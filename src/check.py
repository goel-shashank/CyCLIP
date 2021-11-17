import sys
import argparse
import pandas as pd
from models import *
from utils import *

def process(options):
    config = configreader.read(options.config)

    device = "cuda:0" if(torch.cuda.is_available() and config["use_gpu"]) else "cpu"

    names = []
    models = []
    for key, value in config["Modality"].items():
        value["embedding_dim"] = config["embedding_dim"]
        names.append(key)
        models.append(eval(f'{value["class"]}(**value).to(device)'))
    
    alphas = pd.read_csv(config["MultiModalLoss"]["alphas"], index_col = 0)
    alphas = torch.Tensor(alphas[names].loc[names].values)

    multimodal_model = MultiModalModel(embedding_dim = config["embedding_dim"], models = models)
    multimodal_loss = MultiModalLoss(temperature = config["MultiModalLoss"]["temperature"], alphas = alphas)
    
    # Test
    text_input = torch.randint(0, config["Modality"]["text"]["vocab_size"], (64, config["Modality"]["text"]["context_size"])).to(device)
    image_input = torch.randn((64, config["Modality"]["image"]["in_channels"], config["Modality"]["image"]["input_dim"], config["Modality"]["image"]["input_dim"])).to(device)

    inputs = [text_input, image_input]
    outputs = multimodal_model(inputs)
    
    loss = multimodal_loss(outputs)

    print(loss)

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser(description = "Multimodal Learner")
    parser.add_argument("-c", "--config", dest = "config", type = str, default = f"configs/config.yaml")
    options = parser.parse_args(sys.argv[1:])
    
    process(options)