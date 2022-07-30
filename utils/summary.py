import torch
import warnings
from pkgs.openai.clip import load

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, processor = load(name = "RN50", pretrained = False)
model.to(device)

print(sum(parameter.numel() for parameter in model.visual.parameters() if parameter.requires_grad))
print(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) - sum(parameter.numel() for parameter in model.visual.parameters() if parameter.requires_grad) - 1)