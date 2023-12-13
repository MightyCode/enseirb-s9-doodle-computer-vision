import torch

def device_section():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    return device