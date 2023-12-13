import torch
from datetime import datetime
import os

weights_folder = "weigths"

def save_checkpoint(model, epoch, metrics, optimizer):

    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)

    file_name = f'{model.__class__.__name__}_{epoch}_epochs.pt'

    current_time = datetime.now()

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'date': current_time.strftime("%Y-%m-%d %H:%M:%S")
        }, os.path.join(weights_folder, file_name))
    
    print(f"model saved to : {os.path.join(weights_folder, file_name)}")

def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint