import torch
from datetime import datetime
import os
import numpy as np


class PytorchUtils:
    WEIGHTS_FOLDER = "weigths"


    @staticmethod
    def tensor_to_numpy(tensor):
        return tensor.cpu().detach().numpy()

    @staticmethod
    def tensor_to_img(tensor, width, height):
        return PytorchUtils.tensor_to_numpy(tensor).reshape((width, height))
    
    @staticmethod
    def numpy_to_tensor(numpy_array, device):
        return torch.from_numpy(numpy_array.astype(np.float32)).to(device)
    
    @staticmethod
    def save_checkpoint(model, epoch, metrics, losses, optimizer):

        if not os.path.exists(PytorchUtils.WEIGHTS_FOLDER):
            os.makedirs(PytorchUtils.WEIGHTS_FOLDER)

        file_name = f'{model.__class__.__name__}_{epoch}_epochs.pt'

        current_time = datetime.now()

        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'losses' : losses,
                'date': current_time.strftime("%Y-%m-%d %H:%M:%S")
            }, os.path.join(PytorchUtils.WEIGHTS_FOLDER, file_name))
        
        print(f"model saved to : {os.path.join(PytorchUtils.WEIGHTS_FOLDER, file_name)}")

    @staticmethod
    def load_checkpoint(path):
        checkpoint = torch.load(path)
        
        return checkpoint

    @staticmethod
    def device_section():
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.empty_cache()
        else:
            device = torch.device("cpu")

        return device