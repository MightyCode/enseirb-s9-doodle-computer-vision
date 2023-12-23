from .base_model import BaseModel
from torch.nn import functional as F

class BaseAutoencoder(BaseModel):
    def __init__(self, layer_sizes, device, width, height, classes, hyperparameters={}):
        super().__init__(layer_sizes, device, width, height, classes, hyperparameters)

    def get_decoded(self, input, labels):
        _, decoded = self.forward(input, labels)
        return decoded
    
    def mse_loss(self, inputs, info):
        decoded = info["decoded"]

        loss =  F.mse_loss(decoded, inputs)

        return {
            "total_loss": loss,
        }