import torch
import torch.nn.functional as F

from .base_model import BaseModel

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

class BaseVariationalAutoencoder(BaseModel):
    def __init__(self, layer_sizes, device, width, height, classes, 
                 hyperparameters={}):
        super().__init__(layer_sizes, device, width, height, classes, 
                 hyperparameters)

        self.rl = hyperparameters["rl"]
        self.kl = hyperparameters["kl"]
        self.lpips = hyperparameters["lpips"]
        self.sample_mode = False

        self.lpips_evaluator = LPIPS(net_type='vgg').to(self.device)

    def set_sample_mode(self, mode: bool):
        self.sample_mode = mode
        
    def vae_loss(self, inputs, info):
        mean = info["mu"]
        logvar = info["sigma"]
        decoded = info["decoded"]

        # Fonction de perte de reconstruction
        reproduction_loss = F.mse_loss(decoded, inputs)
        
        # KL divergence entre la distribution latente et une distribution normale
        KLD = - 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / (self.width * self.height)

        # Combinaison des deux termes de perte
        return {
            "rl_loss" : self.rl * reproduction_loss,
            "kl_loss" : self.kl * KLD
        }
    
    def lpips_loss(self, inputs, info):
        loss = self.vae_loss(inputs, info)

        # LPIPS loss

        decoded = info["decoded"]

        decoded_rgb = decoded.unsqueeze(1).expand(-1, 3, -1, -1)
        inputs_rgb = inputs.unsqueeze(1).expand(-1, 3, -1, -1)
        
        loss["lpips_loss"] = self.lpips_evaluator(decoded_rgb, inputs_rgb) * self.lpips

        return loss