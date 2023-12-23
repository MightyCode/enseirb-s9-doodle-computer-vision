from src.conv_autoencoder_embed import ConvAutoencoderEmbed
from src.conv_autoencoder import ConvAutoencoder
from src.conv_variational_autoencoder import ConvVariationalAutoencoder
from src.conv_variational_autoencoder_embed import ConvVariationalAutoencoderEmbed
from src.conv_VAE_2 import ConvVariationalAutoencoder2
from src.conv_VAE_embed_residual import ConvVariationalAutoencoderEmbedResidual

from src.linear_autoencoder_embed import LinearAutoencoderEmbed
from src.linear_variational_autoencoder import LinearVariationalAutoencoder
from src.linear_autoencoder import LinearAutoencoder

import torch

class InitModel:
    def init_model(
            MODEL : str, 
            device, 
            WIDTH, HEIGHT, classes,  
            CONV_ARCHITECTURE : list, LINEAR_ARCHITECTURE : list, hyperparameters,
            verbose=True):
        MODEL_INIT = None
        architecture = None
        is_embed_model = False

        if "conv" in MODEL or "convolutionnal" in MODEL:
            architecture = CONV_ARCHITECTURE
            if "variational" in MODEL or "var" in MODEL:
                if "lpips" in MODEL or "residual" in MODEL or "resid" in MODEL:
                    MODEL_INIT = ConvVariationalAutoencoderEmbedResidual
                    is_embed_model = True
                elif "2" in MODEL:
                    MODEL_INIT = ConvVariationalAutoencoder2
                    is_embed_model = True
                elif "embed" in MODEL or "embedded" in MODEL or "embedding" in MODEL:
                        MODEL_INIT = ConvVariationalAutoencoderEmbed
                        is_embed_model = True
                else:
                    MODEL_INIT = ConvVariationalAutoencoder
                    is_embed_model = False

            elif "embed" in MODEL or "embedded" in MODEL or "embedding" in MODEL:
                MODEL_INIT = ConvAutoencoderEmbed
                is_embed_model = True
            else:
                MODEL_INIT = ConvAutoencoder
        elif "linear" in MODEL:
            architecture = LINEAR_ARCHITECTURE
            if "variational" in MODEL or "var" in MODEL:
                MODEL_INIT = LinearVariationalAutoencoder
                is_embed_model = True
            elif "embed" in MODEL or "embedded" in MODEL or "embedding" in MODEL:
                MODEL_INIT = LinearAutoencoderEmbed
                is_embed_model = True
            else:
                MODEL_INIT = LinearAutoencoder

        if verbose:
            print("Chosen model : ", MODEL_INIT)

        if "variational" in MODEL or "var" in MODEL:
            autoencoder_model = MODEL_INIT(architecture, device, WIDTH, HEIGHT, classes, hyperparameters)
        else:
            autoencoder_model = MODEL_INIT(architecture, device, WIDTH, HEIGHT, classes, hyperparameters)
            
        autoencoder_model.to(device)

        return autoencoder_model, is_embed_model\
    
    def create_criterion_optimizer(MODEL, autoencoder_model, LR):
        # Define loss function and optimizer
        criterion = autoencoder_model.lpips_loss if "lpips" in MODEL else \
            autoencoder_model.vae_loss if "variational" in MODEL \
            else autoencoder_model.mse_loss
        
        print ("Criterion : ", criterion, )
        optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=LR, betas=(0.5, 0.999))
        return criterion, optimizer

    def print_model_characteristics(autoencoder_model):
        print("Class : ", autoencoder_model.__class__.__name__)
        # Print architecture 
        autoencoder_model.print_model()

        nb_params = sum(p.numel() for p in autoencoder_model.parameters() if p.requires_grad)
        print("Nb params", nb_params)

        # Compression factor 
        print(f'Compression factor: {(len(autoencoder_model.architecture)-2)*2}')