from src.conv_autoencoder_embed import ConvAutoencoderEmbed
from src.conv_autoencoder import ConvAutoencoder
from src.conv_variational_autoencoder import ConvVariationalAutoencoder
from src.conv_variational_autoencoder_embed import ConvVariationalAutoencoderEmbed
from src.conv_VAE_2 import ConvVariationalAutoencoder2

from src.linear_autoencoder_embed import LinearAutoencoderEmbed
from src.linear_variational_autoencoder import LinearVariationalAutoencoder
from src.linear_autoencoder import LinearAutoencoder

import torch
import torch.nn as nn

class InitModel:
    def init_model(
            MODEL : str, 
            device, 
            WIDTH, HEIGHT, classes,  
            CONV_ARCHITECTURE : list, LINEAR_ARCHITECTURE : list,  DROPOUT, BATCH_NORM, RL, KL,
            verbose=True):
        MODEL_INIT = None
        architecture = None
        is_embed_model = False

        if "conv" in MODEL or "convolutionnal" in MODEL:
            architecture = CONV_ARCHITECTURE
            if "variational" in MODEL or "var" in MODEL:
                if "2" in MODEL:
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
            autoencoder_model = MODEL_INIT(architecture, device, WIDTH, HEIGHT, classes, dropout=DROPOUT, 
                                           batch_norm=BATCH_NORM, rl=RL, kl=KL)
        else:
            autoencoder_model = MODEL_INIT(architecture, device, WIDTH, HEIGHT, classes, dropout=DROPOUT, 
                                           batch_norm=BATCH_NORM)
        autoencoder_model.to(device)

        return autoencoder_model, is_embed_model
    
    def create_criterion_optimizer(MODEL, autoencoder_model, LR):
        # Define loss function and optimizer
        criterion = autoencoder_model.vae_loss if "variational" in MODEL else nn.MSELoss()
        optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=LR)
        return criterion, optimizer

    def print_model_characteristics(autoencoder_model):
        # Print architecture 
        autoencoder_model.print_model()

        nb_params = sum(p.numel() for p in autoencoder_model.parameters() if p.requires_grad)
        print("Nb params", nb_params)

        # Compression factor 
        print(f'Compression factor: {(len(autoencoder_model.architecture)-2)*2}')