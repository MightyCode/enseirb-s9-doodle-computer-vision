import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from .base_model import BaseModel

from .model_saver import save_checkpoint, load_checkpoint
import os

class BaseVariationalAutoencoder(BaseModel):
    def __init__(self, layer_sizes, device, width, height, classes, 
                 encode_class=False,
                 rl = 1.0,
                 kl = 0.0):
        super().__init__(layer_sizes, device, width, height, classes, 
                 encode_class)

        self.rl = rl
        self.kl = kl

    def vae_loss(self, mean, logvar, decoded, inputs):
        # Fonction de perte de reconstruction
        reproduction_loss = F.mse_loss(decoded, inputs, reduction='sum')
        
        # KL divergence entre la distribution latente et une distribution normale
        KLD = - 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        # Combinaison des deux termes de perte
        return self.rl * reproduction_loss, self.kl * KLD

    def train_autoencoder(self, train_loader: DataLoader, valid_loader: DataLoader, 
                          optimizer, criterion=vae_loss, num_epochs=10, path=None):

        self.losses = {
            'train': {
                "reconstruction_loss": [],
                "kl_loss": [],
                "total_loss": []
            },
            'validation': {
                "reconstruction_loss": [],
                "kl_loss": [],
                "total_loss": []
            }
        }

        self.metrics = {
            'train': {
                'psnr': [],
                'ssim': []
            },
            'validation': {
                'psnr': [],
                'ssim': []
            }
        }

        metrics_to_save = {}
        epochs_to_perform = num_epochs
        
        if path and os.path.exists(path):
            checkpoint = load_checkpoint(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epochs_to_perform = num_epochs - checkpoint['epoch']
            num_epochs = checkpoint['epoch']
            metrics_to_save = checkpoint['metrics']

        if(epochs_to_perform > 0):
            for epoch in range(num_epochs):
                self.train()
                # Train by batch of images
                for data in train_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    pack = self.forward(inputs, labels=labels)

                    mu = pack["mu"]
                    sigma = pack["sigma"]

                    decoded = pack["decoded"]

                    rl_loss, kl_loss = criterion(mu, sigma, decoded, inputs)
                    loss = rl_loss + kl_loss

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                # Train loss
                self.losses['train']['reconstruction_loss'].append(rl_loss.item())
                self.losses['train']['kl_loss'].append(kl_loss.item())
                self.losses['train']['total_loss'].append(loss.item())

                metrics_to_save['train_RL'] = rl_loss.item()
                metrics_to_save['train_KL'] = kl_loss.item()
                metrics_to_save['train_loss'] = loss.item()

                self.eval()

                with torch.no_grad():

                    # loop on validation to compute validation loss
                    for data in valid_loader:
                        inputs, labels = data
                        inputs, labels = inputs.to(self.device), labels.to(self.device)

                        # Forward pass
                        pack = self.forward(inputs, labels=labels)

                        mu = pack["mu"]
                        sigma = pack["sigma"]

                        decoded = pack["decoded"]

                        rl_loss, kl_loss = criterion(mu, sigma, decoded, inputs)
                        loss = rl_loss + kl_loss

                    self.losses['validation']['reconstruction_loss'].append(round(rl_loss.item(), 2))
                    self.losses['validation']['kl_loss'].append(round(kl_loss.item(), 2))
                    self.losses['validation']['total_loss'].append(round(loss.item(), 2))

                    metrics_to_save['train_RL'] = rl_loss.item()
                    metrics_to_save['train_KL'] = kl_loss.item()
                    metrics_to_save['train_loss'] = loss.item()

                    # Calculate PSNR and SSIM for train and test sets
                    train_psnr = 0
                    train_ssim = 0
                    validation_psnr = 0
                    validation_ssim = 0

                    nb_train_images = 0
                    nb_valid_images = 0

                    for data in train_loader:
                        inputs, labels = data
                        inputs, labels = inputs.to(self.device), labels.to(self.device)

                        pack = self.forward(inputs, labels=labels)

                        decoded = pack["decoded"]

                        for i in range(inputs.size(0)):
                            nb_train_images += 1
                            img_as_tensor = inputs[i]
                            decoded_as_tensor = decoded[i]

                            image_matrix = img_as_tensor.cpu().detach().numpy()
                            decoded_matrix = decoded_as_tensor.squeeze().cpu().detach().numpy()

                            train_psnr += psnr(image_matrix, decoded_matrix)
                            train_ssim += ssim(image_matrix, decoded_matrix, data_range=decoded_matrix.max() - decoded_matrix.min())

                    for data in valid_loader:
                        inputs, labels = data
                        inputs, labels = inputs.to(self.device), labels.to(self.device)

                        pack = self.forward(inputs, labels=labels)

                        decoded = pack["decoded"]

                        for i in range(inputs.size(0)):
                            nb_valid_images += 1
                            img_as_tensor = inputs[i]
                            decoded_as_tensor = decoded[i]

                            image_matrix = img_as_tensor.cpu().detach().numpy()
                            decoded_matrix = decoded_as_tensor.squeeze().cpu().detach().numpy()

                            validation_psnr += psnr(image_matrix, decoded_matrix)
                            validation_ssim += ssim(image_matrix, decoded_matrix, data_range=decoded_matrix.max() - decoded_matrix.min())

                train_psnr /= nb_train_images
                train_ssim /= nb_train_images
                validation_psnr /= nb_valid_images
                validation_ssim /= nb_valid_images

                self.metrics['train']['psnr'].append(round(train_psnr, 2))
                self.metrics['train']['ssim'].append(round(train_ssim, 2))
                self.metrics['validation']['psnr'].append(round(validation_psnr, 2))
                self.metrics['validation']['ssim'].append(round(validation_ssim, 2))

                metrics_to_save['train_psnr'] = train_psnr
                metrics_to_save['train_ssim'] = train_ssim
                metrics_to_save['validation_psnr'] = validation_psnr
                metrics_to_save['validation_ssim'] = validation_ssim
                
                print(f'Ep [{epoch+1}/{num_epochs}]', end=" ")
                print(f'T L: {self.losses["train"]["total_loss"][-1]:.4f}', end=" ")
                print(f'T RL: {self.losses["train"]["reconstruction_loss"][-1]:.4f}', end=" ")
                print(f'T KL: {self.losses["train"]["kl_loss"][-1]:.4f}', end=" ")
                print(f'V L: {self.losses["validation"]["total_loss"][-1]:.4f}', end=" ")
                print(f'V RL: {self.losses["validation"]["reconstruction_loss"][-1]:.4f}', end=" ")
                print(f'V KL: {self.losses["validation"]["kl_loss"][-1]:.4f}', end=" ")
                print(f'T PSNR: {self.metrics["train"]["psnr"][-1]:.4f}', end=" ")
                print(f'T SSIM: {self.metrics["train"]["ssim"][-1]:.4f}', end=" ")
                print(f'V PSNR: {self.metrics["validation"]["psnr"][-1]:.4f}', end=" ")
                print(f'V SSIM: {self.metrics["validation"]["ssim"][-1]:.4f}')

        save_checkpoint(self, num_epochs, metrics_to_save, optimizer)