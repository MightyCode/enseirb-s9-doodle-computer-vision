import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from .base_model import BaseModel

class BaseVariationalAutoencoder(BaseModel):
    def __init__(self, layer_sizes, device, width, height, classes, 
                 encode_class=False,
                 rl = 1.0,
                 kl = 0.0):
        super().__init__(layer_sizes, device, width, height, classes, 
                 encode_class)

        self.rl = rl
        self.kl = kl

    def get_decoded(self, input, labels):
        _, _, _, decoded = self.forward(input, labels)
        return decoded

    def vae_loss(self, mean, logvar, decoded, inputs):
        # Fonction de perte de reconstruction
        reproduction_loss = F.mse_loss(decoded, inputs, reduction='sum')
        
        # KL divergence entre la distribution latente et une distribution normale
        KLD = - 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        # Combinaison des deux termes de perte
        return self.rl * reproduction_loss + self.kl * KLD

    def train_autoencoder(self, train_loader: DataLoader, valid_loader: DataLoader, optimizer,  criterion=vae_loss, num_epochs=10):

        self.train_psnr_values = []
        self.train_ssim_values = []

        self.validation_psnr_values = []
        self.validation_ssim_values = []

        self.train_loss_values = []
        self.validation_loss_values = []

        for epoch in range(num_epochs):
            self.train()
            # Train by batch of images
            for data in train_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                mu, sigma, z, decoded = self.forward(inputs, labels=labels)
                loss = criterion(mu, sigma, decoded, inputs)

                # Backward pass
                loss.backward()
                optimizer.step()

            # Train loss
            self.train_loss_values.append(loss.detach().cpu().item())

            self.eval()

            with torch.no_grad():

                # loop on validation to compute validation loss
                for data in valid_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # Forward pass
                    mu, sigma, z, decoded = self.forward(inputs, labels=labels)
                    loss = criterion(mu, sigma, decoded, inputs)

                self.validation_loss_values.append(loss.detach().cpu().item())

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

                    _, _, _, decoded = self.forward(inputs, labels=labels)

                    for i in range(inputs.size(0)):
                        nb_train_images+=1
                        img_as_tensor = inputs[i]
                        decoded_as_tensor = decoded[i]

                        image_matrix = img_as_tensor.cpu().detach().numpy()
                        decoded_matrix = decoded_as_tensor.squeeze().cpu().detach().numpy()

                        train_psnr += psnr(image_matrix, decoded_matrix)
                        train_ssim += ssim(image_matrix, decoded_matrix, data_range=decoded_matrix.max() - decoded_matrix.min())

                for data in valid_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    _, _, _, decoded = self.forward(inputs, labels=labels)

                    for i in range(inputs.size(0)):
                        nb_valid_images+=1
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

            self.train_psnr_values.append(train_psnr)
            self.train_ssim_values.append(train_ssim)
            self.validation_psnr_values.append(validation_psnr)
            self.validation_ssim_values.append(validation_ssim)
            print(f'Epoch [{epoch+1}/{num_epochs}]\tLoss: {self.train_loss_values[-1]:.4f}\tTest Loss {self.validation_loss_values[-1]:.4f}\t', end = "")
            print(f'Train PSNR: {train_psnr:.4f}\tTrain SSIM: {train_ssim:.4f}\tValidation PSNR: {validation_psnr:.4f}\tValidation SSIM: {validation_ssim:.4f}')