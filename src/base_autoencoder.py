from torch.utils.data import DataLoader

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt

from .base_model import BaseModel

class BaseAutoencoder(BaseModel):
    def __init__(self, layer_sizes, device, width, height, classes, encode_class=False):
        super().__init__(layer_sizes, device, width, height, classes, encode_class)

    def get_decoded(self, input, labels):
        _, decoded = self.forward(input, labels)
        return decoded
    
    def train_autoencoder(self, train_loader: DataLoader, valid_loader: DataLoader, criterion, optimizer, num_epochs):

        self.losses = {
            'train': {
                "total_loss": []
            },
            'validation': {
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
                loss = criterion(input=pack["decoded"], target=inputs)

                # Backward pass
                loss.backward()
                optimizer.step()

            # Train loss
            self.losses['train']['total_loss'].append(loss.item())

            self.eval()
            # loop on validation to compute validation loss
            for data in valid_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                pack = self.forward(inputs, labels=labels)
                loss = criterion(input=pack["decoded"], target=inputs)

            self.losses['validation']['total_loss'].append(loss.item())

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

                for i in range(inputs.size(0)):
                    nb_train_images+=1
                    img_as_tensor = inputs[i]
                    decoded_as_tensor = pack["decoded"][i]

                    image_matrix = img_as_tensor.cpu().detach().numpy()
                    decoded_matrix = decoded_as_tensor.squeeze().cpu().detach().numpy()

                    train_psnr += psnr(image_matrix, decoded_matrix)
                    train_ssim += ssim(image_matrix, decoded_matrix, 
                                       data_range=decoded_matrix.max() - decoded_matrix.min())

            for data in valid_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                pack = self.forward(inputs, labels=labels)

                for i in range(inputs.size(0)):
                    nb_valid_images+=1
                    img_as_tensor = inputs[i]
                    decoded_as_tensor = pack["decoded"][i]

                    image_matrix = img_as_tensor.cpu().detach().numpy()
                    decoded_matrix = decoded_as_tensor.squeeze().cpu().detach().numpy()

                    validation_psnr += psnr(image_matrix, decoded_matrix)
                    validation_ssim += ssim(image_matrix, decoded_matrix, 
                                            data_range=decoded_matrix.max() - decoded_matrix.min())

            train_psnr /= nb_train_images
            train_ssim /= nb_train_images
            validation_psnr /= nb_valid_images
            validation_ssim /= nb_valid_images

            self.metrics['train']['psnr'].append(train_psnr)
            self.metrics['train']['ssim'].append(train_ssim)
            self.metrics['validation']['psnr'].append(validation_psnr)
            self.metrics['validation']['ssim'].append(validation_ssim)

            print(f'Epoch [{epoch+1}/{num_epochs}]', end=" ")
            print(f'T Loss: {self.losses["train"]["total_loss"][-1]:.4f}', end=" ")
            print(f'V Loss: {self.losses["validation"]["total_loss"][-1]:.4f}', end=" ")
            print(f'T PSNR: {self.metrics["train"]["psnr"][-1]:.4f}', end=" ")
            print(f'T SSIM: {self.metrics["train"]["ssim"][-1]:.4f}', end=" ")
            print(f'V PSNR: {self.metrics["validation"]["psnr"][-1]:.4f}', end=" ")
            print(f'V SSIM: {self.metrics["validation"]["ssim"][-1]:.4f}')