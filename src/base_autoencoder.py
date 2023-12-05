import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt
import random

class BaseAutoencoder(nn.Module):
    def __init__(self, layer_sizes, device, width, height, classes, dropout=0., batch_norm=True):
        super(BaseAutoencoder, self).__init__()
        self.architecture = layer_sizes
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        self.device = device
        self.width = width
        self.height = height
        self.classes = classes

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return encoded, decoded
    
    def _filter_batchnorm_layers(self, layers):
        filtered_layers = []
        for layer in layers.children():
            if isinstance(layer, nn.BatchNorm1d):
                continue
            elif isinstance(layer, nn.Sequential):
                filtered_layers.extend(self._filter_batchnorm_layers(layer))
            else:
                filtered_layers.append(layer)
        return filtered_layers

    def print_model(self):
        print(self.encoder)
        print(self.decoder)

    def train_autoencoder(self, train_loader: DataLoader, valid_loader: DataLoader, criterion, optimizer, num_epochs):
        self.train_psnr_values = []
        self.train_ssim_values = []

        self.validation_psnr_values = []
        self.validation_ssim_values = []

        self.train_loss_values = []
        self.validation_loss_values = []

        self.train()

        for epoch in range(num_epochs):
            # Train by batch of images
            for data in train_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                _, decoded = self.forward(inputs)
                loss = criterion(input=decoded, target=inputs)

                # Backward pass
                loss.backward()
                optimizer.step()

            # Train loss
            self.train_loss_values.append(loss.item())

            # loop on validation to compute validation loss
            for data in valid_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                
                # Forward pass
                _, decoded = self.forward(inputs)
                loss = criterion(input=decoded, target=inputs)

            self.validation_loss_values.append(loss.item())

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

                _, decoded = self.forward(inputs)

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

                _, decoded = self.forward(inputs)

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

        self.eval()
    
    def plot_psnr_ssim(self):
        _, axes = plt.subplots(1, 3, figsize=(12, 6))

        axes[0].plot(self.train_loss_values, label='Train Loss')
        axes[0].plot(self.validation_loss_values, label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        axes[1].plot(self.train_psnr_values, label='Train PSNR')
        axes[1].plot(self.validation_psnr_values, label='Validation PSNR')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('PSNR')
        axes[1].legend()

        axes[2].plot(self.train_ssim_values, label='Train SSIM')
        axes[2].plot(self.validation_ssim_values, label='Validation SSIM')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('SSIM')
        axes[2].legend()

        plt.tight_layout()
        plt.show()

    def show_images(self, train_set, validation_set):
        # show original and reconstructed images
        num_cols = 4
        num_rows = 2

        _, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))

        for i in range(num_cols // 2):
            # Train images
            index = random.randint(0, train_set.__len__())

            axes[0, i].imshow(train_set.get_image_2d(index), cmap='gray')
            axes[0, i].set_title(f"Train original {self.classes[train_set[index][1]]}")
            axes[0, i].axis('off')

            # Reconstructed images
            train_tensor = torch.from_numpy(train_set[index][0]).to(self.device)
            _, decoded = self.forward(train_tensor.unsqueeze(0))
            decoded = decoded.view(-1, self.height, self.width)  # Reshape decoded images

            axes[1, i].imshow(decoded.cpu().detach().numpy()[0], cmap='gray')
            axes[1, i].set_title(f"Train reconstructed {self.classes[train_set[index][1]]}")
            axes[1, i].axis('off')
        
        for i in range(num_cols // 2, num_cols):
            # Test images
            index = random.randint(0, validation_set.__len__())

            axes[0, i].imshow(validation_set.get_image_2d(index), cmap='gray')
            axes[0, i].set_title(f"Validation original, {self.classes[validation_set[index][1]]}")
            axes[0, i].axis('off')

            # Reconstructed images
            validation_tensor = torch.from_numpy(validation_set[index][0]).to(self.device)
            _, decoded = self.forward(validation_tensor.unsqueeze(0))
            decoded = decoded.view(-1, self.height, self.width)

            axes[1, i].imshow(decoded.cpu().detach().numpy()[0], cmap='gray')
            axes[1, i].set_title(f"Validation reconstructed {self.classes[validation_set[index][1]]}")
            axes[1, i].axis('off')

        plt.show()

    def return_lowest_image_index_psnr_ssim(self, dataset):
        # Show the lowest psnr then ssim in the test set
        lowest_psnr = 100
        lowest_ssim = 100
        lowest_psnr_index = 0
        lowest_ssim_index = 0

        for batch in dataset:
            test_images, test_labels = batch
            test_images, test_labels = test_images.to(self.device), test_labels.to(self.device)

            _, decoded = self(test_images)

            decoded_matrices = decoded.cpu().detach().numpy()
            test_images_matrices = test_images.cpu().detach().numpy()

            for i in range(test_images.size(0)):
                image_matrix = test_images_matrices[i]
                decoded_matrix = decoded_matrices[i]

                psnr_value = psnr(image_matrix, decoded_matrix)
                ssim_value = ssim(image_matrix, decoded_matrix, data_range=decoded_matrix.max() - decoded_matrix.min())
                
                if psnr_value < lowest_psnr:
                    lowest_psnr = psnr_value
                    lowest_psnr_index = i

                if ssim_value < lowest_ssim:
                    lowest_ssim = ssim_value
                    lowest_ssim_index = i

        return [lowest_psnr_index, lowest_psnr], [lowest_ssim_index, lowest_ssim]
    
    def show_lowest_psnr_ssim_image(self, image_set, lowest_psnr, lowest_ssim):
        # Show image with the lowest psnr and ssim compared to their original the test set on same plot

        lowest_psnr_index, psnr  = lowest_psnr[0], lowest_psnr[1]
        psnr_image_label = self.classes[image_set[lowest_psnr_index][1]]

        lowest_ssim_index, ssim  = lowest_ssim[0], lowest_ssim[1]
        ssim_image_label = self.classes[image_set[lowest_ssim_index][1]]

        _, axes = plt.subplots(2, 2, figsize=(7, 6))

        # PSNR image
        axes[0, 0].imshow(image_set.get_image_2d(lowest_psnr_index), cmap='gray')
        axes[0, 0].set_title("Original : " + psnr_image_label)
        axes[0, 0].axis('off')

        psnr_image_tensor = torch.from_numpy(image_set[lowest_psnr_index][0]).to(self.device)
        _, decoded = self(psnr_image_tensor.unsqueeze(0))
        decoded = decoded.view(-1, self.height, self.width)  # Reshape decoded images

        axes[0, 1].imshow(decoded.cpu().detach().numpy()[0], cmap='gray')
        axes[0, 1].set_title(f"Reconstructed : {psnr_image_label}, PSNR: {psnr:.4f}")
        axes[0, 1].axis('off')

        # SSIM image
        axes[1, 0].imshow(image_set.get_image_2d(lowest_ssim_index), cmap='gray')
        axes[1, 0].set_title("Original : " + ssim_image_label)
        axes[1, 0].axis('off')

        ssim_image_tensor = torch.from_numpy(image_set[lowest_ssim_index][0]).to(self.device)
        _, decoded = self(ssim_image_tensor.unsqueeze(0))
        decoded = decoded.view(-1, self.height, self.width)  # Reshape decoded images

        axes[1, 1].imshow(decoded.cpu().detach().numpy()[0], cmap='gray')
        axes[1, 1].set_title(f"Reconstructed : {ssim_image_label}, SSIM: {ssim:.4f}")
        axes[1, 1].axis('off')

        plt.show()