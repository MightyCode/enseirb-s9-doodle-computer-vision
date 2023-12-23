import torch
import torch.nn as nn

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt
import random


from utils.PytorchUtils import PytorchUtils
import os
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class BaseModel(nn.Module):
    def __init__(self, layer_sizes, device, width, height, classes, hyperparameters={}):
        super(BaseModel, self).__init__()
        self.architecture = layer_sizes
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        self.device = device
        self.width = width
        self.height = height
        self.classes = classes
        self.layer_sizes = layer_sizes
        self.hyperparameters = hyperparameters
        self.encode_class = hyperparameters["encode_class"] if "encode_class" in hyperparameters else False

        self.latent_type = "vector"

    def get_embed(self, labels):
        raise("Not implemented")

    def add_class_to_encoded(self, encoded_before, embedding):
        raise("Not implemented")
    
    def get_latent_dim(self):
        raise("Not implemented")
    
    def decode(self, x):
        raise("Not implemented")

    def forward(self, x, labels=None):
        raise("Not implemented")

    def get_total_and_make_sum(self, epoch_losses, losses):
        total_loss = None

        for key in losses.keys():
            if key not in epoch_losses.keys():
                epoch_losses[key] = 0

            epoch_losses[key] += losses[key].item()

            if key == "total_loss":
                total_loss = losses[key]

        if total_loss == None:
            first_key  = list(losses.keys())[0]
            total_loss = losses[first_key]

            for key in losses.keys():
                if key != first_key:
                    total_loss += losses[key]

            if "total_loss" not in epoch_losses.keys():
                epoch_losses["total_loss"] = 0

            epoch_losses["total_loss"] += total_loss.item()

        return total_loss

    def train_autoencoder(self, train_loader: DataLoader, valid_loader: DataLoader, 
                          optimizer, criterion, num_epochs=10, path=None):
        self.losses = {
            'train': {
            },
            'validation': {
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

        epochs_to_perform = num_epochs
        start_epoch = 0
        
        print(f"Attempting to load weights from : {path}")
        if path and os.path.exists(path):
            print(f'loading weights from : {path}')

            checkpoint = PytorchUtils.load_checkpoint(path)

            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            epochs_to_perform = num_epochs - checkpoint['epoch']
            original_num_epochs = num_epochs
            start_epoch = checkpoint['epoch']
            num_epochs = start_epoch+epochs_to_perform

            self.losses = checkpoint['losses']
            self.metrics = checkpoint['metrics']

        if(epochs_to_perform > 0):
            for epoch in range(start_epoch, start_epoch+epochs_to_perform):
                self.train()
                # Train by batch of images
                epoch_losses = {}

                for data in train_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    pack = self.forward(inputs, labels=labels)
                    current_losses = criterion(inputs, pack)
                    total_loss = self.get_total_and_make_sum(epoch_losses, current_losses)
                    total_loss.backward()

                    # Backward pass
                    optimizer.step()

                # Register train losses
                for key in epoch_losses.keys():
                    if key not in self.losses['train']:
                        self.losses['train'][key] = []

                    self.losses['train'][key].append(epoch_losses[key] / len(train_loader))
                
                self.eval()
                eval_epoch_losses = {}
                with torch.no_grad():
                    # loop on validation to compute validation loss
                    for data in valid_loader:
                        inputs, labels = data
                        inputs, labels = inputs.to(self.device), labels.to(self.device)

                        # Forward pass
                        pack = self.forward(inputs, labels=labels)
                        current_losses = criterion(inputs, pack)

                        total_loss = self.get_total_and_make_sum(eval_epoch_losses, current_losses)

                    # Register validation losses
                    for key in eval_epoch_losses.keys():
                        if key not in self.losses['validation']:
                            self.losses['validation'][key] = []

                        self.losses['validation'][key].append(eval_epoch_losses[key] / len(valid_loader))

                    self.eval()
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
                            nb_valid_images += 1
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

                # if epoch 1 print the columns names with a column number
                if epoch == 0:
                    print("0: Epoch\t", end="")

                    number = 1
                    # print the train loss with a column number
                    for key in self.losses['train'].keys():
                        print(f"{number} : Train {key}\t", end="")
                        number += 1
                    
                    # print the validation loss with a column number
                    for key in self.losses['validation'].keys():
                        print(f"{number} : Validation {key}\t", end="")
                        number += 1
                    
                    # print the train psnr and ssim with a column number
                    for key in self.metrics['train'].keys():
                        print(f"{number} : Train {key}\t", end="")
                        number += 1
                    
                    # print the validation psnr and ssim with a column number
                    for key in self.metrics['validation'].keys():
                        print(f"{number} : Validation {key}\t", end="")
                        number += 1

                number = 0
                # print the epoch number with a column number
                print(f"\n{epoch+1}\t", end="")
                number += 1

                # print the train loss with a column number
                for key in self.losses['train'].keys():
                    print(f"{number} : {self.losses['train'][key][-1]:.4f}\t", end="")
                    number += 1
                
                # print the validation loss with a column number
                for key in self.losses['validation'].keys():
                    print(f"{number} : {self.losses['validation'][key][-1]:.4f}\t", end="")
                    number += 1
                
                # print the train psnr and ssim with a column number
                for key in self.metrics['train'].keys():
                    print(f"{number} : {self.metrics['train'][key][-1]:.4f}\t", end="")
                    number += 1
                
                # print the validation psnr and ssim with a column number
                for key in self.metrics['validation'].keys():
                    print(f"{number} : {self.metrics['validation'][key][-1]:.4f}\t", end="")
                    number += 1
        else:
            print(f'attempting to train {original_num_epochs} epochs but {start_epoch} epochs already done -> no training performed')
    
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

    def plot_training(self, path=None):
        # Plot total_loss on one subplot, and other loss in another subplot if they exist
        other_loss_exist = False
        for key in self.losses["train"].keys():
            if key != "total_loss":
                other_loss_exist = True
                break

        _, axes = plt.subplots(1, 4 if other_loss_exist else 3, figsize=(15 if other_loss_exist else 12, 6))

        number = 0
        axes[number].plot(self.losses["train"]["total_loss"], label='Train Loss')
        axes[number].plot(self.losses["validation"]["total_loss"], label='Validation Loss')
        axes[number].set_xlabel('Epoch')
        axes[number].set_ylabel('Loss')
        axes[number].legend()

        if other_loss_exist:
            number += 1
            for key in self.losses["train"].keys():
                if key != "total_loss":
                    axes[number].plot(self.losses["train"][key], label=f'Train {key}')
                    axes[number].plot(self.losses["validation"][key], label=f'Validation {key}')
                    
            axes[number].set_xlabel('Epoch')
            axes[number].set_ylabel('Loss')
            axes[number].legend()

        number += 1
        axes[number].plot(self.metrics["train"]["psnr"], label='Train PSNR')
        axes[number].plot(self.metrics["validation"]["psnr"], label='Validation PSNR')
        axes[number].set_xlabel('Epoch')
        axes[number].set_ylabel('PSNR')
        axes[number].legend()

        number += 1
        axes[number].plot(self.metrics["train"]["ssim"], label='Train SSIM')
        axes[number].plot(self.metrics["validation"]["ssim"], label='Validation SSIM')
        axes[number].set_xlabel('Epoch')
        axes[number].set_ylabel('SSIM')
        axes[number].legend()

        plt.tight_layout()

        if path:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

    def show_images(self, train_set, other_set, train_set_name="Train", other_set_name="Test", path=None):
        # show original and reconstructed images
        num_cols = 4
        num_rows = 2

        _, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))

        for i in range(num_cols // 2):
            # Train images
            index = random.randint(0, train_set.__len__() - 1)

            axes[0, i].imshow(train_set.get_image_2d(index), cmap='gray')
            axes[0, i].set_title(f"{train_set_name} ori. {self.classes[train_set[index][1]]}")
            axes[0, i].axis('off')

            # Reconstructed images
            with torch.no_grad():
                image, label = train_set[index]
                train_tensor = torch.from_numpy(image).to(self.device)
                label_tensor = torch.tensor(label).to(self.device)

                pack = self.forward(train_tensor.unsqueeze(0), labels=label_tensor.unsqueeze(0))
                
                decoded = pack["decoded"]
                decoded = decoded.view(-1, self.height, self.width)  # Reshape decoded images

                axes[1, i].imshow(decoded.cpu().detach().numpy()[0], cmap='gray')
                axes[1, i].set_title(f"{other_set_name} recon. {self.classes[train_set[index][1]]}")
                axes[1, i].axis('off')
        
        for i in range(num_cols // 2, num_cols):
            # Test images
            index = random.randint(0, other_set.__len__() - 1)

            axes[0, i].imshow(other_set.get_image_2d(index), cmap='gray')
            axes[0, i].set_title(f"Validation ori., {self.classes[other_set[index][1]]}")
            axes[0, i].axis('off')

            # Reconstructed images
            with torch.no_grad():
                image, label = other_set[index]
                validation_tensor = torch.from_numpy(image).to(self.device)
                label_tensor = torch.tensor(label).to(self.device)

                pack = self.forward(validation_tensor.unsqueeze(0), labels=label_tensor.unsqueeze(0))
                decoded = pack["decoded"]
                decoded = decoded.view(-1, self.height, self.width)

                axes[1, i].imshow(decoded.cpu().detach().numpy()[0], cmap='gray')
                axes[1, i].set_title(f"Validation recon. {self.classes[other_set[index][1]]}")
                axes[1, i].axis('off')

        if path:
            plt.savefig(path)
            plt.close()
        else:
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

            with torch.no_grad():
                pack = self.forward(test_images, labels=test_labels)
                decoded = pack["decoded"]

                decoded_matrices = decoded.cpu().detach().numpy()
                test_images_matrices = test_images.cpu().detach().numpy()

                for i in range(test_images.size(0)):
                    image_matrix = test_images_matrices[i]
                    decoded_matrix = decoded_matrices[i]

                    psnr_value = psnr(image_matrix, decoded_matrix)
                    ssim_value = ssim(image_matrix, decoded_matrix,
                                    data_range=decoded_matrix.max() - decoded_matrix.min())
                    
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

        with torch.no_grad():
            # PSNR image
            axes[0, 0].imshow(image_set.get_image_2d(lowest_psnr_index), cmap='gray')
            axes[0, 0].set_title("Original : " + psnr_image_label)
            axes[0, 0].axis('off')


            image, label = image_set[lowest_psnr_index]
            psnr_image_tensor = torch.from_numpy(image).to(self.device)
            label_tensor = torch.tensor(label).to(self.device)

            pack = self.forward(psnr_image_tensor.unsqueeze(0), labels=label_tensor.unsqueeze(0))
            decoded = pack["decoded"]
            decoded = decoded.view(-1, self.height, self.width)  # Reshape decoded images

            axes[0, 1].imshow(decoded.cpu().detach().numpy()[0], cmap='gray')
            axes[0, 1].set_title(f"Reconstructed : {psnr_image_label}, PSNR: {psnr:.4f}")
            axes[0, 1].axis('off')

            # SSIM image
            axes[1, 0].imshow(image_set.get_image_2d(lowest_ssim_index), cmap='gray')
            axes[1, 0].set_title("Original : " + ssim_image_label)
            axes[1, 0].axis('off')

            image, label = image_set[lowest_ssim_index]
            ssim_image_tensor = torch.from_numpy(image).to(self.device)
            label_tensor = torch.tensor(label).to(self.device)

            pack = self.forward(ssim_image_tensor.unsqueeze(0), labels=label_tensor.unsqueeze(0))
            decoded = pack["decoded"]
            decoded = decoded.view(-1, self.height, self.width)  # Reshape decoded images

            axes[1, 1].imshow(decoded.cpu().detach().numpy()[0], cmap='gray')
            axes[1, 1].set_title(f"Reconstructed : {ssim_image_label}, SSIM: {ssim:.4f}")
            axes[1, 1].axis('off')

            plt.show()