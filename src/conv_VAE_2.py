from .base_variational_autoencoder import BaseVariationalAutoencoder

import torch
import torch.nn as nn

class ConvVariationalAutoencoder2(BaseVariationalAutoencoder):
    def __init__(self, layer_sizes, device, width, height, classes, latent_dim=None, dropout=0., batch_norm=True, rl=1, kl=0, latent_channels=64):
        super().__init__(layer_sizes, device, width, height, classes, rl=rl, kl=kl)
        self.latent_type = "convolutional"

        self.layer_sizes = layer_sizes
        self.latent_channels = latent_channels
        
        kernel_size = 3

        # decoder layers
        for i in range(len(layer_sizes) - 1):
            self.encoder.add_module(f"encoder_{i}", nn.Conv2d(layer_sizes[i], layer_sizes[i+1], kernel_size=kernel_size, padding=1))
            if i < len(layer_sizes) - 2:
                self.encoder.add_module(f"encoder_relu_{i}", nn.ELU())
                self.encoder.add_module(f"encoder_max_pool_{i}", nn.MaxPool2d(2, 2))
                self.encoder.add_module(f"encoder_dropout_{i}", nn.Dropout(dropout))
                if batch_norm:
                    self.encoder.add_module(f"encoder_batchnorm_{i}", nn.BatchNorm2d(layer_sizes[i+1]))

        # latent space
        self.mu = nn.Conv2d(layer_sizes[-1], latent_channels, kernel_size=kernel_size, padding=1)
        self.logvar = nn.Conv2d(layer_sizes[-1], latent_channels, kernel_size=kernel_size, padding=1)

        self.latent_space_output = nn.Conv2d(latent_channels, layer_sizes[-1], kernel_size=kernel_size, padding=1)
        
        # decoder layers
        for i in range(len(layer_sizes)-1, 0, -1):
            self.decoder.add_module(f"decoder_{i}", nn.Conv2d(layer_sizes[i], layer_sizes[i-1], kernel_size=kernel_size, padding=1))
            if i > 1:
                self.decoder.add_module(f"decoder_relu_{i}", nn.ELU())
                self.decoder.add_module(f"encoder_upsample_{i}", nn.Upsample(scale_factor=2, mode='nearest'))
                self.decoder.add_module(f"encoder_dropout_{i}", nn.Dropout(dropout))
                if batch_norm:
                    self.decoder.add_module(f"encoder_batchnorm_{i}", nn.BatchNorm2d(layer_sizes[i-1]))

        self.decoder.add_module("decoder_sigmoid", nn.Sigmoid())

        self.N = torch.distributions.Normal(0, 1)
    
    def print_model(self):
        print('encoder :')
        print(self.encoder)
        print('latent space')
        print(self.mu)
        print(self.logvar)
        print(self.latent_space_output)
        print('decoder')
        print(self.decoder)
    
    
    def encode(self, x, labels=None):
        x = self.encoder(x)

        mu = self.mu(x)
        logvar = self.logvar(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(mu.shape).to(self.device)
        return mu + eps * std

    def decode(self, z):
        latent_space_output = self.latent_space_output(z)
        
        z = self.decoder(latent_space_output)
        return z
    
    def get_latent_dim(self):
        self.encoded_width = self.width
        self.encoded_height = self.height
        for _ in range(len(self.layer_sizes)-2):
            self.encoded_height//=2
            self.encoded_width//=2
        self.encoded_num_channels = self.latent_channels
        return (self.encoded_num_channels, self.encoded_width, self.encoded_height)

    def forward(self, x, labels=None):
        x = x.view(-1, 1, self.width, self.height)

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)

        return {
            'encoded': z.squeeze(), 
            'decoded': x_reconstructed.squeeze(), 
            'mu': mu, 
            'sigma': logvar
        }