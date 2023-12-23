from .base_variational_autoencoder import BaseVariationalAutoencoder

import torch
import torch.nn as nn

class ConvVariationalAutoencoder(BaseVariationalAutoencoder):
    def __init__(self, layer_sizes, device, width, height, classes, hyperparameters={}):
        super().__init__(layer_sizes, device, width, height, classes, hyperparameters)
        self.latent_type = "convolutional-variational"
        
        dropout = hyperparameters["dropout"]
        batch_norm = hyperparameters["batch_norm"]
        kernel_size = hyperparameters["kernel_size"] if "kernel_size" in hyperparameters else 3
        self.latent_vector_size = hyperparameters["latent_channels"] if "latent_channels" in hyperparameters else None

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

        self.encoded_width = self.width
        self.encoded_height = self.height
        for _ in range(len(layer_sizes)-2):
            self.encoded_height//=2
            self.encoded_width//=2
        self.encoded_num_channels = int(self.encoder[-1].out_channels)

        flattened_shape = int(self.encoded_height*self.encoded_width*self.encoded_num_channels)

        if self.latent_vector_size == None:
            self.latent_vector_size = flattened_shape

        self.flatten = nn.Flatten()
        latent_input_dim = self.latent_vector_size*2
        self.latent_space_input = nn.Linear(flattened_shape, latent_input_dim)

        latent_output_dim = self.latent_vector_size
        self.latent_space_output = nn.Linear(latent_output_dim, flattened_shape)
        
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
        print(self.flatten)
        print(self.latent_space_input)
        print(self.latent_space_output)
        print('decoder')
        print(self.decoder)
    
    
    def encode(self, x, labels=None):
        x = self.encoder(x)
        x = self.flatten(x)

        latent_space_input = self.latent_space_input(x)
        split = latent_space_input.split(self.latent_vector_size, dim=1)
        mu, logvar = split[0], split[1]

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(mu.shape).to(self.device)
        return mu + eps * std

    def decode(self, z):
        latent_space_output = self.latent_space_output(z)

        latent_space_output_reshaped = latent_space_output.view(latent_space_output.shape[0],
                                   self.encoded_num_channels,
                                   self.encoded_width,
                                   self.encoded_height)
        
        z = self.decoder(latent_space_output_reshaped)
        return z

    def get_latent_dim(self):
        return self.latent_vector_size
    
    def forward(self, x, labels=None):
        x = x.view(-1, 1, self.width, self.height)

        mu, logvar = self.encode(x)

        if self.sample_mode or self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        x_reconstructed = self.decode(z)

        return {
            'encoded': z.squeeze(), 
            'decoded': x_reconstructed.squeeze(), 
            'mu': mu, 
            'sigma': logvar
        }