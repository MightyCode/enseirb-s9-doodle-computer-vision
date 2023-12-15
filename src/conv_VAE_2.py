from .base_variational_autoencoder import BaseVariationalAutoencoder

import torch
import torch.nn as nn

class ConvVariationalAutoencoder2(BaseVariationalAutoencoder):
    def __init__(self, layer_sizes, device, width, height, classes, dropout=0., batch_norm=True, rl=1, kl=0, latent_channels=64, class_number=8):
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

        self.latent_dim = self.get_latent_dim()
        embedding_dim = int(self.latent_dim[0] * self.latent_dim[1] * self.latent_dim[2])
        self.embedding = nn.Embedding(num_embeddings=class_number, 
                                      embedding_dim=embedding_dim)
    
    def print_model(self):
        print('encoder :')
        print(self.encoder)
        print('latent space')
        print(f'mu : {self.mu}')
        print(f'logvar : {self.logvar}')
        print(f'latent_exit : {self.latent_space_output}')
        print('decoder :')
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
    
    def get_embed(self, labels):
        embedding = self.embedding(labels)
        embedding = embedding.reshape(
            embedding.shape[0],
            self.latent_dim[0],
            self.latent_dim[1],
            self.latent_dim[2]
        )
        return embedding
    
    def add_class_to_encoded(self, encoded_before, embedding):
        return encoded_before + embedding

    def forward(self, x, labels=None):
        x = x.view(-1, 1, self.width, self.height)

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        embedding = self.get_embed(labels)

        encoded_class = self.add_class_to_encoded(z, embedding)
        
        x_reconstructed = self.decode(encoded_class)

        return {
            'encoded_before': z.squeeze(), 
            'encoded': encoded_class,
            'decoded': x_reconstructed.squeeze(), 
            'mu': mu, 
            'sigma': logvar
        }