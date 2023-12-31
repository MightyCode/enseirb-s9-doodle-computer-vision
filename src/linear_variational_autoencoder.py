"""
Embed variational autoencoder in a linear layer.
"""

from .base_variational_autoencoder import BaseVariationalAutoencoder

import torch
import torch.nn as nn

class LinearVariationalAutoencoder(BaseVariationalAutoencoder):
    def __init__(self, layer_sizes, device, width, height, classes, hyperparameters={}):
        super().__init__(layer_sizes, device, width, height, classes, hyperparameters)

        dropout = hyperparameters["dropout"]
        batch_norm = hyperparameters["batch_norm"]

        for i in range(len(layer_sizes) - 2):
            self.encoder.add_module(f"encoder_{i}", nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.encoder.add_module(f"encoder_relu_{i}", nn.ReLU())
                self.encoder.add_module(f"encoder_dropout_{i}", nn.Dropout(dropout))
                if batch_norm:
                    self.encoder.add_module(f"encoder_batchnorm_{i}", nn.BatchNorm1d(layer_sizes[i+1]))

        # Add decoder layers
        for i in range(len(layer_sizes)-1, 0, -1):
            self.decoder.add_module(f"decoder_{i}", nn.Linear(layer_sizes[i], layer_sizes[i-1]))
            if i > 1:
                self.decoder.add_module(f"decoder_relu_{i}", nn.ReLU())
                self.decoder.add_module(f"encoder_dropout_{i}", nn.Dropout(dropout))
                if batch_norm:
                    self.decoder.add_module(f"encoder_batchnorm_{i}", nn.BatchNorm1d(layer_sizes[i-1]))

        self.embedding = nn.Embedding(len(self.classes), layer_sizes[-1])

        self.mean = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self.log_var = nn.Linear(layer_sizes[-2], layer_sizes[-1])

        self.decoder.add_module("decoder_sigmoid", nn.Sigmoid())

        self.N = torch.distributions.Normal(0, 1)
    
    def print_model(self):
        print(self.encoder)
        print(self.mean)
        print(self.log_var)
        print(self.decoder)
    
    def get_embed(self, labels):
        embedding = self.embedding(labels)

        return embedding

    def add_class_to_encoded(self, encoded_before, embedding):
        return encoded_before + embedding
    
    def get_latent_dim(self):
        return self.layer_sizes[-1]
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x, labels=None):
        encoded = self.encoder(x)

        #print("encoded", encoded.max(), encoded.min())
        mu =  self.mean(encoded)
        #print("mu", mu.max(), mu.min())
        sigma = self.log_var(encoded)

        #print
        z = mu + sigma * torch.randn_like(sigma, device=self.device)
        #print("z", z.max(), z.min())

        # encoded and embedding is tensor of same shape, add it
        embedding = self.get_embed(labels)
        z_class = self.add_class_to_encoded(z, embedding)

        decoded = self.decoder(z_class)
        #print("decoded", decoded.max(), decoded.min())
        #print("===================")

        return {
            'encoded_before': z, 
            'encoded': z_class, 
            'decoded': decoded, 
            'mu': mu, 
            'sigma': sigma
        }