"""
Embed variational autoencoder in a linear layer.
"""

from .base_variational_autoencoder import BaseVariationalAutoencoder

import torch
import torch.nn as nn

class LinearVariationalAutoencoder(BaseVariationalAutoencoder):
    def __init__(self, layer_sizes, device, width, height, classes, dropout=0., batch_norm=True):
        super().__init__(layer_sizes, device, width, height, classes, 
                dropout=0., batch_norm=True)

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

        self.mean = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self.log_var = nn.Linear(layer_sizes[-2], layer_sizes[-1])

        self.decoder.add_module("decoder_sigmoid", nn.Sigmoid())

        self.print_model()

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0
    
    def print_model(self):
        print(self.encoder)
        print(self.mean)
        print(self.log_var)
        print(self.decoder)
    
    def forward(self, x, labels=None):
        encoded = self.encoder(x)

        mu =  self.mean(encoded)
        print("mu", mu.max(), mu.min())
        sigma = torch.exp(self.log_var(encoded))    

        print
        z = mu + sigma * self.N.sample(mu.shape)
        print("z", z.max(), z.min())

        # KL divergence

        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        print("kl", self.kl)

        decoded = self.decoder(z)
        print("decoded", decoded.max(), decoded.min())

        return z, decoded