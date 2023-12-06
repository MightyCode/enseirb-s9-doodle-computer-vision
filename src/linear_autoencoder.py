from .base_autoencoder import BaseAutoencoder

import torch
import torch.nn as nn

class LinearAutoencoder(BaseAutoencoder):
    def __init__(self, layer_sizes, device, width, height, classes, dropout=0., batch_norm=True, encode_class=False):
        super().__init__(layer_sizes, device, width, height, classes, dropout=0., batch_norm=True, encode_class=encode_class)

        for i in range(len(layer_sizes) - 1):
            self.encoder.add_module(f"encoder_{i}", nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.encoder.add_module(f"encoder_relu_{i}", nn.ReLU())
                self.encoder.add_module(f"encoder_dropout_{i}", nn.Dropout(dropout))
                if batch_norm:
                    self.encoder.add_module(f"encoder_batchnorm_{i}", nn.BatchNorm1d(layer_sizes[i+1]))

        if self.encode_class:
            layer_sizes[-1] += 1

        # Add decoder layers
        for i in range(len(layer_sizes)-1, 0, -1):
            self.decoder.add_module(f"decoder_{i}", nn.Linear(layer_sizes[i], layer_sizes[i-1]))
            if i > 1:
                self.decoder.add_module(f"decoder_relu_{i}", nn.ReLU())
                self.decoder.add_module(f"encoder_dropout_{i}", nn.Dropout(dropout))
                if batch_norm:
                    self.decoder.add_module(f"encoder_batchnorm_{i}", nn.BatchNorm1d(layer_sizes[i-1]))


        self.decoder.add_module("decoder_sigmoid", nn.Sigmoid())
        
    def add_class_to_encoded(self, encoded_features, labels):
        return torch.cat((encoded_features, labels.unsqueeze(1)), dim=1)
    
    def forward(self, x, labels=None):
        encoded = self.encoder(x)

        if self.encode_class:
            if labels is None:
                raise Exception("Labels must be provided if encode_class is True")
            encoded = self.add_class_to_encoded(encoded, labels)

        decoded = self.decoder(encoded)
        
        return encoded, decoded