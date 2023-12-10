from .base_autoencoder import BaseAutoencoder

import torch
import torch.nn as nn

class LinearAutoencoderEmbed(BaseAutoencoder):
    def __init__(self, layer_sizes, device, width, height, classes, dropout=0., batch_norm=True, class_number=8):
        super().__init__(layer_sizes, device, width, height, classes, dropout=0., batch_norm=True, encode_class=True)

        for i in range(len(layer_sizes) - 1):
            self.encoder.add_module(f"encoder_{i}", nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.encoder.add_module(f"encoder_relu_{i}", nn.ReLU())
                self.encoder.add_module(f"encoder_dropout_{i}", nn.Dropout(dropout))
                if batch_norm:
                    self.encoder.add_module(f"encoder_batchnorm_{i}", nn.BatchNorm1d(layer_sizes[i+1]))

        self.embedding = nn.Embedding(class_number, layer_sizes[-1])

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
    
    def forward_full(self, x, labels):
        encoded = self.encoder(x)

        embedding = self.embedding(labels)
        
        # encoded and embedding is tensor of same shape, add it
        encoded_class = encoded + embedding

        decoded = self.decoder(encoded_class)

        return encoded, encoded_class, decoded

    def forward(self, x, labels):
        encoded, encoded_class, decoded = self.forward_full(x, labels)

        return encoded_class, decoded