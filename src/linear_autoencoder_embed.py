from .base_autoencoder import BaseAutoencoder

import torch
import torch.nn as nn

class LinearAutoencoderEmbed(BaseAutoencoder):
    def __init__(self, layer_sizes, device, width, height, classes, dropout=0., batch_norm=True, class_number=8):
        super().__init__(layer_sizes, device, width, height, classes, encode_class=True)

        self.layer_sizes = layer_sizes

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

    def get_embed(self, labels):
        return self.embedding(labels)
    
    def add_class_to_encoded(self, encoded_before, embedding):
        return encoded_before + embedding
    
    def get_latent_dim(self):
        return self.layer_sizes[-1]
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x, labels):
        encoded = self.encoder(x)

        embedding = self.embedding(labels)
        
        # encoded and embedding is tensor of same shape, add it
        encoded_class = self.add_class_to_encoded(encoded, embedding)

        decoded = self.decoder(encoded_class)

        return {
            'encoded_before': encoded, 
            'encoded': encoded_class, 
            'decoded': decoded
        }