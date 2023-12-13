from .base_autoencoder import BaseAutoencoder

import torch.nn as nn

class ConvAutoencoder(BaseAutoencoder):
    def __init__(self, layer_sizes, device, width, height, classes, dropout=0., batch_norm=True):
        super().__init__(layer_sizes, device, width, height, classes)
        self.conv_model = True
        kernel_size = 3

        for i in range(len(layer_sizes) - 1):
            self.encoder.add_module(f"encoder_{i}", nn.Conv2d(layer_sizes[i], layer_sizes[i+1], kernel_size=kernel_size, padding=1))
            if i < len(layer_sizes) - 2:
                self.encoder.add_module(f"encoder_relu_{i}", nn.ReLU())
                self.encoder.add_module(f"encoder_max_pool_{i}", nn.MaxPool2d(2, 2))
                self.encoder.add_module(f"encoder_dropout_{i}", nn.Dropout(dropout))
                if batch_norm:
                    self.encoder.add_module(f"encoder_batchnorm_{i}", nn.BatchNorm2d(layer_sizes[i+1]))

        # Add decoder layers
        for i in range(len(layer_sizes)-1, 0, -1):
            self.decoder.add_module(f"decoder_{i}", nn.Conv2d(layer_sizes[i], layer_sizes[i-1], kernel_size=kernel_size, padding=1))
            if i > 1:
                self.decoder.add_module(f"decoder_relu_{i}", nn.ReLU())
                self.decoder.add_module(f"encoder_upsample_{i}", nn.Upsample(scale_factor=2, mode='nearest'))
                self.decoder.add_module(f"encoder_dropout_{i}", nn.Dropout(dropout))
                if batch_norm:
                    self.decoder.add_module(f"encoder_batchnorm_{i}", nn.BatchNorm2d(layer_sizes[i-1]))


        self.decoder.add_module("decoder_sigmoid", nn.Sigmoid())
    
    def forward(self, x, labels=None):
        x = x.view(-1, 1, self.width, self.height)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return {
            "encoded": encoded.squeeze(),
            "decoded": decoded.squeeze()
        }
