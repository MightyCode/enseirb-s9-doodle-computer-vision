from .conv_autoencoder import ConvAutoencoder

import torch.nn as nn

class ConvAutoencoderEmbed(ConvAutoencoder):
    def __init__(self, layer_sizes, device, width, height, classes, dropout=0., batch_norm=True, class_number=8):
        super().__init__(layer_sizes, device, width, height, classes, dropout, batch_norm)

        self.encoded_width = int(self.width/(2*(len(layer_sizes)-2)))
        self.encoded_height = int(self.height/(2*(len(layer_sizes)-2)))
        self.encoded_num_channels = int(self.encoder[-1].out_channels)

        flattened_shape = int(self.encoded_height*self.encoded_width*self.encoded_num_channels)

        self.embedding = nn.Embedding(num_embeddings=class_number, 
                                      embedding_dim=flattened_shape)

    def forward_full(self, x, labels):
        encoded = self.encoder(x)

        embedding = self.embedding(labels)

        embedding = embedding.view(embedding.shape[0],
                                   self.encoded_num_channels,
                                   self.encoded_width,
                                   self.encoded_height)
        
        # encoded and embedding is tensor of same shape, add it
        encoded_class = encoded + embedding

        decoded = self.decoder(encoded_class)

        return encoded, encoded_class, decoded

    def forward(self, x, labels):
        x = x.view(-1, 1, self.width, self.height)

        _, encoded_class, decoded = self.forward_full(x, labels)

        return encoded_class.squeeze(), decoded.squeeze()
