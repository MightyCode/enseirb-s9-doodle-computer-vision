from .conv_variational_autoencoder import ConvVariationalAutoencoder

import torch.nn as nn

class ConvVariationalAutoencoderEmbed(ConvVariationalAutoencoder):
    def __init__(self, layer_sizes, device, width, height, classes, hyperparameters={}):
        super().__init__(layer_sizes, device, width, height, classes, hyperparameters)

        self.embedding = nn.Embedding(num_embeddings=len(classes), 
                                      embedding_dim=self.latent_vector_size)
        
    def get_embed(self, labels):
        embedding = self.embedding(labels)
        return embedding
    
    def add_class_to_encoded(self, encoded_before, embedding):
        return encoded_before + embedding
    
    def forward(self, x, labels=None):
        x = x.view(-1, 1, self.width, self.height)

        mu, logvar = self.encode(x)
        
        if self.sample_mode or self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        embedding = self.get_embed(labels)

        encoded_class = self.add_class_to_encoded(z, embedding)
        
        x_reconstructed = self.decode(encoded_class)

        return {
            'encoded_before': z.squeeze(), 
            'encoded': encoded_class.squeeze(),
            'decoded': x_reconstructed.squeeze(), 
            'mu': mu, 
            'sigma': logvar
        }