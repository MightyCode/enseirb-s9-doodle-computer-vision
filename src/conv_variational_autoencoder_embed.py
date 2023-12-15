from .conv_variational_autoencoder import ConvVariationalAutoencoder

import torch.nn as nn

class ConvVariationalAutoencoderEmbed(ConvVariationalAutoencoder):
    def __init__(self, layer_sizes, device, width, height, classes, latent_dim=None, dropout=0., batch_norm=True, rl=1, kl=0, class_number=8):
        super().__init__(layer_sizes, device, width, height, classes, rl=rl, kl=kl)

        self.embedding = nn.Embedding(num_embeddings=class_number, 
                                      embedding_dim=self.latent_dim)
        
    def get_embed(self, labels):
        embedding = self.embedding(labels)
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
            'encoded': encoded_class.squeeze(),
            'decoded': x_reconstructed.squeeze(), 
            'mu': mu, 
            'sigma': logvar
        }