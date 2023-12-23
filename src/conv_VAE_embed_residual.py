import torch
import torch.nn as nn
import torch.utils.data

from .base_variational_autoencoder import BaseVariationalAutoencoder

# Inspired by https://github.com/LukeDitria/CNN-VAE

class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3):
        super(ResDown, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out // 2, kernel_size, stride=2, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channel_out // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, stride=2, padding=kernel_size // 2)

        self.act_fnc = nn.ELU()

    def forward(self, x):
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        
        return self.act_fnc(self.bn2(x + skip))


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2):
        super(ResUp, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_in // 2, kernel_size, stride=1, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channel_in // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, stride=1, padding=kernel_size // 2)

        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")

        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.up_nn(x)
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return self.act_fnc(self.bn2(x + skip))


class ResBlock(nn.Module):
    """
    Residual block
    """

    def __init__(self, channel_in, channel_out, kernel_size=3):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_in // 2, kernel_size, stride=1, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channel_in // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

        if not channel_in == channel_out:
            self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, stride=1, padding=kernel_size // 2)
        else:
            self.conv3 = nn.Identity()

        self.act_fnc = nn.ELU()

    def forward(self, x):
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return self.act_fnc(self.bn2(x + skip))


class Encoder(nn.Module):
    """
    Encoder block
    """

    def __init__(self, channels, blocks=(1, 2, 4, 8), latent_channels=512):
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv2d(channels, blocks[0], 3, stride=1, padding=1)

        widths_in = list(blocks)
        widths_out = list(blocks[1:]) + [blocks[-1]]

        layer_blocks = []

        for w_in, w_out in zip(widths_in, widths_out):
            layer_blocks.append(ResDown(w_in, w_out))

        layer_blocks.append(ResBlock(blocks[-1], blocks[-1]))
        layer_blocks.append(ResBlock(blocks[-1], blocks[-1]))

        self.res_blocks = nn.Sequential(*layer_blocks)

        self.conv_mu = nn.Conv2d(blocks[-1], latent_channels, 1, stride=1)
        self.conv_log_var = nn.Conv2d(blocks[-1], latent_channels, 1, stride=1)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.act_fnc(self.conv_in(x))
        x = self.res_blocks(x)

        mu = self.conv_mu(x)
        log_var = self.conv_log_var(x)

        return  mu, log_var


class Decoder(nn.Module):
    """
    Decoder block
    Built to be a mirror of the encoder block
    """

    def __init__(self, channels, blocks=(1, 2, 4, 8), latent_channels=512):
        super(Decoder, self).__init__()
        self.conv_in = nn.Conv2d(latent_channels, blocks[-1], kernel_size=1, stride=1)

        widths_out = list(blocks)[::-1]
        widths_in = (list(blocks[1:]) + [blocks[-1]])[::-1]

        layer_blocks = [ResBlock(blocks[-1], blocks[-1]),
                        ResBlock(blocks[-1], blocks[-1])]

        for w_in, w_out in zip(widths_in, widths_out):
            layer_blocks.append(ResUp(w_in, w_out))

        self.res_blocks = nn.Sequential(*layer_blocks)

        self.conv_out = nn.Conv2d(blocks[0], channels, 3, 1, 1)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.act_fnc(self.conv_in(x))
        x = self.res_blocks(x)
        mu = torch.tanh(self.conv_out(x))
        
        return mu


class ConvVariationalAutoencoderEmbedResidual(BaseVariationalAutoencoder):
    """
    VAE network, uses the above encoder and decoder blocks
    """
    def __init__(self, layer_sizes, device, width, height, classes, hyperparameters={}):
        super().__init__(layer_sizes, device, width, height, classes, hyperparameters)
        """Res VAE Network
        channel_in  = number of channels of the image 
        z = the number of channels of the latent representation
        (for a 64x64 image this is the size of the latent vector)
        """
        
        channel_in=1

        self.latent_channels = hyperparameters["latent_channels"]

        self.encoder = Encoder(channel_in, blocks=layer_sizes, latent_channels=self.latent_channels)
        self.decoder = Decoder(channel_in, blocks=layer_sizes, latent_channels=self.latent_channels)

        
        self.N = torch.distributions.Normal(0, 1)

        self.latent_dim = self.get_latent_dim()
        embedding_dim = int(self.latent_dim[0] * self.latent_dim[1] * self.latent_dim[2])
        self.embedding = nn.Embedding(num_embeddings=len(classes), 
                                      embedding_dim=embedding_dim)
        

    def get_latent_dim(self):
        self.encoded_width = self.width
        self.encoded_height = self.height
        for _ in range(len(self.layer_sizes)):
            self.encoded_height//=2
            self.encoded_width//=2
        self.encoded_num_channels = self.latent_channels
        
        return (self.encoded_num_channels, self.encoded_width, self.encoded_height)

    def encode(self, x, labels=None):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(mu.shape).to(self.device)

        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)
    
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