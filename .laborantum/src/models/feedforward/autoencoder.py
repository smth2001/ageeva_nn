import torch


class Autoencoder(torch.nn.Module):
    def __init__(
        self,
        channels,
        activation=torch.nn.ReLU
    ):
        super().__init__()

        encoder_layers = []
        for index in range(len(channels) - 1):
            encoder_layers.append(torch.nn.Linear(channels[index], channels[index + 1]))
            encoder_layers.append(activation())
        encoder_layers.pop()

        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_layers = []
        decoder_channels = channels[::-1]
        for index in range(len(decoder_channels) - 1):
            decoder_layers.append(torch.nn.Linear(decoder_channels[index], decoder_channels[index + 1]))
            decoder_layers.append(activation())
        decoder_layers.pop()

        self.decoder = torch.nn.Sequential(*decoder_layers)

    def __forward_kernel(self, signal):
        input_shape = signal.shape

        res = signal.reshape(signal.shape[0], -1)
        res = self.encoder(res)
        res = self.decoder(res)
        res = res.reshape(input_shape)

        return res

    def forward(self, batch):
        if isinstance(batch, dict):
            signal = batch['data']['image']
            reconstruction = self.__forward_kernel(signal)

            batch['signals'] = {
                'reconstruction': reconstruction
            }

            return batch

        return self.__forward_kernel(batch)

class Sampler(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.mu_regressor = torch.nn.Linear(channels, channels)
        self.logvar_regressor = torch.nn.Linear(channels, channels)

    def forward(self, signal):
        mu = self.mu_regressor(signal)
        logvar = self.logvar_regressor(signal)

        sigma = torch.exp(logvar / 2)

        if self.training:
            noise = torch.randn_like(mu)
            res = mu + noise * sigma
        else:
            res = mu

        return res, mu, sigma


class VAE(torch.nn.Module):
    def __init__(
        self,
        channels,
        activation=torch.nn.ReLU
    ):
        super().__init__()

        encoder_layers = []
        for index in range(len(channels) - 1):
            encoder_layers.append(torch.nn.Linear(channels[index], channels[index + 1]))
            encoder_layers.append(activation())
        encoder_layers.pop()

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.sampler = Sampler(channels[-1])

        decoder_layers = []
        decoder_channels = channels[::-1]
        for index in range(len(decoder_channels) - 1):
            decoder_layers.append(torch.nn.Linear(decoder_channels[index], decoder_channels[index + 1]))
            decoder_layers.append(activation())
        decoder_layers.pop()

        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, signal):
        input_shape = signal.shape

        res = signal.reshape([signal.shape[0], -1])
        res = self.encoder(res)
        res, mu, sigma = self.sampler(res)
        res = self.decoder(res)
        res = res.reshape(input_shape)

        return res, mu, sigma