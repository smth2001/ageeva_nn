import copy
import torch

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.strength * grad_output, None


class GradientReversalLayer(torch.nn.Module):
    def __init__(self, strength=1.0):
        super().__init__()
        self.strength = float(strength)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.strength)

class GAN(torch.nn.Module):
    def __init__(
        self,
        channels,
        gradient_reversal_strength=1.0,
        activation=torch.nn.LeakyReLU(negative_slope=0.5),
    ):
        super().__init__()

        self.channels = list(channels)

        self.generator_discriminator_bridge = GradientReversalLayer(
            gradient_reversal_strength
        )
        self.gradient_reversal = self.generator_discriminator_bridge

        generator_layers = []
        for in_features, out_features in zip(self.channels[:-1], self.channels[1:]):
            generator_layers.append(torch.nn.Linear(in_features, out_features))
            if out_features != self.channels[-1]:
                generator_layers.append(copy.deepcopy(activation))
        generator_layers.append(torch.nn.Tanh())
        self.generator = torch.nn.Sequential(*generator_layers)

        discriminator_channels = list(reversed(self.channels))
        discriminator_layers = []
        for in_features, out_features in zip(
            discriminator_channels[:-1],
            discriminator_channels[1:],
        ):
            discriminator_layers.append(torch.nn.Linear(in_features, out_features))
            if out_features != discriminator_channels[-1]:
                discriminator_layers.append(copy.deepcopy(activation))
        self.discriminator = torch.nn.Sequential(*discriminator_layers)

        self.classifier = torch.nn.Linear(self.channels[0], 1)
    def forward(self, batch):
        noise = batch['data']['noise']
        image = batch['data']['image']

        batch_size = noise.shape[0]

        generated = self.generator(noise)

        real_flat = image.view(batch_size, -1)

        fake_input = self.generator_discriminator_bridge(generated)
        discriminator_input = torch.cat([fake_input, real_flat], dim=0)

        discriminator_features = self.discriminator(discriminator_input)
        discriminator_logits = self.classifier(discriminator_features).squeeze(-1)

        fake_logits = discriminator_logits[:batch_size]
        real_logits = discriminator_logits[batch_size:]

        batch['signals'] = {
            'generated': generated,
            'discriminator_logits': discriminator_logits,
            'fake_logits': fake_logits,
            'real_logits': real_logits,
            'discriminator_scores': discriminator_logits,
            'fake_scores': fake_logits,
            'real_scores': real_logits,
        }

        batch['postprocessed'] = {
            'discriminator_score': discriminator_logits,
            'fake_score': fake_logits,
            'real_score': real_logits,
            'discriminator_probability': torch.sigmoid(discriminator_logits),
            'fake_probability': torch.sigmoid(fake_logits),
            'real_probability': torch.sigmoid(real_logits),
        }

        return batch