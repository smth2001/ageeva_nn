import torchvision.datasets

class MNISTSimpleDataset:
    def __init__(self, train=True):
        self.mnist = torchvision.datasets.MNIST(
            root="~/",
            train=train,
            download=True
        )

        self.X = self.mnist.data
        self.y = self.mnist.targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        image = self.X[index].float()

        # normalize from [0, 255] to [-1, 1]
        image = image / 255.0
        image = image * 2.0 - 1.0

        label = self.y[index].long()

        sample = {
            "image": image,
            "label": label
        }

        return sample
