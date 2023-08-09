import torch


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


def load(package, pretrained=True):
    return Identity()
