# coding=utf-8

import torch
from torch import nn


class MyNetwork(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = input + 1
        return output


net = MyNetwork()
x = torch.tensor(1.0)
output = net(x)
print(output)