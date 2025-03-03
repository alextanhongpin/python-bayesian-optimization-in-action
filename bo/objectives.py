import torch
from math import pi


def forrester(x: torch.Tensor) -> torch.Tensor:
    # a modification of https://www.sfu.ca/~ssurjano/forretal08.html
    y = -((x + 1) ** 2) * torch.sin(2 * x + 2) / 5 + 1

    return y


def forrester_1d(x: torch.Tensor) -> torch.Tensor:
    return forrester(x).squeeze(-1)


def ackley(x: torch.Tensor) -> torch.Tensor:
    # a modification of https://www.sfu.ca/~ssurjano/ackley.html
    return -20 * torch.exp(
        -0.2 * torch.sqrt((x[:, 0] ** 2 + x[:, 1] ** 2) / 2)
    ) - torch.exp(torch.cos(2 * pi * x[:, 0] / 3) + torch.cos(2 * pi * x[:, 1]))
