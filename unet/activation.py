from typing import Any
import torch.nn.functional as F
import utils
from torch import nn


def _custom_op(x):
    x = F.relu(x, inplace=True)
    return x ** 2


class PySiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x) -> Any:
        return x / (1 + _custom_op(-x))


def pysilu(x):
    obj = PySiLU()
    return obj(x)

# utils.show_activation_curve(pysilu)