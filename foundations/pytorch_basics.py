import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType


class Solution:

  def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
    # Reshape (M, N) tensor to (M*N/2, 2) using total elements
    total_elements = to_reshape.numel()
    reshaped = torch.reshape(to_reshape, (total_elements // 2, 2))
    return torch.round(reshaped, decimals=4)

  def average(self, to_avg: TensorType[float]) -> TensorType[float]:
    # Compute column-wise mean (average across rows)
    avg = torch.mean(to_avg, dim=0)
    return torch.round(avg, decimals=4)

  def concatenate(
      self, cat_one: TensorType[float], cat_two: TensorType[float]
  ) -> TensorType[float]:
    # Join two tensors side-by-side along dim=1
    cat = torch.cat((cat_one, cat_two), dim=1)
    return torch.round(cat, decimals=4)

  def get_loss(
      self, prediction: TensorType[float], target: TensorType[float]
  ) -> TensorType[float]:
    # Compute Mean Squared Error between prediction and target
    loss = F.mse_loss(prediction, target)
    return torch.round(loss, decimals=4)