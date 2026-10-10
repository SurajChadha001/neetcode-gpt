import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        stats = []
        hooks = []

        def hook_fn(module, input, output):
            mean = round(output.mean().item(), 4)
            std = round(output.std().item(), 4) if output.numel() > 1 else 0.0
            
            # Compute dead fraction per neuron across the batch dimension (dim=0)
            if output.ndim >= 2:
                dead_mask = (output <= 0).all(dim=0)
                dead_fraction = round(dead_mask.float().mean().item(), 4)
            else:
                dead_fraction = round(float((output <= 0).any().item()), 4)
                
            stats.append({
                'mean': mean,
                'std': std,
                'dead_fraction': dead_fraction
            })

        # Register forward hook on all nn.Linear layers
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                hooks.append(layer.register_forward_hook(hook_fn))

        with torch.no_grad():
            model(x)

        # Clean up hooks
        for h in hooks:
            h.remove()

        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        model.zero_grad()
        
        pred = model(x)
        loss_fn = nn.MSELoss()
        loss = loss_fn(pred, y)
        loss.backward()
        
        stats = []
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                if layer.weight.grad is not None:
                    grad = layer.weight.grad
                    mean = round(grad.mean().item(), 4)
                    std = round(grad.std().item(), 4) if grad.numel() > 1 else 0.0
                    norm = round(grad.norm().item(), 4)
                else:
                    mean, std, norm = 0.0, 0.0, 0.0
                    
                stats.append({
                    'mean': mean,
                    'std': std,
                    'norm': norm
                })
                
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Check in priority order: dead_neurons -> exploding_gradients -> vanishing_gradients -> healthy
        for stat in activation_stats:
            if stat.get('dead_fraction', 0.0) > 0.5:
                return 'dead_neurons'
                
        for stat in gradient_stats:
            if stat.get('norm', 0.0) > 100.0:
                return 'exploding_gradients'
                
        if gradient_stats and all(stat.get('norm', 0.0) < 1e-3 for stat in gradient_stats):
            return 'vanishing_gradients'
            
        return 'healthy'