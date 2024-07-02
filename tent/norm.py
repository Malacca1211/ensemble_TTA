from copy import deepcopy

import torch
import torch.nn as nn


class Norm(nn.Module):
    """Norm adapts a model by estimating feature statistics during testing.

    Once equipped with Norm, the model normalizes its features during testing
    with batch-wise statistics, just like batch norm does during training.
    """

    def __init__(self, model, eps=1e-5, momentum=0.1,
                 reset_stats=False, no_stats=False):
        super().__init__()
        self.model = model
        self.model = configure_model(model, eps, momentum, reset_stats,
                                     no_stats)
        self.model_state = deepcopy(self.model.state_dict())

    def forward(self, x, weights):
        self.outputs = self.model(x)
        # earlyOutput
        outputList = self.model(x)
        # outputList = [outputList]
        self.outputs = outputList[-1]

        # weights = [0,0,1]
        # 计算加权平均
        weighted_sum = torch.zeros_like(outputList[0])
        total_weight = sum(weights)
        for i, tensor in enumerate(outputList):
            weighted_sum += weights[i] * tensor

        output = weighted_sum
        # output = weighted_sum / total_weight

        return output

    def backward_(self, label, optimizer):
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(self.outputs, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)


def collect_stats(model):
    """Collect the normalization stats from batch norms.

    Walk the model's modules and collect all batch normalization stats.
    Return the stats and their names.
    """
    stats = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            state = m.state_dict()
            if m.affine:
                del state['weight'], state['bias']
            for ns, s in state.items():
                stats.append(s)
                names.append(f"{nm}.{ns}")
    return stats, names


def configure_model(model, eps, momentum, reset_stats, no_stats):
    """Configure model for adaptation by test-time normalization."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # use batch-wise statistics in forward
            m.train()
            # configure epsilon for stability, and momentum for updates
            m.eps = eps
            m.momentum = momentum
            if reset_stats:
                # reset state to estimate test stats without train stats
                m.reset_running_stats()
            if no_stats:
                # disable state entirely and use only batch stats
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)