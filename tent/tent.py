from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, weights):
        if self.episodic:
            self.reset()
        # outputs = self.model(x)
        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer, weights)
        self.outputs = outputs

        return outputs

    @torch.enable_grad()
    def backward_(self, outputEn, label):
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputEn, label)

        loss.backward(retain_graph=True)
        # print(self.model.module.bn1.weight.grad)
        self.optimizer.step()
        self.optimizer.zero_grad()

    @torch.enable_grad()
    def backward_Cut(self, outputEn, label):
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputEn, label)

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.clone())

        # 对梯度进行截断
        max_gradient_norm = 10  # 截断的梯度阈值
        for grad in gradients:
            grad.clamp_(-max_gradient_norm, max_gradient_norm)  # 将梯度限制在[-max_gradient_norm, max_gradient_norm]范围内

        # 更新模型参数
        self.optimizer.step()
        self.optimizer.zero_grad()

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, weights=[]):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    # outputs = model(x)
    # earlyOutput
    outputList = model(x)

    weights = weights #[0.05, 0.05, 0, 0.2, 0.6]
    # weights = [0,0,1]

    # 端到端模型加一层
    # outputList = [outputList]
    # 计算加权平均
    weighted_sum = torch.zeros_like(outputList[0])
    total_weight = sum(weights)
    for i, tensor in enumerate(outputList):
        weighted_sum += weights[i] * tensor
    # last_three_outputs = outputList[-5:]
    # outputs = sum(last_three_outputs) / len(last_three_outputs)
    outputs = weighted_sum

        # adapt
    loss = softmax_entropy(outputList[-1]).mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # naive ensemble
    return outputs
    # tent
    # return outputList
@torch.enable_grad()
def backward_adapt(outputs, label, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # adapt
    # loss = softmax_entropy(outputs).mean(0)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, label)
    loss.backward(retain_graph=True)
    optimizer.step()
    optimizer.zero_grad()


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # if nm.startswith('module.aux_classifier') or nm.startswith('module.decoder'):
        #     break
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statistics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
