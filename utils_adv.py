import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def pgd_whitebox(
    model,
    x,
    y,
    device,
    epsilon,
    num_steps,
    step_size,
    clip_min,
    clip_max,
    is_random=True,
    distance="linf",
):
    assert distance in ["linf", "l2"]

    if distance == "linf":
        if is_random:
            random_noise = (
                torch.FloatTensor(x.shape)
                .uniform_(-epsilon, epsilon)
                .to(device)
                .detach()
            )
        x_pgd = Variable(x.detach().data + random_noise, requires_grad=True)
        for _ in range(num_steps):
            with torch.enable_grad():
                loss = nn.CrossEntropyLoss()(model(x_pgd), y)
            loss.backward()
            x_pgd.data = x_pgd.data + step_size * x_pgd.grad.data.sign()
            eta = torch.clamp(x_pgd.data - x.data, -epsilon, epsilon)
            x_pgd.data = torch.clamp(x.data + eta, clip_min, clip_max)
            x_pgd.grad.data = torch.zeros_like(
                x_pgd.grad.data
            )  # zero out accumulated gradients

    if distance == "l2":
        if is_random:
            random_noise = (
                torch.FloatTensor(x.shape).uniform_(-1, 1).to(device).detach()
            )
            random_noise.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_pgd = Variable(x.detach().data + random_noise, requires_grad=True)
        for _ in range(num_steps):
            with torch.enable_grad():
                loss = nn.CrossEntropyLoss()(model(x_pgd), y)
            loss.backward()
            # renorming gradient
            grad_norms = x_pgd.grad.view(len(x), -1).norm(p=2, dim=1)
            x_pgd.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                x_pgd.grad[grad_norms == 0] = torch.randn_like(
                    x_pgd.grad[grad_norms == 0]
                )
            # optimizer_delta.step()
            x_pgd.data += step_size * x_pgd.grad.data
            eta = x_pgd.data - x.data
            eta.renorm_(p=2, dim=0, maxnorm=epsilon)
            x_pgd.data = torch.clamp(x.data + eta, clip_min, clip_max)
            x_pgd.grad.data = torch.zeros_like(
                x_pgd.grad.data
            )  # zero out accumulated gradients
    return x_pgd


# ref: https://github.com/yaodongyu/TRADES
def trades_loss(
    model,
    x_natural,
    y,
    device,
    optimizer,
    step_size,
    epsilon,
    perturb_steps,
    beta,
    clip_min,
    clip_max,
    distance="linf",
    natural_criterion=nn.CrossEntropyLoss(),
):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = (
        x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    )
    if distance == "linf":
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(
                    F.log_softmax(model(x_adv), dim=1),
                    F.softmax(model(x_natural), dim=1),
                )
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(
                torch.max(x_adv, x_natural - epsilon), x_natural + epsilon
            )
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    elif distance == "l2":
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(
                    F.log_softmax(model(adv), dim=1), F.softmax(model(x_natural), dim=1)
                )
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(
                    delta.grad[grad_norms == 0]
                )
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(clip_min, clip_max).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits, logits_adv = model(x_natural), model(x_adv)
    loss_natural = natural_criterion(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(
        F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1)
    )
    loss = loss_natural + beta * loss_robust
    return loss, logits, logits_adv