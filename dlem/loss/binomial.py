import torch
from torch import nn

class DirectMultinomialNLLLoss1D(nn.Module):
    def __init__(self):
        super(DirectMultinomialNLLLoss1D, self).__init__()

    def forward(self, probs, target):
        # Ensure probs are in log space
        #log_probs = torch.log(probs)
        log_probs = probs

        # Compute the negative log likelihood (without the combinatorial factor)
        nll = -torch.sum(target * log_probs)

        # Factorial terms (log factorial of target sums and each target)
        target_sum = target.sum()
        log_factorial_target_sum = torch.lgamma(target_sum + 1)
        log_factorial_target = torch.lgamma(target + 1).sum()

        # Final loss: nll + log_factorial_target_sum - log_factorial_target
        loss_total = nll - log_factorial_target_sum + log_factorial_target

        return loss_total.mean()

#loss_fn = DirectMultinomialNLLLoss1D()
#probs = torch.tensor([0.3, 0.2, 0.5], requires_grad=True)
#target = torch.tensor([1.0, 3.0, 6.0])
#loss = loss_fn(probs, target)
#print(loss)
#
#
#probs = torch.tensor([0.3, 0.2, 0.5], requires_grad=True)
#target = torch.tensor([1.0, 3.0, 6.0])*100
#loss = loss_fn(probs, target)
#print(loss)
