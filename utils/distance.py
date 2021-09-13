import torch
import torch.nn as nn

eps = 1e-7  # Avoid calculating log(0). Use the small value of float16. It also works fine using 1e-35 (float32).

class KLDiv(nn.Module):
    # Calculate KL-Divergence
        
    def forward(self, predict, target):
       assert predict.ndimension()==2,'Input dimension must be 2'
       target = target.detach()

       # KL(T||I) = \sum T(logT-logI)
       predict += eps
       target += eps
       logI = predict.log()
       logT = target.log()
       TlogTdI = target * (logT - logI)
       kld = TlogTdI.sum(1)
       return kld


def pdist(sample_1, sample_2, norm=2, eps=1e-7):
    r"""Compute the matrix of all squared pairwise distances.
    Copied from https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/util.py
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)


def MMD(sample_1, sample_2, alpha=None):
    r"""The *unbiased* MMD test of :cite:`gretton2012kernel`.
    Modified from https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/statistics_diff.py
    The kernel used is
    .. math::
        k(x, x') = \sum_{j=1}^k e^{-\alpha_j \|x - x'\|^2},
    for the provided ``alphas``.
    Arguments
    ---------
    sample_1: :class:`torch:torch.autograd.Variable`
        The first sample, of size ``(n_1, d)``.
    sample_2: variable of shape (n_2, d)
        The second sample, of size ``(n_2, d)``.
    alphas : list of :class:`float`
        The kernel parameters.
    Returns
    -------
    :class:`float`
        The test statistic."""

    n_1 = sample_1.size(0)
    n_2 = sample_2.size(0)

    # The three constants used in the test.
    a00 = 1. / (n_1 * (n_1 - 1))
    a11 = 1. / (n_2 * (n_2 - 1))
    a01 = - 1. / (n_1 * n_2)

    sample_12 = torch.cat((sample_1, sample_2), 0)
    distances = pdist(sample_12, sample_12, norm=2)

    if alpha is None:
        v, _ = distances.view(-1).sort()
        alpha = 1. / v[distances.nelement()//2]**2

    kernels = torch.exp(- alpha * distances ** 2)

    k_1 = kernels[:n_1, :n_1]
    k_2 = kernels[n_1:, n_1:]
    k_12 = kernels[:n_1, n_1:]

    mmd = (2 * a01 * k_12.sum() +
            a00 * (k_1.sum() - torch.trace(k_1)) +
            a11 * (k_2.sum() - torch.trace(k_2)))

    return mmd