# This library contains various gradient estimators
# including REINFORCE, REINFORCE+,
# REBAR/RELAX, NVIL, and gumbel_softmax

import numpy as np

import torch
import torch.nn as nn

from torch.distributions import Categorical, Gumbel
from common_utils import get_one_hot_encoding_from_int, sample_class_weights
from gumbel import compute_log_R, compute_log_R_O_nfac, log1mexp, all_perms, log_pl_rec
import torch.nn.functional as F

import gumbel_softmax_lib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_reinforce_grad_sample(conditional_loss, log_class_weights,
                                baseline = 0.0):
    # computes the REINFORCE gradient estimate
    assert len(conditional_loss) == len(log_class_weights)

    return (conditional_loss - baseline).detach() * log_class_weights


"""
Below are the gradient estimates for
REINFORCE, REINFORCE+, REBAR/RELAX, NVIL, and Gumbel-softmax.
Each follow the pattern,

Parameters
----------
conditional_loss_fun : function
    A function that returns the loss conditional on an instance of the
    categorical random variable. It must take in a one-hot-encoding
    matrix (batchsize x n_categories) and return a vector of
    losses, one for each observation in the batch.
log_class_weights : torch.Tensor
    A tensor of shape batchsize x n_categories of the log class weights
class_weights_detached : torch.Tensor
    A tensor of shape batchsize x n_categories of the class weights.
    Must be detached, i.e. we do not compute gradients
seq_tensor : torch.Tensor
    A tensor containing values \{1 ... batchsize\}
    TODO: is there a way to cache this?
z_sample : torch.Tensor
    The cateories (not one-hot-encoded) at which to evaluate the ps loss.
epoch : int
    The epoch of the optimizer (for Gumbel-softmax, which has an annealing rate)
data : torch.Tensor
    The data at which we evaluate the loss (for NVIl and RELAX, which have
    a data dependent baseline)
grad_estimator_kwargs : dict
    Additional arguments to the gradient estimators

Returns
-------
ps_loss :
    a value such that ps_loss.backward() returns an
    estimate of the gradient.
    In general, ps_loss might not equal the actual loss.
"""

def reinforce(conditional_loss_fun, log_class_weights,
                class_weights_detached, seq_tensor,
                z_sample, epoch, data, grad_estimator_kwargs = None, baseline=0.0):
    # z_sample should be a vector of categories
    # conditional_loss_fun is a function that takes in a one hot encoding
    # of z and returns the loss

    assert len(z_sample) == log_class_weights.shape[0]

    # compute loss from those categories
    n_classes = log_class_weights.shape[1]
    one_hot_z_sample = get_one_hot_encoding_from_int(z_sample, n_classes)
    conditional_loss_fun_i = conditional_loss_fun(one_hot_z_sample)
    assert len(conditional_loss_fun_i) == log_class_weights.shape[0]

    # get log class_weights
    log_class_weights_i = log_class_weights[seq_tensor, z_sample]
    
    return get_reinforce_grad_sample(conditional_loss_fun_i,
                    log_class_weights_i, baseline = baseline) + \
                        conditional_loss_fun_i

def get_baseline(conditional_loss_fun, log_class_weights, n_samples, deterministic):

    with torch.no_grad():
        n_classes = log_class_weights.size(-1)
        if deterministic:
            # Compute baseline deterministically as normalized weighted topk elements
            weights, ind = log_class_weights.topk(n_samples, -1)
            weights = weights - weights.logsumexp(-1, keepdim=True)  # normalize
            baseline = (torch.stack([
                conditional_loss_fun(get_one_hot_encoding_from_int(ind[:, i], n_classes))
                for i in range(n_samples)
            ], -1) * weights.exp()).sum(-1)
        else:
            # Sample with replacement baseline and compute mean
            baseline = torch.stack([
                conditional_loss_fun(get_one_hot_encoding_from_int(sample_class_weights(log_class_weights.exp()), n_classes))
                for i in range(n_samples)
            ], -1).mean(-1)
    return baseline

rf_cache = {
    'prev_data': None,
    'prev_baseline': None
}
def reinforce_w_double_sample_baseline(\
            conditional_loss_fun, log_class_weights,
            class_weights_detached, seq_tensor, z_sample,
            epoch, data,
           baseline_reuse=False,
           baseline_n_samples=1,
           baseline_deterministic=False,
            grad_estimator_kwargs = None):
    # This is what we call REINFORCE+ in our paper,
    # where we use a second, independent sample from the discrete distribution
    # to use as a baseline

    assert len(z_sample) == log_class_weights.shape[0]

    # compute loss from those categories
    n_classes = log_class_weights.shape[1]
    one_hot_z_sample = get_one_hot_encoding_from_int(z_sample, n_classes)
    conditional_loss_fun_i = conditional_loss_fun(one_hot_z_sample)
    assert len(conditional_loss_fun_i) == log_class_weights.shape[0]

    # get log class_weights
    log_class_weights_i = log_class_weights[seq_tensor, z_sample]

    if baseline_reuse and data is rf_cache['prev_data']:
        # This way we use the same baseline for all topk and the sample
        baseline = rf_cache['prev_baseline']
    else:
        baseline = get_baseline(conditional_loss_fun, log_class_weights, baseline_n_samples, baseline_deterministic)
        # get baseline
        # z_sample2 = sample_class_weights(class_weights_detached)
        # one_hot_z_sample2 = get_one_hot_encoding_from_int(z_sample2, n_classes)
        # baseline = conditional_loss_fun(one_hot_z_sample2)
        if baseline_reuse:
            rf_cache['prev_baseline'] = baseline
            rf_cache['prev_data'] = data

    return get_reinforce_grad_sample(conditional_loss_fun_i,
                    log_class_weights_i, baseline) + conditional_loss_fun_i

def reinforce_wr(conditional_loss_fun, log_class_weights,
                class_weights_detached, seq_tensor,
                z_sample, epoch, data, n_samples = 1,
                baseline_constant=None):
    # z_sample should be a vector of categories, but is ignored as this function samples itself
    # conditional_loss_fun is a function that takes in a one hot encoding
    # of z and returns the loss

    assert len(z_sample) == log_class_weights.shape[0]

    # Sample with replacement
    ind = torch.stack([sample_class_weights(class_weights_detached) for _ in range(n_samples)], -1)

    log_p = log_class_weights.gather(-1, ind)
    n_classes = log_class_weights.shape[1]
    costs = torch.stack([
        conditional_loss_fun(get_one_hot_encoding_from_int(z_sample, n_classes))
        for z_sample in ind.t()
    ], -1)

    if baseline_constant is None:
        adv = (costs - costs.mean(-1, keepdim=True)) * n_samples / (n_samples - 1)
    else:
        adv = costs - baseline_constant

    # Add the costs in case there is a direct dependency on the parameters
    return (adv.detach() * log_p + costs).mean(-1)

def reinforce_unordered(conditional_loss_fun, log_class_weights,
                class_weights_detached, seq_tensor,
                z_sample, epoch, data, n_samples = 1,
                baseline_separate=False,
               baseline_n_samples=1,
               baseline_deterministic=False,
                baseline_constant=None):

    # Sample without replacement using Gumbel top-k trick
    phi = log_class_weights.detach()
    g_phi = Gumbel(phi, torch.ones_like(phi)).rsample()
    _, ind = g_phi.topk(n_samples, -1)

    log_p = log_class_weights.gather(-1, ind)
    n_classes = log_class_weights.shape[1]
    costs = torch.stack([
        conditional_loss_fun(get_one_hot_encoding_from_int(z_sample, n_classes))
        for z_sample in ind.t()
    ], -1)

    with torch.no_grad():  # Don't compute gradients for advantage and ratio
        # log_R_s, log_R_ss = compute_log_R(log_p)
        log_R_s, log_R_ss = compute_log_R_O_nfac(log_p)

        if baseline_constant is not None:
            bl_vals = baseline_constant
        elif baseline_separate:
            bl_vals = get_baseline(conditional_loss_fun, log_class_weights, baseline_n_samples, baseline_deterministic)
            # Same bl for all samples, so add dimension
            bl_vals = bl_vals[:, None]
        elif log_p.size(-1) > 1:
            # Compute built in baseline
            bl_vals = ((log_p[:, None, :] + log_R_ss).exp() * costs[:, None, :]).sum(-1)
        else:
            bl_vals = 0.  # No bl
        adv = costs - bl_vals
    # Also add the costs (with the unordered estimator) in case there is a direct dependency on the parameters
    loss = ((log_p + log_R_s).exp() * adv.detach() + (log_p + log_R_s).exp().detach() * costs).sum(-1)
    return loss


def reinforce_sum_and_sample(conditional_loss_fun, log_class_weights,
                class_weights_detached, seq_tensor,
                z_sample, epoch, data, n_samples = 1,
                baseline_separate=False,
               baseline_n_samples=1,
               baseline_deterministic=False, rao_blackwellize=False):

    # Sample without replacement using Gumbel top-k trick
    phi = log_class_weights.detach()
    g_phi = Gumbel(phi, torch.ones_like(phi)).rsample()

    _, ind = g_phi.topk(n_samples, -1)

    log_p = log_class_weights.gather(-1, ind)
    n_classes = log_class_weights.shape[1]
    costs = torch.stack([
        conditional_loss_fun(get_one_hot_encoding_from_int(z_sample, n_classes))
        for z_sample in ind.t()
    ], -1)

    with torch.no_grad():  # Don't compute gradients for advantage and ratio
        if baseline_separate:
            bl_vals = get_baseline(conditional_loss_fun, log_class_weights, baseline_n_samples, baseline_deterministic)
            # Same bl for all samples, so add dimension
            bl_vals = bl_vals[:, None]
        else:
            assert baseline_n_samples < n_samples
            bl_sampled_weight = log1mexp(log_p[:, :baseline_n_samples-1].logsumexp(-1)).exp().detach()
            bl_vals = (log_p[:, :baseline_n_samples - 1].exp() * costs[:, :baseline_n_samples -1]).sum(-1)\
                      + bl_sampled_weight * costs[:, baseline_n_samples - 1]
            bl_vals = bl_vals[:, None]

    # We compute an 'exact' gradient if the sum of probabilities is roughly more than 1 - 1e-5
    # in which case we can simply sum al the terms and the relative error will be < 1e-5
    use_exact = log_p.logsumexp(-1) > -1e-5
    not_use_exact = use_exact == 0

    cost_exact = costs[use_exact]
    exact_loss = compute_summed_terms(log_p[use_exact], cost_exact, cost_exact - bl_vals[use_exact])

    log_p_est = log_p[not_use_exact]
    costs_est = costs[not_use_exact]
    bl_vals_est = bl_vals[not_use_exact]

    if rao_blackwellize:
        ap = all_perms(torch.arange(n_samples, dtype=torch.long), device=log_p_est.device)
        log_p_ap = log_p_est[:, ap]
        bl_vals_ap = bl_vals_est.expand_as(costs_est)[:, ap]
        costs_ap = costs_est[:, ap]
        cond_losses = compute_sum_and_sample_loss(log_p_ap, costs_ap, bl_vals_ap)

        # Compute probabilities for permutations
        log_probs_perms = log_pl_rec(log_p_ap, -1)
        cond_log_probs_perms = log_probs_perms - log_probs_perms.logsumexp(-1, keepdim=True)
        losses = (cond_losses * cond_log_probs_perms.exp()).sum(-1)
    else:
        losses = compute_sum_and_sample_loss(log_p_est, costs_est, bl_vals_est)

    # If they are summed we can simply concatenate but for consistency it is best to place them in order
    all_losses = log_p.new_zeros(log_p.size(0))
    all_losses[use_exact] = exact_loss
    all_losses[not_use_exact] = losses
    return all_losses


def compute_sum_and_sample_loss(log_p, costs, baseline):
    adv = (costs - baseline).detach()
    sampled_weight = log1mexp(log_p[..., :-1].logsumexp(-1)).exp().detach()
    # reinf_loss_topk = (log_p[..., :-1].exp() * adv[..., :-1].detach()).sum(-1)
    reinf_loss_sample = log_p[..., -1] * adv[..., -1].detach()

    # pathwise_loss_topk = (log_p[..., :-1].exp().detach() * costs[..., :-1]).sum(-1)
    pathwise_loss_sample = costs[..., -1]

    loss_topk = compute_summed_terms(log_p[..., :-1], costs[..., :-1], adv[..., :-1])

    return loss_topk + sampled_weight * (reinf_loss_sample + pathwise_loss_sample)


def compute_summed_terms(log_p, costs, adv):
    reinf_loss = (log_p.exp() * adv.detach()).sum(-1)
    pathwise_loss = (log_p.exp().detach() * costs).sum(-1)
    return reinf_loss + pathwise_loss


class RELAXBaseline(nn.Module):
    def __init__(self, input_dim):
        # this is a neural network for the NVIL baseline
        super(RELAXBaseline, self).__init__()

        # image / model parameters
        self.input_dim = input_dim

        # define the linear layers
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):

        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)

        return h

def relax(conditional_loss_fun, log_class_weights,
            class_weights_detached, seq_tensor, z_sample,
            epoch, data,
            temperature = torch.Tensor([1.0]),
            eta = 1.,
            c_phi = lambda x : torch.Tensor([0.0])):
    # with the default c_phi value, this is just REBAR
    # RELAX adds a learned component c_phi

    # sample gumbel
    gumbel_sample = log_class_weights + \
        gumbel_softmax_lib.sample_gumbel(log_class_weights.size())

    # get hard z
    _, z_sample = gumbel_sample.max(dim=-1)
    n_classes = log_class_weights.shape[1]
    z_one_hot = get_one_hot_encoding_from_int(z_sample, n_classes)
    temperature = torch.clamp(temperature, 0.01, 5.0)

    # get softmax z
    z_softmax = F.softmax(gumbel_sample / temperature[0], dim=-1)

    # conditional softmax z
    z_cond_softmax = \
        gumbel_softmax_lib.gumbel_softmax_conditional_sample(\
            log_class_weights, temperature[0], z_one_hot)

    # get log class_weights
    log_class_weights_i = log_class_weights[seq_tensor, z_sample]

    # reinforce term
    f_z_hard = conditional_loss_fun(z_one_hot.detach())
    f_z_softmax = conditional_loss_fun(z_softmax)
    f_z_cond_softmax = conditional_loss_fun(z_cond_softmax)

    # baseline terms
    c_softmax = c_phi(z_softmax).squeeze()
    z_cond_softmax_detached = \
        gumbel_softmax_lib.gumbel_softmax_conditional_sample(\
            log_class_weights, temperature[0], z_one_hot, detach = True)
    c_cond_softmax = c_phi(z_cond_softmax_detached).squeeze()

    reinforce_term = \
        (f_z_hard - eta * (f_z_cond_softmax - c_cond_softmax)).detach() * \
                        log_class_weights_i + \
                        log_class_weights_i * eta * c_cond_softmax

    # correction term
    correction_term = eta * (f_z_softmax - c_softmax) - \
                        eta * (f_z_cond_softmax - c_cond_softmax)

    return reinforce_term + correction_term + f_z_hard

def gumbel(conditional_loss_fun, log_class_weights,
            class_weights_detached, seq_tensor, z_sample,
            epoch, data,
            annealing_fun,
            straight_through = True):

    # get temperature
    temperature = annealing_fun(epoch)

    # sample gumbel
    if straight_through:
        gumbel_sample = \
            gumbel_softmax_lib.gumbel_softmax(log_class_weights, temperature)
    else:
        gumbel_sample = \
            gumbel_softmax_lib.gumbel_softmax_sample(\
                    log_class_weights, temperature)

    f_gumbel = conditional_loss_fun(gumbel_sample)

    return f_gumbel

class BaselineNN(nn.Module):
    def __init__(self, slen = 28):
        # this is a neural network for the NVIL baseline
        super(BaselineNN, self).__init__()

        # image / model parameters
        self.n_pixels = slen ** 2
        self.slen = slen

        # define the linear layers
        self.fc1 = nn.Linear(self.n_pixels, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 1)


    def forward(self, image):

        # feed through neural network
        h = image.view(-1, self.n_pixels)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = self.fc4(h)

        return h


def nvil(conditional_loss_fun, log_class_weights,
            class_weights_detached, seq_tensor, z_sample,
            epoch, data,
            baseline_nn):

    assert len(z_sample) == log_class_weights.shape[0]

    # compute loss from those categories
    n_classes = log_class_weights.shape[1]
    one_hot_z_sample = get_one_hot_encoding_from_int(z_sample, n_classes)
    conditional_loss_fun_i = conditional_loss_fun(one_hot_z_sample)
    assert len(conditional_loss_fun_i) == log_class_weights.shape[0]

    # get log class_weights
    log_class_weights_i = log_class_weights[seq_tensor, z_sample]

    # get baseline
    baseline = baseline_nn(data).squeeze()

    return get_reinforce_grad_sample(conditional_loss_fun_i,
                    log_class_weights_i, baseline = baseline) + \
                        conditional_loss_fun_i + \
                        (conditional_loss_fun_i.detach() - baseline)**2
