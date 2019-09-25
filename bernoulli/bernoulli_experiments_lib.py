import numpy as np

import os
import torch
import torch.nn as nn

import torch.optim as optim

from torch.distributions import Normal, Categorical, Bernoulli
import torch.nn.functional as F

import sys
sys.path.insert(0, '../../rb_utils/')
import rao_blackwellization_lib as rb_lib

import timeit

from copy import deepcopy

import itertools

softmax = nn.Softmax(dim = 0)
sigmoid = nn.Sigmoid()

def get_bernoulli_log_prob(e_b, draw):
    return torch.log(e_b + 1e-12) * torch.sum(draw) + \
                torch.log(1 - e_b + 1e-12) * torch.sum(1 - draw)

# class to run Bernoulli Experiments
class BernoulliExperiments(object):
    def __init__(self, p0, dim, phi0):
        # the target vector inside the expectation
        self.p0 = p0

        # dimension of the Bernoulli random variable
        self.dim = dim

        # define and cache the 2**d categories
        self.set_draw_array()

        # the variational parameter is just the probability of
        # the Bernoulli random variable
        self.var_params = {'phi': deepcopy(phi0)}

    def set_var_params(self, phi):
        self.var_params = {'phi': deepcopy(phi)}

    def set_draw_array(self):
        # defines the 2**d vector of possible combinations

        self.draw_array = torch.zeros((self.dim, 2**self.dim))
        i = 0
        for draw in itertools.product(range(2), repeat=self.dim):
            draw_tensor = torch.Tensor(draw)
            self.draw_array[:, i] = draw_tensor
            i += 1

    def f_z(self, i):
        # returns the loss for the ith entry in draw array
        draw = (self.draw_array * i).sum(dim = 1)
        return torch.Tensor([torch.sum((draw - self.p0) ** 2)])

    def get_log_q(self):
        # returns a vector of log probabilities for all the possible draws
        log_probs = torch.zeros((1, 2**self.dim))

        e_b = sigmoid(self.var_params['phi'])

        for i in range(2**self.dim):
            draw_tensor = self.draw_array[:, i]
            log_probs[0, i] = get_bernoulli_log_prob(e_b, draw_tensor)

        return log_probs

    def get_pm_loss(self, topk, grad_estimator, grad_estimator_kwargs, sample_topk=False, n_samples = 1):
        # returns the pseudo-loss: when backwards is called, this returns
        # an estimate of the gradient.

        log_q = self.get_log_q()

        pm_loss = 0.0
        for i in range(n_samples):
            pm_loss += rb_lib.get_raoblackwell_ps_loss(self.f_z, log_q, topk, sample_topk,
                                        grad_estimator, grad_estimator_kwargs=grad_estimator_kwargs)

        return pm_loss / n_samples

    def get_full_loss(self):
        # return the loss, fully marginalizing the discrete random variable
        log_q = self.get_log_q()
        class_weights = torch.exp(log_q)
        return rb_lib.get_full_loss(self.f_z, class_weights)


def sample_bern_gradient(phi0, bern_experiment, topk,
                            grad_estimator,
                            grad_estimator_kwargs,
                            sample_topk=False,
                            n_samples = 10000):
    # repeatedly compute the gradient for the Bernoulli experiment
    # return an array of samples

    params = [phi0]
    optimizer = optim.SGD(params, lr = 1.0)

    grad_array = torch.zeros(n_samples)

    for i in range(n_samples):
        bern_experiment.set_var_params(deepcopy(phi0))
        optimizer.zero_grad()
        ps_loss = bern_experiment.get_pm_loss(topk = topk, grad_estimator = grad_estimator,
                                              grad_estimator_kwargs=grad_estimator_kwargs, sample_topk=sample_topk)
        ps_loss.backward()

        grad_array[i] = bern_experiment.var_params['phi'].grad

    return grad_array
