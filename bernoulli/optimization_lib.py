import numpy as np

import torch

import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_SGD(get_loss, params,
                lr = 1.0, n_steps = 10000,
                get_full_loss = None,
                **kwargs):
    """
    Optimimize the loss function using SGD

    Parameters
    ----------
    get_loss : function
        A function that returns a ps_loss such that
        ps_loss.backward() returns an estimate of the gradient.
        in general, ps_loss might not equal the actual loss.
    params : dictionary
        Dictionary containing parameters to optimize
    lr : float
        Learning rate of SGD
    n_steps : int
        number of steps to run SGD
    get_full_loss : function
        A function that returns the actual loss, summed over
        the discrete random variables (optional)

    Returns
    ----------
    loss_array : np.array
        Array with loss at each step of the optimization
    opt_params :
        The parameters at the end of running SGD
    """

    # set up optimizer
    params_list = [{'params': params[key]} for key in params]
    optimizer = optim.SGD(params_list, lr = lr)

    loss_array = np.zeros(n_steps)

    for i in range(n_steps):
        # run gradient descent
        optimizer.zero_grad()

        loss = get_loss(**kwargs)
        loss.backward()
        optimizer.step()

        # save losses
        if get_full_loss is not None:
            full_loss = get_full_loss()
        else:
            full_loss = loss

        loss_array[i] = full_loss.detach().numpy()

    opt_params = params

    return loss_array, opt_params
