import tensorflow as tf
from .gumbel import gumbel_with_maximum

def gather_topk(vals, ind):
    # https://stackoverflow.com/questions/54196149/how-to-use-indices-from-tf-nn-top-k-with-tf-gather-nd
    # This would have been one line in pytorch, should be an easier way?
    inds = tf.meshgrid(*(tf.range(s) for s in ind.shape), indexing='ij')
    # Stack complete index
    index = tf.stack(inds[:-1] + [ind], axis=-1)
    return tf.gather_nd(vals, index)


def beam_search(marglogp, k, stochastic=False):
    # marglogp should be [batch_size, n_dimensions, n_categories_per_dimension]
    # and should be normalized e.g. marglogp.exp().sum(-1) == 1
    # If stochastic is True, this is stochastic beam search https://arxiv.org/abs/1903.06059
    phi = marglogp[:, 0, :]
    criterium = phi
    if stochastic:
        g_phi, _ = gumbel_with_maximum(phi, tf.zeros(phi.shape[:-1]))
        criterium = g_phi

    crit_topk, ind_topk = tf.math.top_k(criterium, k)

    if stochastic:
        g_phi = crit_topk
        phi = gather_topk(phi, ind_topk)
    else:
        phi = crit_topk

    batch_size = phi.shape[0]
    n_dim = marglogp.shape[1]

    ind_first_action = ind_topk
    trace = []

    # Forward computation
    for i in range(1, n_dim):
        marglogpi = marglogp[:, i, :]

        num_actions = marglogpi.shape[-1]
        # expand_phi = [batch_size, num_parents, num_actions]
        expand_phi = phi[:, :, None] + marglogpi[:, None, :]
        expand_phi_flat = tf.reshape(expand_phi, [batch_size, -1])
        if stochastic:
            expand_g_phi, _ = gumbel_with_maximum(expand_phi, g_phi)
            criterium = tf.reshape(expand_g_phi, [batch_size, -1])
        else:
            criterium = expand_phi_flat

        crit_topk, ind_topk = tf.math.top_k(criterium, k)
        ind_parent, ind_action = ind_topk // num_actions, ind_topk % num_actions

        if stochastic:
            g_phi = crit_topk
            phi = gather_topk(expand_phi_flat, ind_topk)
        else:
            phi = crit_topk

        trace.append((ind_parent, ind_action))

    # Backtrack to get the sample
    prev_ind_parent = None
    actions = []
    for ind_parent, ind_action in reversed(trace):

        if prev_ind_parent is not None:
            ind_action = tf.batch_gather(ind_action, prev_ind_parent)
            ind_parent = tf.batch_gather(ind_parent, prev_ind_parent)
        actions.append(ind_action)
        prev_ind_parent = ind_parent

    if prev_ind_parent is None:
        actions.append
    actions.append(
        tf.batch_gather(ind_first_action, prev_ind_parent)
        if prev_ind_parent is not None
        else ind_first_action
    )
    return tf.stack(list(reversed(actions)), axis=-1), phi, g_phi if stochastic else None
