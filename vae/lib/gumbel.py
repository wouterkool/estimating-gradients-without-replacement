import tensorflow as tf
from itertools import permutations


def gumbel_with_maximum(phi, T, axis=-1):
    g_phi = phi - tf.log(-tf.log(tf.random.uniform(phi.shape)))
    Z, argmax = tf.reduce_max(g_phi, axis=-1), tf.argmax(g_phi, axis=-1)
    g = shift_gumbel_maximum(g_phi, T, axis, Z=Z)
    return g, argmax


def shift_gumbel_maximum(g_phi, T, axis=-1, Z=None):
    g = _shift_gumbel_maximum(g_phi, T, axis, Z)
    g_inv = _shift_gumbel_maximum(g, Z, axis)

    CHECK_VALIDITY = True
    if CHECK_VALIDITY:
        check = tf.reduce_all(((g_phi - g_inv) < 1e-3) | (g_phi == g_inv))
        # print("Check", check)#tf.assert(check)

    return g


def _shift_gumbel_maximum(g_phi, T, axis=-1, Z=None):
    if Z is None:
        Z = tf.reduce_max(g_phi, axis=axis)
    T_ = tf.expand_dims(T, axis=axis)
    u = T_ - g_phi + tf.log1p(-tf.exp(g_phi - tf.expand_dims(Z, axis=axis)))
    return T_ - tf.nn.relu(u) - tf.nn.softplus(-tf.abs(u))


def log1mexp(x):
    # Computes log(1-exp(-|x|))
    # See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    x = -tf.abs(x)
    return tf.where(x > -0.693, tf.log(-tf.expm1(x)), tf.log1p(-tf.exp(x)))


def all_perms(k):
    with tf.variable_scope("all_perms", reuse=tf.AUTO_REUSE):
        allp = tf.get_variable(f"perms_{k}", dtype=tf.int32,
              initializer=tf.constant(list(permutations(range(k)))))
    return allp


def all_2nd_order_perms(k):
    ap = all_perms(k)
    apf = tf.reshape(tf.boolean_mask(ap, ap[:, 0] < ap[:, 1]), [k * (k - 1) // 2, -1, k])
    return apf[:, 0, :2], apf[:, :, 2:]


def log_pl_rec(log_p, dim=-1):
    """Recursive function of Plackett Luce log probability has better numerical stability
    since 1 - sum_i p_i can get very close to 0, this version never computes sum p_i directly"""
    assert dim == -1
    if log_p.shape[-1] == 1:
        return log_p[..., 0]
    return log_p[..., 0] + log_pl_rec(log_p[..., 1:] - log1mexp(log_p[..., 0:1]), dim=dim)


SO_PERM_CACHE = {}
def compute_log_R_O_nfac(log_p, so_perms=None):
    """
    Computes all first and second order log ratio's by computing P(S)
    for all second order sets leaving two elements out of S
    where the individual P(S) are computed by naive enumeration of all permutations
    This is inefficient especially for large sample sizes but can be used
    to validate alternative implementations
    """

    k = int(log_p.shape[-1])
    if so_perms is None:
        if k in SO_PERM_CACHE:
            so_perms = SO_PERM_CACHE[k]
        else:
            so_perms = all_2nd_order_perms(k)
            SO_PERM_CACHE[k] = so_perms

        # perm_ids = all_perms(torch.arange(k - 2, dtype=torch.long), device=log_p.device)

    keys, rest = so_perms
    first, second = tf.unstack(keys, axis=-1)

    norm1 = log1mexp(tf.gather(log_p, first, axis=-1))
    norm2 = norm1 + log1mexp(tf.gather(log_p, second, axis=-1) - norm1)

    # Index to get
    # (batch_size, num_second_orders, num_perms, rest=k-2)
    log_p_rest = tf.gather(log_p, rest, axis=-1) - norm2[..., None, None]

    # (batch_size, num_second_orders, num_perms)
    logprobs = log_pl_rec(log_p_rest, -1)

    # (batch_size, num_second_orders)
    log_P = tf.reduce_logsumexp(logprobs, axis=-1)

    # We build the 2d matrix of second order values as a list of list (of batch values)
    # such that we can convert it to a tensor later
    # Probably should also be possible with some scatter functionality
    ind = 0
    log_P2s_list = [[None] * k for i in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            log_P2_ij = log_P[:, ind]
            log_P2s_list[i][j] = log_P2_ij
            log_P2s_list[j][i] = log_P2_ij
            ind += 1

    # Compute first order log_P
    for i in range(k):
        # P(S) = sum_{s in S} p(s) P^{D\s}(S\s)
        log_p_without_i = tf.concat((log_p[:, :i], log_p[:, i + 1:]), axis=-1) - log1mexp(log_p[:, i, None])
        log_P2s_without_i = tf.stack(log_P2s_list[i][:i] + log_P2s_list[i][i + 1:], axis=-1)
        log_P1_i = tf.reduce_logsumexp(log_p_without_i + log_P2s_without_i, axis=-1)
        log_P2s_list[i][i] = log_P1_i

    log_P2s_list_flat = [log_P2s_list[i][j] for i in range(k) for j in range(k)]
    log_P2s = tf.reshape(tf.stack(log_P2s_list_flat, axis=1), [-1, k, k])
    log_P1s = tf.stack([log_P2s_list[i][i] for i in range(k)], axis=1)

    log_P = tf.reduce_logsumexp(log_p + log_P1s, axis=-1)

    # Bit hacky but if we have (allmost) all probability mass on a few
    # categories we have numerical problems since the probability for other classes
    # is basically zero
    # In this case we can just compute an exact gradient
    # Whereas we can just compute an exact gradient by setting
    # We choose this where the probability mass > 1 - 1e-5, so approx logprob > -1e-5
    is_exact = tf.reduce_logsumexp(log_p, axis=-1) > -1e-5

    log_R1 = log_P1s - log_P[..., None]
    log_R2 = log_P2s - log_P1s[..., None]

    log_R1 = tf.where(tf.broadcast_to(is_exact[:, None], log_R1.shape), tf.zeros_like(log_R1), log_R1)
    log_R2 = tf.where(tf.broadcast_to(is_exact[:, None, None], log_R2.shape), tf.zeros_like(log_R2), log_R2)

    #     log_R1[is_exact] = 0
    #     log_R2[is_exact] = 0

    tf.check_numerics(log_R1, "Nans in log_R1")
    tf.check_numerics(log_R2, "Nans in log_R2")

    return log_R1, log_R2
