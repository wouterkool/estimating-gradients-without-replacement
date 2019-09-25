import torch
import torch.nn.functional as F
from torch.distributions import Gumbel
from itertools import permutations


def log1mexp(x):
    # Computes log(1-exp(-|x|))
    # See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    x = -x.abs()
    return torch.where(x > -0.693, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def gumbel_log_survival(x):
    """Computes log P(g > x) = log(1 - P(g < x)) = log(1 - exp(-exp(-x))) for a standard Gumbel"""
    y = torch.exp(-x)
    return torch.where(
        x >= 10,  # means that y < 1e-4 so O(y^6) <= 1e-24 so we can use series expansion
        -x - y / 2 + y ** 2 / 24 - y ** 4 / 2880,  # + O(y^6), https://www.wolframalpha.com/input/?i=log(1+-+exp(-y))
        log1mexp(y)  # Hope for the best
    )

def all_perms(S, device=None):
    return torch.tensor(list(permutations(S)), device=device)

def log_pl(log_p, dim=-1):
    # Sampling has been done without replacement, compute likelihood without replacement
    # https://math.stackexchange.com/questions/2729561/
    # probability-of-an-unordered-sample-under-weighted-sampling-without-replacement
    # Note that we compute the likelihood for the ordered sample
    a, _ = log_p.max(dim, keepdim=True)
    p = (log_p - a).exp()
    # P = p_1 / 1 * p_2 / (1 - p_1) * p_3 / (1 - p_1 - p_2) ...
    # log P = log p_1 - log(1) + log p_2 - log(1 - p_1) + ...
    #       = sum_i log p_i - sum_i log(1 - sum_j<i p_j)
    # Note that the first term is log_likelihood,
    # and note that sum_j<i p_j = (sum_j<=i p_j) - p_i = cumsum(p_i) - p_i
    # log_partition = partition.log().sum()
    return log_p.sum(dim) - log1mexp(a + (p.cumsum(dim) - p).log()).sum(dim)

def log_pl_rec(log_p, dim=-1):
    """Recursive function of Plackett Luce log probability has better numerical stability
    since 1 - sum_i p_i can get very close to 0, this version never computes sum p_i directly"""
    assert dim == -1
    if log_p.size(-1) == 1:
        return log_p[..., 0]
    return log_p[..., 0] + log_pl_rec(log_p[..., 1:] - log1mexp(log_p[..., 0:1]), dim=dim)

def log_pS_Onfac(log_p):
    return torch.logsumexp(log_pl(all_perms(log_p, device=log_p.device)), -1)

def log_pS_Onfac_rec(log_p):
    return torch.logsumexp(log_pl_rec(all_perms(log_p, device=log_p.device)), -1)

def compute_log_R(log_p, num_points=1000, a=5.):
    # Computes the (log) ratio P(S\{s}|S \subseteq D\{s}) / P(S),
    # where S is an unordered sample under the Plackett-Luce model
    # Additionally computes the (conditional) second order log ratios
    # P(S\{s,s'}|S \subseteq D\{s,s'}) / P(S\{s}|S \subseteq D\{s})
    # Multiplying (or adding in log space) the results gives
    # The unconditional second order log ratios
    # P(S\{s,s'}|S \subseteq D\{s,s'}) / P(S)

    # Constant for numeric stability
    a = log_p.new_tensor(a)

    # Integrals are computed by the trapezoidal rule,
    # which equates to approximating the integral by
    # dx * sum_{i=1}^N (f(i) + f(i-1)) / 2 = dx / 2 * (f(0) + f(N) + 2 * sum_{i = 1}^{N-1} f(i))
    # Since f(0) and f(N) in our integral will be zero, we can just compute
    # dx * sum_{i = 1}^{N-1} f(i)
    # See https://en.wikipedia.org/wiki/Trapezoidal_rule

    # Create range of integration points, (1 ... N-1)/N (bounds are 0 to 1)
    log_v = (torch.arange(1, num_points, out=log_p.new()) / num_points).log()

    # First dim, numerical integration (N - 1)
    # Second dim, batch dimension (B)
    # Third dim, i in S (|S|)
    _q = gumbel_log_survival(-((log_p + a)[None, :, :] + torch.log(-log_v)[:, None, None]))

    # Compute the integrands (N - 1 x B)
    q = _q.sum(-1) + (torch.expm1(a + log1mexp(torch.logsumexp(log_p, -1)))[None, :] * log_v[:, None])

    # Subtract one factor for element that is left out
    q_without_s = q[..., None] - _q

    # Don't subtract same element twice for diagonals
    skip_diag = 1 - torch.eye(log_p.size(-1), out=log_p.new())[None, None, :, :]
    q_without_ss = q_without_s[..., None] - _q[..., None, :] * skip_diag  # 2nd order

    # To compute the log probabilities, we should add constant a + phi_S, but these cancel out
    sum_S = torch.logsumexp(q, 0)  # e.g. log_P_S = a + phi_S + sum_S
    sum_S_s = torch.logsumexp(q_without_s, 0)
    sum_S_ss = torch.logsumexp(q_without_ss, 0)
    return sum_S_s - sum_S[..., None], sum_S_ss - sum_S_s[..., None]


def all_2nd_order_perms(S, device=None):
    k = S.size(-1)
    ap = all_perms(S, device=device)
    apf = ap[ap[:, 0] < ap[:, 1]].view(k * (k - 1) // 2, -1, k)
    return apf[:, 0, :2], apf[:, :, 2:]

SO_PERM_CACHE = {}
def compute_log_R_O_nfac(log_p, so_perms=None):
    """
    Computes all first and second order log ratio's by computing P(S)
    for all second order sets leaving two elements out of S
    where the individual P(S) are computed by naive enumeration of all permutations
    This is inefficient especially for large sample sizes but can be used
    to validate alternative implementations
    """

    k = log_p.size(-1)
    if k == 1:
        # If k = 1, second order is not defined, and first order
        # P(S\{s}) / P(S) = P{{}} / P({s}) = 1 / p_s
        # log (1 / p_s) = - log p_s
        return -log_p[...], None

    if so_perms is None:
        if k in SO_PERM_CACHE:
            so_perms = SO_PERM_CACHE[k]
        else:
            so_perms = all_2nd_order_perms(torch.arange(k, dtype=torch.long), device=log_p.device)
            SO_PERM_CACHE[k] = so_perms

        # perm_ids = all_perms(torch.arange(k - 2, dtype=torch.long), device=log_p.device)

    keys, rest = so_perms
    first, second = torch.unbind(keys, -1)

    norm1 = log1mexp(log_p[..., first])
    norm2 = norm1 + log1mexp(log_p[..., second] - norm1)

    # Second order leave out log_probabilities
    log_P2s = log_p.new_zeros(log_p.size(0), k, k)

    if k > 2:  # For k = 2, thre remainder set is empty with log probability zero
        # Index to get
        # (batch_size, num_second_orders, num_perms, rest=k-2)
        log_p_rest = log_p[..., rest] - norm2[..., None, None]

        # (batch_size, num_second_orders, num_perms)
        logprobs = log_pl_rec(log_p_rest, -1)

        # (batch_size, num_second_orders)
        log_P = logprobs.logsumexp(-1)


        log_P2s[:, first, second] = log_P
        log_P2s[:, second, first] = log_P

    # Compute first order log_P
    log_P1s = torch.zeros_like(log_p)
    for i in range(k):
        # P(S) = sum_{s in S} p(s) P^{D\s}(S\s)
        log_p_without_i = torch.cat((log_p[:, :i], log_p[:, i + 1:]), -1) - log1mexp(log_p[:, i, None])
        log_P2s_without_i = torch.cat((log_P2s[:, i, :i], log_P2s[:, i, i + 1:]), -1)
        log_P1s[:, i] = (log_p_without_i + log_P2s_without_i).logsumexp(-1)
        log_P2s[:, i, i] = log_P1s[:, i]

    log_P = (log_p + log_P1s).logsumexp(-1)

    # Bit hacky but if we have (allmost) all probability mass on a few
    # categories we have numerical problems since the probability for other classes
    # is basically zero
    # In this case we can just compute an exact gradient
    # Whereas we can just compute an exact gradient by setting
    # We choose this where the probability mass > 1 - 1e-5, so approx logprob > -1e-5
    is_exact = log_p.logsumexp(-1) > -1e-5
    
    log_R1 = log_P1s - log_P[..., None]
    log_R2 = log_P2s - log_P1s[..., None]

    log_R1[is_exact] = 0
    log_R2[is_exact] = 0

    assert not torch.isnan(log_R1).any()
    assert not torch.isnan(log_R2).any()

    return log_R1, log_R2


def gumbel_with_maximum(phi, T, dim=-1):
    """
    Samples a set of gumbels which are conditioned on having a maximum along a dimension
    phi.max(dim)[0] should be broadcastable with the desired maximum T
    """
    # Gumbel with location phi, use PyTorch distributions so you cannot get -inf or inf (which causes trouble)
    g_phi = Gumbel(phi, torch.ones_like(phi)).rsample()
    Z, argmax = g_phi.max(dim)
    g = _shift_gumbel_maximum(g_phi, T, dim, Z=Z)
    return g, argmax


def shift_gumbel_maximum(g_phi, T, dim=-1, Z=None):
    g = _shift_gumbel_maximum(g_phi, T, dim, Z)
    CHECK_VALIDITY = True
    if CHECK_VALIDITY:
        g_inv = _shift_gumbel_maximum(g, Z, dim)
        if not (((g_phi - g_inv) < 1e-3) | (g_phi == g_inv)).all():
            # Disabled, in some cases we simply loose accuracy since we store absolute gumbel values instead of
            # the difference to their maximum which is closer to 0 and thus more stable
            RAISE_INVALID = False
            if RAISE_INVALID:
                assert False
        # assert (((g_phi - g_inv) < 1e-3) | (g_phi == g_inv)).all()
    return g


def _shift_gumbel_maximum(g_phi, T, dim=-1, Z=None):
    if Z is None:
        Z, _ = g_phi.max(dim)
    u = T.unsqueeze(dim) - g_phi + torch.log1p(-torch.exp(g_phi - Z.unsqueeze(dim)))
    return T.unsqueeze(dim) - F.relu(u) - torch.log1p(torch.exp(-u.abs()))
