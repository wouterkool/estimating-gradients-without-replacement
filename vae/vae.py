import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
import pickle
import argparse
from tqdm import tqdm
from beam_search import beam_search
from gumbel import compute_log_R_O_nfac, log1mexp
import urllib

slim = tf.contrib.slim
Bernoulli = tf.contrib.distributions.Bernoulli
OneHotCategorical = tf.contrib.distributions.OneHotCategorical
RelaxedOneHotCategorical = tf.contrib.distributions.RelaxedOneHotCategorical
Categorical = tf.contrib.distributions.Categorical
Dirichlet = tf.contrib.distributions.Dirichlet

parser = argparse.ArgumentParser(description='VAE')

parser.add_argument('--estimator', type=str, required=True, help='rfwr, rfunord, gs, stgs')
parser.add_argument('--no_binarization', action='store_true', help='Set flag to NOT binarize input')
parser.add_argument('--n_samples', type=int, default=1, help='Number of samples per datapoint')
parser.add_argument('--experiment_name', type=str, default='')
parser.add_argument('--eager', action='store_true', help='Use eager execution (for debugging)')
parser.add_argument('--pb', action='store_true', help='Display progress bar')
parser.add_argument('--single_optimizer', action='store_true')
parser.add_argument('--log_variance', action='store_true')
parser.add_argument('--add_direct_gradient', action='store_true')
parser.add_argument('--num_categories', type=int, default=10)
parser.add_argument('--num_dimensions', type=int, default=20)
parser.add_argument('--larochelle', action='store_true', help='Use Hugo Larochelle\'s binary mnist data ')
parser.add_argument('--lr', type=float, default=1e-4)

args = parser.parse_args()

args.experiment_name = "{}{}{}_lr{}{}{}".format(
    args.experiment_name + "_" if args.experiment_name != "" else "",
    args.estimator, args.n_samples, args.lr,
    "nobin" if args.no_binarization else "", "dg" if args.add_direct_gradient else "")

if args.estimator in ('reinforce', 'reinforce_bl', 'gs', 'stgs', 'relax', 'arsm') and not args.log_variance:
    assert args.n_samples == 1

n_samp = args.n_samples

if args.eager:
    tf.enable_eager_execution()

K = args.num_categories
N = args.num_dimensions

# %%
directory = os.getcwd() + f'/discrete_out_vae_new/{K}^{N}/'
if args.larochelle:
    directory = directory + 'larochelle/'
if not os.path.exists(directory):
    os.makedirs(directory)
batch_size = 200

training_epochs = 1000

tau0=1.0 # initial temperature

b_dim = N * K
x_dim = 784

learn_temp = True


# Code from https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vq_vae.py
BERNOULLI_PATH = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
FILE_TEMPLATE = "binarized_mnist_{split}.amat"
def download(directory, filename):
  """Downloads a file."""
  filepath = os.path.join(directory, filename)
  if os.path.isfile(filepath):
    return filepath
  if not os.path.exists(directory):
    os.makedirs(directory)
  url = os.path.join(BERNOULLI_PATH, filename)
  print("Downloading %s to %s" % (url, filepath))
  urllib.request.urlretrieve(url, filepath)
  return filepath

# based on https://github.com/yburda/iwae/blob/master/datasets.py
FILE_TEMPLATE_NP = "binarized_mnist_{split}.npy"
def load_mnist_binary_dataset(directory, split):
    from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
    np_filepath = os.path.join(directory, FILE_TEMPLATE_NP.format(split=split))
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    if os.path.isfile(np_filepath):
        np_data = np.load(np_filepath)
    else:
        with open(download(directory, FILE_TEMPLATE.format(split=split))) as f:
            lines = f.readlines()
        np_data = lines_to_np_array(lines).astype('float32')
        np.save(np_filepath, np_data)
    return DataSet(np_data.reshape([-1, 28, 28, 1]) * 255, np.zeros(len(np_data)))

def load_larochelle(directory):
    from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
    return Datasets(
        train=load_mnist_binary_dataset(directory, 'train'),
        validation=load_mnist_binary_dataset(directory, 'valid'),
        test=load_mnist_binary_dataset(directory, 'test')
    )

def lrelu(x, alpha=0.1):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def encoder(x, b_dim, reuse=False):
    with tf.variable_scope("encoder", reuse=reuse):
        h2 = slim.stack(x, slim.fully_connected, [512,256], activation_fn=lrelu)
        log_alpha = tf.layers.dense(h2, b_dim, activation=None)
    return log_alpha


def decoder(b, x_dim, reuse=False):
    # return logits
    with tf.variable_scope("decoder", reuse=reuse):
        h2 = slim.stack(b, slim.fully_connected, [256,512], activation_fn=lrelu)
        log_alpha = tf.layers.dense(h2, x_dim, activation=None)
    return log_alpha


def kl_cat(q_logit, p_logit):
    '''
    input: N*n_cv*n_class
    '''
    eps = 1e-5
    q = tf.nn.softmax(q_logit, dim=2)
    p = tf.nn.softmax(p_logit, dim=2)
    return tf.reduce_sum(q * (tf.log(q + eps) - tf.log(p + eps)), axis=[1, 2])


def bernoulli_loglikelihood(b, log_alpha):
    return b * (-tf.nn.softplus(-log_alpha)) + (1 - b) * (-log_alpha - tf.nn.softplus(-log_alpha))


def fun(x_star, E, logits_y, reuse_decoder=True):
    '''
    x_star is N*d_x, E is N* (n_cv*n_class), z_concate is N*n_cv*n_class
    prior_logit0 is n_cv*n_class
    calculate log p(x_star|E) + log p(E) - log q(E|x_star)
    x_star is observe x; E is latent b
    return (N,)
    '''

    logits_py = tf.ones_like(logits_y) * 1. / K  # uniform
    # (bs)
    KL = kl_cat(logits_y, logits_py)

    # log p(x_star|E)
    logit_x = decoder(E, x_dim, reuse=reuse_decoder)
    log_p_x_given_b = bernoulli_loglikelihood(x_star, logit_x)
    # (N,)
    log_p_x_given_b = tf.reduce_sum(log_p_x_given_b, axis=-1)

    neg_elbo = - log_p_x_given_b + KL

    return neg_elbo

def fun2(x_star, E, logits_y, reuse_decoder=True):
    logits_x = decoder(E, x_dim, reuse=reuse_decoder)

    p_x = Bernoulli(logits=logits_x)

    recons = tf.reduce_sum(p_x.log_prob(x_star), 1)
    logits_py = tf.ones_like(logits_y) * 1. / K  # uniform

    p_cat_y = OneHotCategorical(logits=logits_py)
    q_cat_y = OneHotCategorical(logits=logits_y)
    KL_qp = tf.distributions.kl_divergence(q_cat_y, p_cat_y)

    KL = tf.reduce_sum(KL_qp, 1)

    neg_elbo = KL - recons

    return neg_elbo

tf.reset_default_graph()

eps = 1e-6
if args.estimator == 'relax':
    eps = 1e-8  # For some reason we get nans otherwise

lr = tf.constant(0.0001)

if tf.executing_eagerly():
    mnist = input_data.read_data_sets(os.getcwd() + '/MNIST', one_hot=True)
    debug_data = mnist.train
    x0, _ = debug_data.next_batch(7)
else:
    x0 = tf.placeholder(tf.float32, shape=(batch_size, 784), name='x')

x = x0 if args.no_binarization else tf.to_float(x0 > .5)

logits_y_ = encoder(x, b_dim)
logits_y = tf.reshape(logits_y_,[-1,N,K])
log_p_y = tf.nn.log_softmax(logits_y, -1)

q_y = Categorical(logits=logits_y)

y_sample_ = q_y.sample() #N*n_cv
y_sample = tf.cast(tf.one_hot(y_sample_,depth=K) ,tf.float32)

y_flat = slim.flatten(y_sample)


eval_neg_elbo = fun(x, y_flat, logits_y, reuse_decoder=False)

eval_costs = tf.reduce_mean(eval_neg_elbo)

eval_avgprob = tf.reduce_mean(tf.exp(tf.reduce_sum(q_y.log_prob(y_sample_), -1)))

flat_grads = {}
if args.estimator in ('gs', 'stgs'):
    # Gumbel-Softmax (and straight-through Gumbel-Softmax)
    tau = tf.Variable(tau0, name="temperature", trainable=learn_temp)

    q_y = RelaxedOneHotCategorical(tau, logits_y)
    y = q_y.sample()
    if args.estimator == 'stgs':
        y_hard = tf.cast(tf.one_hot(tf.argmax(y, -1), K), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    net = slim.flatten(y)

    neg_elbo = fun(x, net, logits_y)

    train_costs = tf.reduce_mean(neg_elbo)
    loss = train_costs

    gs_grad = tf.gradients(loss, logits_y)

    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

if args.estimator not in ('gs', 'stgs', 'arsm') or args.log_variance:
    losses = {}
    if args.estimator in ('reinforce', 'reinforce_bl') or args.log_variance:
        # loss = tf.reduce_mean(neg_elbo)
        train_costs = eval_costs
        gen_loss = train_costs

        ###############################

        theta = tf.nn.softmax(logits_y, dim=-1) #bs*N*K

        logq = tf.reduce_sum(tf.reduce_sum(y_sample*log_p_y,-1),-1)

        #bs
        if args.estimator == 'reinforce' or args.log_variance:
            F = eval_neg_elbo # fun(x,y_flat,logits_y,reuse_decoder=True) #to minimize
            inf_loss = tf.reduce_mean(tf.stop_gradient(F)*logq)

            losses['reinforce'] = (gen_loss, inf_loss)

        if args.estimator == 'reinforce_bl' or args.log_variance:
            y_sample_bl_ = q_y.sample()  # N*n_cv
            y_sample_bl = tf.cast(tf.one_hot(y_sample_bl_, depth=K), tf.float32)

            y_flat_bl = slim.flatten(y_sample_bl)

            bl = fun(x, y_flat_bl, logits_y, reuse_decoder=True)

            inf_loss = tf.reduce_mean(tf.stop_gradient(eval_neg_elbo - bl) * logq)

            losses['reinforce_bl'] = (gen_loss, inf_loss)

    if args.estimator == 'relax' or args.log_variance:
        train_costs = eval_costs
        gen_loss = train_costs

        ###############################
        # given Categorical logits_y, use RELAX to compute gradient to the logits

        u1 = tf.random_uniform(shape=[N, K])
        u1 = u1[None, :]
        u2 = tf.random_uniform(shape=[N, K])
        u2 = u2[None, :]
        theta = tf.nn.softmax(logits_y, dim=-1)  # bs*N*K
        z = tf.log(theta + eps) - tf.log(-tf.log(u1 + eps) + eps)
        b = tf.argmax(z, axis=-1)

        b_onehot0 = tf.one_hot(b, depth=K)
        b_onehot = b_onehot0 * (-1)
        b_onehot = b_onehot + 1  # make everywhere 1 except for i = b is 0
        b_flat = tf.cast(slim.flatten(b_onehot0), tf.float32)

        tmp = -b_onehot * tf.log(u2 + eps) / (theta + eps)
        z_tilde = -tf.log(tmp - tf.log(u2 + eps))
        # bs
        logp = tf.reduce_sum(tf.log(tf.reduce_sum(b_onehot0 * theta, -1) + eps), -1)

        z_flat = slim.flatten(z)
        z_tilde_flat = slim.flatten(z_tilde)
        # bs
        F = fun(x, b_flat, logits_y, reuse_decoder=True)  # to minimize

        def cv(b, reuse=False):
            # return control_variates
            with tf.variable_scope("control_var", reuse=reuse):
                h2 = slim.stack(b, slim.fully_connected, [200, 200])
                # h2 = slim.stack(b, slim.fully_connected, [512, 256])
                out = tf.layers.dense(h2, 1, activation=None)
            return out

        cv_z = tf.squeeze(cv(z_flat))
        cv_z_tilde = tf.squeeze(cv(z_tilde_flat, reuse=True))

        inf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')

        inf_loss = tf.reduce_mean(tf.stop_gradient(F - cv_z_tilde) * logp - cv_z_tilde + cv_z)
        # alpha_grads = tf.gradients(relax_loss, logits_y_)
        # inf_grads = tf.gradients(logits_y_, inf_vars, grad_ys=alpha_grads)
        # should be the same
        inf_grads = tf.gradients(inf_loss, inf_vars)  # Bit inefficient since gradient will be computed later again

        ###############################
        vectorized_grads = tf.concat(
            [tf.reshape(g, [-1]) for g in inf_grads if g is not None], axis=0)
        # vectorized_grads = tf.reshape(alpha_grads, [-1])

        relax_cv_loss = tf.reduce_mean(tf.square(vectorized_grads))
        relax_cv_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='control_var')
        relax_cv_train_op = tf.train.AdamOptimizer(lr).minimize(relax_cv_loss, var_list=relax_cv_vars)

        losses['relax'] = (gen_loss, inf_loss)

    if args.estimator == 'rfwr' or args.log_variance:

        # Repeat for number of samples
        logits_y_tile = tf.tile(logits_y, [n_samp, 1, 1])

        q_y = Categorical(logits=logits_y_tile)

        y_sample_tile = q_y.sample()  # N*n_cv
        y_sample_tile = tf.one_hot(y_sample_tile, depth=K)
        y_sample_tile = tf.cast(y_sample_tile, tf.float32)

        y_flat_tile = slim.flatten(y_sample_tile)

        theta = tf.nn.softmax(logits_y_tile, dim=-1)  # bs*N*K

        avgprob = tf.reduce_mean(tf.reduce_sum(y_sample_tile * theta, -1) + eps)

        neg_elbo = fun(tf.tile(x, [n_samp, 1]), y_flat_tile, logits_y_tile)
        train_costs = tf.reduce_mean(neg_elbo)
        gen_loss = train_costs

        # Note, first dimension is the repeated samples, second dimension is batch now
        neg_elbo_samples = tf.reshape(neg_elbo, [n_samp, -1])
        adv = n_samp / (n_samp - 1) * (neg_elbo_samples - tf.reduce_mean(neg_elbo_samples, 0, keepdims=True))

        logq = tf.reduce_sum(tf.log(tf.reduce_sum(y_sample_tile * theta, -1) + eps), -1)

        logq = tf.reshape(logq, [n_samp, -1])

        inf_loss = tf.reduce_mean(tf.stop_gradient(adv) * logq)

        losses['rfwr'] = (gen_loss, inf_loss)

    if args.estimator in ('rf_unord', 'sas', 'sasbl') or args.log_variance:
        sample, log_p_sample, _ = beam_search(log_p_y, n_samp, stochastic=True)

        sample_flat = tf.reshape(tf.transpose(tf.one_hot(sample, depth=K), perm=[1, 0, 2, 3]), [-1, N * K])

        neg_elbo_all = fun(tf.tile(x, [n_samp, 1]), sample_flat, tf.tile(log_p_y, [n_samp, 1, 1]))
        neg_elbo_samples = tf.transpose(tf.reshape(neg_elbo_all, [n_samp, -1]))

        if args.estimator == 'rf_unord' or args.log_variance:
            log_R1, log_R2 = compute_log_R_O_nfac(log_p_sample)
            neg_elbo = tf.reduce_sum(tf.stop_gradient(tf.exp(log_R1 + log_p_sample)) * neg_elbo_samples, axis=-1)
            train_costs = tf.reduce_mean(neg_elbo)
            gen_loss = train_costs
            # For sampling without replacement, we cannot simply take mean!!

            bl_vals = tf.reduce_sum(tf.exp(log_p_sample[:, None, :] + log_R2) * neg_elbo_samples[:, None, :], axis=-1)
            adv = neg_elbo_samples - bl_vals

            batch_rf_losses = tf.reduce_sum(tf.exp(tf.stop_gradient(log_R1) + log_p_sample) * tf.stop_gradient(adv), -1)

            inf_loss = tf.reduce_mean(batch_rf_losses)
            # avgprob = tf.reduce_mean(tf.reduce_sum(tf.exp(log_p_sample), -1))

            losses['rf_unord'] = (gen_loss, inf_loss)

        if args.estimator in ('sas', 'sasbl') or args.log_variance:

            # note, sum_{i<k} p_i f_i + (1 - sum_{i<k} p_i) f_k is the same as
            # sum+{i<=k} p_i f_i + (1 - sum_{i<=k} p_i) f_k
            sample_weight = tf.stop_gradient(tf.exp(log1mexp(tf.reduce_logsumexp(log_p_sample, -1))))
            gen_sample_term = (sample_weight * neg_elbo_samples[:, -1])
            gen_sum_term = tf.reduce_sum(tf.stop_gradient(tf.exp(log_p_sample)) * neg_elbo_samples, -1)
            gen_loss = tf.reduce_mean(gen_sum_term + gen_sample_term)

            # Inference loss
            if args.estimator == 'sas' or args.log_variance:
                adv = tf.stop_gradient(neg_elbo_samples)
                inf_sample_term = (sample_weight * log_p_sample[:, -1] * tf.stop_gradient(adv[:, -1]))
                # p_i * grad log_p_i = grad p_i, so we don't stop the gradient for p_i here
                inf_sum_term = tf.reduce_sum(tf.exp(log_p_sample) * tf.stop_gradient(adv), -1)
                inf_loss = tf.reduce_mean(inf_sum_term + inf_sample_term)

                losses['sas'] = (gen_loss, inf_loss)

            if args.estimator == 'sasbl' or args.log_variance:
                # Use the single evaluation sample as baseline
                adv = tf.stop_gradient(neg_elbo_samples - eval_neg_elbo[:, None])
                inf_sample_term = (sample_weight * log_p_sample[:, -1] * tf.stop_gradient(adv[:, -1]))
                # p_i * grad log_p_i = grad p_i, so we don't stop the gradient for p_i here
                inf_sum_term = tf.reduce_sum(tf.exp(log_p_sample) * tf.stop_gradient(adv), -1)
                inf_loss = tf.reduce_mean(inf_sum_term + inf_sample_term)

                losses['sasbl'] = (gen_loss, inf_loss)

    if not args.single_optimizer:
        gen_opt = tf.train.AdamOptimizer(lr)
        gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
        inf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')

        if args.estimator in losses:
            (gen_loss, inf_loss) = losses[args.estimator]
            gen_gradvars = gen_opt.compute_gradients(gen_loss, var_list=gen_vars)

            inf_grads = tf.gradients(inf_loss, inf_vars)
            inf_opt = tf.train.AdamOptimizer(lr)
            inf_gradvars = zip(inf_grads, inf_vars)

            # We should only update the gradients after having computed all of them
            gen_grads, _ = zip(*gen_gradvars)

            if args.add_direct_gradient:  # Add missing gradient part
                dgen_dinf = tf.gradients(gen_loss, inf_vars)
                inf_grads = [didi + dgdi for didi, dgdi in zip(inf_grads, dgen_dinf)]
                inf_gradvars = zip(inf_grads, inf_vars)
            with tf.control_dependencies(list(gen_grads) + inf_grads):
                gen_train_op = gen_opt.apply_gradients(gen_gradvars)
                inf_train_op = inf_opt.apply_gradients(inf_gradvars)

            ###############################
            deps = [gen_train_op, inf_train_op]
            if args.estimator == 'relax':
                deps.append(relax_cv_train_op)
            with tf.control_dependencies(deps):
                train_op = tf.no_op()

        if args.log_variance:
            for name, (gen_loss, inf_loss) in losses.items():
                # gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
                gen_gradvars = gen_opt.compute_gradients(gen_loss, var_list=gen_vars)
                gen_grads, _ = zip(*gen_gradvars)
                # inf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
                inf_grads = tf.gradients(inf_loss, inf_vars)

                if args.add_direct_gradient:  # Add missing gradient part
                    dgen_dinf = tf.gradients(gen_loss, inf_vars)
                    inf_grads = [didi + dgdi
                                 if didi is not None or dgdi is not None
                                 else None
                                 for didi, dgdi in zip(inf_grads, dgen_dinf)]

                # inf_gradvars = zip(inf_grads, inf_vars)
                # all_gradvars = gen_gradvars + inf_gradvars
                flat_grad = tf.concat([tf.reshape(grad, [-1]) for grad in list(gen_grads) + inf_grads if grad is not None], 0)
                flat_grads[name] = flat_grad
    else:

        if args.estimator in losses:
            (gen_loss, inf_loss) = losses[args.estimator]
            # Alternative, does this work too? Yes, and it will include the direct gradient from the KL in the
            # inference loss which depends on the log probabilities of the latents which depend on the encoder
            loss = gen_loss + inf_loss
            train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        if args.log_variance:
            flat_grads = {}
            for name, (gen_loss, inf_loss) in losses.items():
                grads = tf.gradients(gen_loss + inf_loss, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
                flat_grad = tf.concat([tf.reshape(grad, [-1]) for grad in grads if grad is not None])
                flat_grads[name] = flat_grad

if args.estimator == 'arsm' or args.log_variance:
    train_costs = eval_costs
    gen_loss = train_costs

    gen_opt = tf.train.AdamOptimizer(lr)
    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
    gen_gradvars = gen_opt.compute_gradients(gen_loss, var_list=gen_vars)

    b_size = tf.shape(x)[0]

    # provide encoder q(b|x) gradient by data augmentation
    Dir = Dirichlet([1.0] * K)
    pai = Dir.sample(sample_shape=[b_size, N])

    x_star_u = x  # N*d_x

    EE = tf.placeholder(tf.float32, [None, N, K])
    EE_flat = slim.flatten(EE)
    # F_ij = funold(x_star_u,EE_flat,prior_logit0,z_concate,reuse_decoder= True)
    F_ij = fun(x_star_u, EE_flat, logits_y, reuse_decoder=True)

    F = tf.placeholder(tf.float32, [None, K, K])  # n_class*n_class
    F0 = F - tf.reduce_mean(F, axis=2, keep_dims=True)
    F1 = tf.expand_dims(F0, axis=1)
    PAI = tf.placeholder(tf.float32, [None, N, K])
    pai1 = 1 / K - tf.tile(tf.expand_dims(PAI, axis=2), [1, 1, K, 1])

    # alpha_grads0 = tf.reduce_mean(F1*pai1, axis = -1)
    # alpha_grads = tf.reshape(alpha_grads0[:,:,1:],[-1,z_dim])
    # alpha_grads = tf.reshape(alpha_grads,[-1,z_dim])

    alpha_grads0 = tf.reduce_mean(F1 * pai1, axis=-1)
    alpha_grads = tf.reshape(alpha_grads0, [-1, b_dim])

    inf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')

    inf_grads = tf.gradients(logits_y_, inf_vars, grad_ys=alpha_grads)  # /b_s
    if args.add_direct_gradient:  # Somehow this seems to make little difference for ARSM where it works for the other implementation
        dgen_dinf = tf.gradients(gen_loss, inf_vars)
        # inf_grads = [didi + dgdi for didi, dgdi in zip(inf_grads, dgen_dinf)]
        # Note: there are some None gradients because of adam optimizer auxiliary variables
        inf_grads = [didi + dgdi
                     if didi is not None or dgdi is not None
                     else None
                     for didi, dgdi in zip(inf_grads, dgen_dinf)]

    inf_gradvars = zip(inf_grads, inf_vars)
    inf_opt = tf.train.AdamOptimizer(lr)

    # We should only update the gradients after having computed all of them
    gen_grads, _ = zip(*gen_gradvars)

    if args.estimator == 'arsm':
        # Don't make the train op if we're not actually training arsm
        with tf.control_dependencies([g for g in list(gen_grads) + inf_grads if g is not None]):
            gen_train_op = gen_opt.apply_gradients(gen_gradvars)
            inf_train_op = inf_opt.apply_gradients(inf_gradvars)

        # prior_train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(gen_loss,var_list=[prior_logit0])

        # with tf.control_dependencies([gen_train_op, inf_train_op, prior_train_op]):
        #    train_op = tf.no_op()
        with tf.control_dependencies([gen_train_op, inf_train_op]):
            train_op = tf.no_op()

    if args.log_variance:
        # gen_gradvars = gen_opt.compute_gradients(gen_loss, var_list=gen_vars)
        gen_grads, _ = zip(*gen_gradvars)
        # inf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
        #inf_grads = tf.gradients(inf_loss, inf_vars)
        # inf_gradvars = zip(inf_grads, inf_vars)
        # all_gradvars = gen_gradvars + inf_gradvars
        flat_grad_arsm = tf.concat([tf.reshape(grad, [-1]) for grad in list(gen_grads) + inf_grads if grad is not None], 0)
        # flat_grads['arsm'] = flat_grad


    def compt_F(sess, train_xs, pai, z_concate):
        pp, ph = sess.run([pai, z_concate], {x0: train_xs})
        FF = np.zeros([batch_size, K, K])
        from tensorflow.keras.utils import to_categorical
        for i in range(K):
            for j in range(i, K):
                pp_ij = np.copy(pp)
                pp_ij[:, :, [i, j]] = pp_ij[:, :, [j, i]]
                s_ij = to_categorical(np.argmin(np.log(pp_ij + 1e-6) - ph, axis=2), num_classes=K)
                FF[:, i, j] = sess.run(F_ij, {x0: train_xs, EE: s_ij})
                FF[:, j, i] = FF[:, i, j]
        return FF, pp

init_op = tf.global_variables_initializer()

# %% TRAIN
# get data
if args.larochelle:
    mnist = load_larochelle('larochelle_data')
    train_data = mnist.train
    test_data = mnist.test
    valid_data = mnist.validation
else:
    mnist = input_data.read_data_sets(os.getcwd() + '/MNIST', one_hot=True)
    train_data = mnist.train
    test_data = mnist.test
    valid_data = mnist.validation

total_points = train_data.num_examples
total_batch = int(total_points / batch_size)
# total_test_batch = int(test_data.num_examples / batch_size)
total_valid_batch = int(valid_data.num_examples / batch_size)

display_step = total_batch


# %%
def get_valid_cost(sess, data, total_batch):
    cost_eval = []
    avgprob_eval = []
    for j in range(total_batch):
        xs, _ = data.next_batch(batch_size)
        ev_cost, ev_ap = sess.run((eval_costs, eval_avgprob), {x0: xs})
        cost_eval.append(ev_cost)
        avgprob_eval.append(ev_ap)
    return np.mean(cost_eval), np.mean(avgprob_eval)

def get_gradient_logvars(sess, data, total_batch, use_batch=1, samples=1000, add_arsm=True):
    # all_grads_repetitions = []
    for i in range(samples):
        flat_grad_values = None
        for j in range(total_batch):  # Iterate through complete dataset so it resets
            if j < use_batch:
                # Only use first use_batch batches
                xs, _ = data.next_batch(batch_size)
                flat_grad_values_i_list = sess.run(list(flat_grads.values()), {x0: xs})

                if add_arsm:
                    FF, pp = compt_F(sess, xs, pai, logits_y)
                    #feed_dict = {**feed_dict, F: FF, PAI: pp}
                    # Get ARSM gradient
                    flat_grad_arsm_values_i = sess.run(flat_grad_arsm, {x0: xs, F: FF, PAI: pp})
                    flat_grad_values_i_list.append(flat_grad_arsm_values_i)

                flat_grad_values_i = np.stack(flat_grad_values_i_list, 0)
                if flat_grad_values is None:
                    flat_grad_values = flat_grad_values_i
                else:
                    flat_grad_values = flat_grad_values + flat_grad_values_i  # Accumulate to have larger minibatch size
        # all_grads_repetitions.append(flat_grad_values)

        if i == 0:
            all_grads_M = flat_grad_values
            all_grads_S = np.zeros_like(flat_grad_values)
        else:
            # Incremental computation of variance, see https://www.johndcook.com/blog/standard_deviation/
            diff = flat_grad_values - all_grads_M
            all_grads_M += diff / (i + 1)
            all_grads_S += (flat_grad_values - all_grads_M) * diff  # Note this is different than diff*2 since update

    # Take variance along repetitions, then sum to get trace of the covariance matrix, this gives 1 value
    # all_grads_rep_np = np.stack(all_grads_repetitions, 0)
    # all_grad_logvars = np.log(all_grads_rep_np.var(0).sum(-1))
    # all_grad_lognorm_sq = np.log((all_grads_rep_np.mean(0) ** 2).sum(-1))

    all_grad_logvars = np.log((all_grads_S / (samples - 1)).sum(-1))
    all_grad_lognorm_sq = np.log((all_grads_M ** 2).sum(-1))
    return all_grad_logvars, all_grad_lognorm_sq

if __name__ == "__main__":

    print('Training starts....', args.experiment_name)

    sess = tf.InteractiveSession()
    sess.run(init_op)
    record = [];
    step = 0

    import time

    start = time.time()
    COUNT = [];
    COST = [];
    TIME = [];
    COST_TEST = [];
    COST_VALID = [];
    epoch_list = [];
    time_list = []
    evidence_r = []
    avgprob_list = []
    avgprob_valid_list = []
    train_gradient_logvar_list = []
    train_gradient_lognorm_sq_list = []
    valid_gradient_logvar_list = []
    valid_gradient_lognorm_sq_list = []
    epoch_it = range(training_epochs)
    if args.pb:
        epoch_it = tqdm(epoch_it)
    for epoch in epoch_it:

        if epoch % 100 == 0 and args.log_variance:
            for i in range(1):
                for logvar_filename, gradient_logvar_list, gradient_lognorm_sq_list, dat, tot_batch in (
                        ("train_logvars_", train_gradient_logvar_list, train_gradient_lognorm_sq_list, train_data, total_batch),
                        ("valid_logvars_", valid_gradient_logvar_list, valid_gradient_lognorm_sq_list, valid_data, total_valid_batch)
                ):
                    print(f"Computing {logvar_filename}...")
                    # Also do before starting, compute variances of all estimators
                    gradient_logvars, gradient_lognorms_sq = get_gradient_logvars(sess, train_data, total_batch, samples=1000)
                    print(dict(zip([*flat_grads.keys(), 'arsm'], zip(gradient_logvars, gradient_lognorms_sq))))
                    gradient_logvar_list.append(gradient_logvars)
                    gradient_lognorm_sq_list.append(gradient_lognorms_sq)
                    grad_logvar_dict = dict(zip([*flat_grads.keys(), 'arsm'], zip(np.column_stack(gradient_logvar_list),
                                                                       np.column_stack(gradient_lognorm_sq_list))))
                    # grad_logvar_dict['arsm'] = get_arsm_gradient_logvar(sess, train_data, total_batch)
                    with open(directory + logvar_filename + args.experiment_name, 'wb') as f:
                        pickle.dump(grad_logvar_dict, f)
                    print("Done")

        record = []
        record_avgprob = []

        for i in range(total_batch):
            train_xs, _ = train_data.next_batch(batch_size)
            feed_dict = {x0: train_xs, lr: args.lr}
            if args.estimator == 'arsm':
                FF, pp = compt_F(sess, train_xs, pai, logits_y)
                feed_dict = {**feed_dict, F: FF, PAI: pp}

            # We run the eval_costs here to compute the costs, since this is the same for all estimators so there can
            # not be differences as a result of implementation
            _, cost, tr_ap = sess.run([train_op, eval_costs, eval_avgprob], feed_dict)
            record.append(cost)
            record_avgprob.append(tr_ap)
            step += 1

        print(epoch, 'cost=', np.mean(record), 'with std=', np.std(record), 'avgprob=', np.mean(record_avgprob))

        if epoch % 1 == 0:
            COUNT.append(step);
            COST.append(np.mean(record));
            avgprob_list.append(np.mean(record_avgprob))
            TIME.append(time.time() - start)
            valid_cost, valid_avgprob = get_valid_cost(sess, valid_data, total_valid_batch)
            COST_VALID.append(valid_cost)
            avgprob_valid_list.append(valid_avgprob)
            print('cost valid', valid_cost, 'avgprob', valid_avgprob)
        if epoch % 5 == 0:
            epoch_list.append(epoch)
            time_list.append(time.time() - start)
            all_ = [COUNT, COST, TIME, COST_TEST, COST_VALID, epoch_list, time_list, evidence_r, avgprob_list]
            with open(directory + args.experiment_name, 'wb') as f:
                pickle.dump(all_, f)





    print(args.experiment_name)

