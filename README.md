# Estimating Gradients for Discrete Random Variables by Sampling without Replacement

This repository contains implementations of the *Unordered set estimator* as derived in our paper [Estimating Gradients for Discrete Random Variables by Sampling without Replacement](https://openreview.net/forum?id=rklEj2EFvB).

* The [Bernoulli gradient variance experiment](bernoulli) is in PyTorch
* The [Categorical VAE experiment](vae) is in TensorFlow

## Paper
For more details, please see our paper [Estimating Gradients for Discrete Random Variables by Sampling without Replacement](https://openreview.net/forum?id=rklEj2EFvB) which has been accepted at [ICLR 2020](https://iclr.cc/Conferences/2020). If this code is useful for your work, please cite our paper:

```
@inproceedings{
    Kool2020Estimating,
    title={Estimating Gradients for Discrete Random Variables by Sampling without Replacement},
    author={Wouter Kool and Herke van Hoof and Max Welling},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=rklEj2EFvB}
}
```

## Stochastic Beam Search
For sampling without replacement from a fully factorized distribution for the VAE experiment, this code contains an implementation of [Stochastic Beam Search](https://arxiv.org/abs/1903.06059) in TensorFlow in [vae/lib/beam_search.py](vae/lib/beam_search.py).