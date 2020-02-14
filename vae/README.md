# Categorical Variational Auto-Encoder

Code for Categorical Variational Auto-Encoder adapted from [ARSM](https://github.com/ARM-gradient/ARSM).
Runs on Python 3.6 and TensorFlow 1.12.
It includes implementations of our unordered set estimator, as well as REINFORCE without replacement (using a built-in baseline) and a number of other baselines.

* See [all_experiments.sh](all_experiments.sh) for the commands that were used to run experiments.
* Run `jupyter notebook` from this (`/vae`) directory and use [plot_vae_results.ipynb](plot_vae_results.ipynb) to plot the results.

### Acknowledgements
Thanks to the original implementation of the Categorical VAE in [https://github.com/ARM-gradient/ARSM](https://github.com/ARM-gradient/ARSM).