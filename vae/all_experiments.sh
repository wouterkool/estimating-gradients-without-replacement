##### Run variances experiments
# Thresholded at 0.5 MNIST dataset
python vae.py --estimator rfwr --n_samples 4 --experiment_name directgradlogvar --lr 1e-3 --log_variance --add_direct_gradient --num_dimensions 20
python vae.py --estimator rfwr --n_samples 4 --experiment_name directgradlogvar --lr 1e-3 --log_variance --add_direct_gradient --num_dimensions 2

# Standard binarized MNIST dataset
python vae.py --estimator rfwr --n_samples 4 --experiment_name directgradlogvar --lr 1e-3 --log_variance --add_direct_gradient --num_dimensions 20 --larochelle
python vae.py --estimator rfwr --n_samples 4 --experiment_name directgradlogvar --lr 1e-3 --log_variance --add_direct_gradient --num_dimensions 2 --larochelle


##### First runs

# Thresholded at 0.5 MNIST dataset, large domain (20 dimensions)

python vae.py --estimator stgs --n_samples 1 --experiment_name directgrad --lr 1e-4 --add_direct_gradient --num_dimensions 20
python vae.py --estimator relax --n_samples 1 --experiment_name directgrad --lr 1e-4 --add_direct_gradient --num_dimensions 20
python vae.py --estimator reinforce_bl --n_samples 1 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 20
python vae.py --estimator arsm --n_samples 1 --experiment_name directgrad --lr 1e-4 --add_direct_gradient --num_dimensions 20

python vae.py --estimator rfwr --n_samples 4 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 20
python vae.py --estimator rf_unord --n_samples 4 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 20
python vae.py --estimator sasbl --n_samples 4 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 20

python vae.py --estimator rfwr --n_samples 8 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 20
python vae.py --estimator rf_unord --n_samples 8 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 20
python vae.py --estimator sasbl --n_samples 8 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 20

# Thresholded at 0.5 MNIST dataset, small domain (2 dimensions)

python vae.py --estimator stgs --n_samples 1 --experiment_name directgrad --lr 1e-4 --add_direct_gradient --num_dimensions 2
python vae.py --estimator relax --n_samples 1 --experiment_name directgrad --lr 1e-4 --add_direct_gradient --num_dimensions 2
python vae.py --estimator reinforce_bl --n_samples 1 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 2
python vae.py --estimator arsm --n_samples 1 --experiment_name directgrad --lr 1e-4 --add_direct_gradient --num_dimensions 2

python vae.py --estimator rfwr --n_samples 4 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 2
python vae.py --estimator rf_unord --n_samples 4 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 2
python vae.py --estimator sasbl --n_samples 4 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 2

python vae.py --estimator rfwr --n_samples 8 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 2
python vae.py --estimator rf_unord --n_samples 8 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 2
python vae.py --estimator sasbl --n_samples 8 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 2


# Standard binarized MNIST dataset, large domain (20 dimensions)

python vae.py --estimator stgs --n_samples 1 --experiment_name directgrad --lr 1e-4 --add_direct_gradient --num_dimensions 20 --larochelle
python vae.py --estimator relax --n_samples 1 --experiment_name directgrad --lr 1e-4 --add_direct_gradient --num_dimensions 20 --larochelle
python vae.py --estimator reinforce_bl --n_samples 1 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 20 --larochelle
python vae.py --estimator arsm --n_samples 1 --experiment_name directgrad --lr 1e-4 --add_direct_gradient --num_dimensions 20 --larochelle

python vae.py --estimator rfwr --n_samples 4 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 20 --larochelle
python vae.py --estimator rf_unord --n_samples 4 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 20 --larochelle
python vae.py --estimator sasbl --n_samples 4 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 20 --larochelle

python vae.py --estimator rfwr --n_samples 8 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 20 --larochelle
python vae.py --estimator rf_unord --n_samples 8 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 20 --larochelle
python vae.py --estimator sasbl --n_samples 8 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 20 --larochelle

# Standard binarized MNIST dataset, small domain (2 dimensions)

python vae.py --estimator stgs --n_samples 1 --experiment_name directgrad --lr 1e-4 --add_direct_gradient --num_dimensions 2 --larochelle
python vae.py --estimator relax --n_samples 1 --experiment_name directgrad --lr 1e-4 --add_direct_gradient --num_dimensions 2 --larochelle
python vae.py --estimator reinforce_bl --n_samples 1 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 2 --larochelle
python vae.py --estimator arsm --n_samples 1 --experiment_name directgrad --lr 1e-4 --add_direct_gradient --num_dimensions 2 --larochelle

python vae.py --estimator rfwr --n_samples 4 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 2 --larochelle
python vae.py --estimator rf_unord --n_samples 4 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 2 --larochelle
python vae.py --estimator sasbl --n_samples 4 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 2 --larochelle

python vae.py --estimator rfwr --n_samples 8 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 2 --larochelle
python vae.py --estimator rf_unord --n_samples 8 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 2 --larochelle
python vae.py --estimator sasbl --n_samples 8 --experiment_name directgrad --lr 1e-3 --add_direct_gradient --num_dimensions 2 --larochelle



##### Second runs (same except for experiment names)

# Thresholded at 0.5 MNIST dataset, large domain (20 dimensions)

python vae.py --estimator stgs --n_samples 1 --experiment_name directgrad2 --lr 1e-4 --add_direct_gradient --num_dimensions 20
python vae.py --estimator relax --n_samples 1 --experiment_name directgrad2 --lr 1e-4 --add_direct_gradient --num_dimensions 20
python vae.py --estimator reinforce_bl --n_samples 1 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 20
python vae.py --estimator arsm --n_samples 1 --experiment_name directgrad2 --lr 1e-4 --add_direct_gradient --num_dimensions 20

python vae.py --estimator rfwr --n_samples 4 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 20
python vae.py --estimator rf_unord --n_samples 4 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 20
python vae.py --estimator sasbl --n_samples 4 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 20

python vae.py --estimator rfwr --n_samples 8 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 20
python vae.py --estimator rf_unord --n_samples 8 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 20
python vae.py --estimator sasbl --n_samples 8 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 20

# Thresholded at 0.5 MNIST dataset, small domain (2 dimensions)

python vae.py --estimator stgs --n_samples 1 --experiment_name directgrad2 --lr 1e-4 --add_direct_gradient --num_dimensions 2
python vae.py --estimator relax --n_samples 1 --experiment_name directgrad2 --lr 1e-4 --add_direct_gradient --num_dimensions 2
python vae.py --estimator reinforce_bl --n_samples 1 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 2
python vae.py --estimator arsm --n_samples 1 --experiment_name directgrad2 --lr 1e-4 --add_direct_gradient --num_dimensions 2

python vae.py --estimator rfwr --n_samples 4 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 2
python vae.py --estimator rf_unord --n_samples 4 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 2
python vae.py --estimator sasbl --n_samples 4 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 2

python vae.py --estimator rfwr --n_samples 8 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 2
python vae.py --estimator rf_unord --n_samples 8 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 2
python vae.py --estimator sasbl --n_samples 8 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 2


# Standard binarized MNIST dataset, large domain (20 dimensions)

python vae.py --estimator stgs --n_samples 1 --experiment_name directgrad2 --lr 1e-4 --add_direct_gradient --num_dimensions 20 --larochelle
python vae.py --estimator relax --n_samples 1 --experiment_name directgrad2 --lr 1e-4 --add_direct_gradient --num_dimensions 20 --larochelle
python vae.py --estimator reinforce_bl --n_samples 1 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 20 --larochelle
python vae.py --estimator arsm --n_samples 1 --experiment_name directgrad2 --lr 1e-4 --add_direct_gradient --num_dimensions 20 --larochelle

python vae.py --estimator rfwr --n_samples 4 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 20 --larochelle
python vae.py --estimator rf_unord --n_samples 4 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 20 --larochelle
python vae.py --estimator sasbl --n_samples 4 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 20 --larochelle

python vae.py --estimator rfwr --n_samples 8 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 20 --larochelle
python vae.py --estimator rf_unord --n_samples 8 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 20 --larochelle
python vae.py --estimator sasbl --n_samples 8 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 20 --larochelle

# Standard binarized MNIST dataset, small domain (2 dimensions)

python vae.py --estimator stgs --n_samples 1 --experiment_name directgrad2 --lr 1e-4 --add_direct_gradient --num_dimensions 2 --larochelle
python vae.py --estimator relax --n_samples 1 --experiment_name directgrad2 --lr 1e-4 --add_direct_gradient --num_dimensions 2 --larochelle
python vae.py --estimator reinforce_bl --n_samples 1 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 2 --larochelle
python vae.py --estimator arsm --n_samples 1 --experiment_name directgrad2 --lr 1e-4 --add_direct_gradient --num_dimensions 2 --larochelle

python vae.py --estimator rfwr --n_samples 4 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 2 --larochelle
python vae.py --estimator rf_unord --n_samples 4 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 2 --larochelle
python vae.py --estimator sasbl --n_samples 4 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 2 --larochelle

python vae.py --estimator rfwr --n_samples 8 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 2 --larochelle
python vae.py --estimator rf_unord --n_samples 8 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 2 --larochelle
python vae.py --estimator sasbl --n_samples 8 --experiment_name directgrad2 --lr 1e-3 --add_direct_gradient --num_dimensions 2 --larochelle



