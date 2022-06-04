
ROOT_PATH="/home/asatk/Documents/code/cern/asatk-improved_CcGAN/Simulation"
SEED=20
NSIM=1
NITERS=1
N_GAUSSIANS=100
N_GAUSSIANS_PLOT=4
N_SAMP_PER_GAUSSIAN=150
STD_GAUSSIAN=0.0075
GEO="line"
RADIUS=1
YVAL=0.5
BATCH_SIZE_D=150
BATCH_SIZE_G=150
LR_GAN=5e-6
SIGMA=-1.0
KAPPA=-2.0
DIM_GAN=2

#lr btwn 10-6 10-3
#batch size as big as possible - classical optimization prob
#network model - overfitting potentially, reduce dimensions and layers
#'looks more like a variational - an autoencoder type problem'

echo "-------------------------------------------------------------------------------------------------"
echo "CcGAN Hard"
CUDA_VISIBLE_DEVICES=0 python3 main.py --root_path $ROOT_PATH --nsim $NSIM --seed $SEED --n_gaussians $N_GAUSSIANS --n_gaussians_plot $N_GAUSSIANS_PLOT --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --sigma_gaussian $STD_GAUSSIAN --geo $GEO --radius $RADIUS --yval $YVAL --niters_gan $NITERS --resume_niters_gan 0 --lr_gan $LR_GAN --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --kernel_sigma $SIGMA --threshold_type hard --kappa $KAPPA --eval --dim_gan $DIM_GAN #2>&1 | tee output_hard.txt

# echo "-------------------------------------------------------------------------------------------------"
# echo "CcGAN Soft"
# CUDA_VISIBLE_DEVICES=0 python3 main.py --root_path $ROOT_PATH --nsim $NSIM --seed $SEED --n_gaussians $N_GAUSSIANS --n_gaussians_plot $N_GAUSSIANS_PLOT --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --sigma_gaussian $STD_GAUSSIAN --geo $GEO --radius $RADIUS --yval $YVAL --niters_gan $NITERS --resume_niters_gan 0 --lr_gan $LR_GAN --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --kernel_sigma $SIGMA --threshold_type soft --kappa $KAPPA --eval --dim_gan $DIM_GAN #2>&1 | tee output_soft.txt
