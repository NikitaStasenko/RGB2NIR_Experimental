set -ex
python train.py --size 160 --dataroot ../data/train_data --dataset_mode apples --name apples_pix2pixHD_160 --model pix2pixHD --netD pix2pixHD_multiscale --netG pip2pixHD_global --norm batch --pool_size 0 --input_nc 3 --output_nc 3 --display_ncols 3 --save_epoch_freq 30 --n_epochs 100 --n_epochs_fix_global 30 --n_epochs_decay 100 --nir_channels_only
