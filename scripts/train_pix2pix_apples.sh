set -ex
python train.py --checkpoints_dir /nmnt/x2-hdd/experiments/pulmonary_trunk/new_test/checkpoints --dataroot /nmnt/x2-hdd/experiments/pulmonary_trunk/new_test/dset1/train --dataset_mode apples --name apples_pix2pix --model pix2pix --netG unet_64 --lambda_L1 100 --norm batch --pool_size 0 --input_nc 3 --output_nc 3 --display_ncols 3 --save_epoch_freq 30 --nir_channels_only
