set -ex
python train.py --dataroot ../data/train_data --dataset_mode apples --name apples_cyclegan --model cycle_gan --netG resnet_6blocks --pool_size 50 --no_dropout --input_nc 3 --output_nc 3 --lambda_identity 0.0 --display_ncols 3 --save_epoch_freq 30 --nir_channels_only
