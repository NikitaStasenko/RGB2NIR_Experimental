set -ex
python train.py --display_id 0 --dataroot ../data/train_data --dataset_mode apples --name apples_pix2pix_netD_pix --model pix2pix --netG unet_64 --netD pixel --direction AtoB --lambda_L1 100 --norm batch --pool_size 0 --input_nc 3 --output_nc 9 --display_ncols 9 --save_epoch_freq 40
