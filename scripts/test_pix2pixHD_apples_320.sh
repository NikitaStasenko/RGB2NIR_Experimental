set -ex
python test.py --ngf 32 --dataroot ../data/test_data --dataset_mode apples --name apples_pix2pixHD_320 --model pix2pixHD --netD pix2pixHD_multiscale --netG pip2pixHD_local --norm batch --input_nc 3 --output_nc 3 --nir_channels_only
