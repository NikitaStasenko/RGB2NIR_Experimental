set -ex
python test.py --dataroot ../data/test_data --dataset_mode apples --name apples_pix2pix_pointwise --model pix2pix --netG pointwise --norm batch --input_nc 3 --output_nc 3 --nir_channels_only --act leaky_relu --n_layers_G 6
python test.py --dataroot ../data/test_data --dataset_mode apples --name apples_pix2pix_pointwise_no_prep --model pix2pix --netG pointwise --norm batch --input_nc 3 --output_nc 3 --nir_channels_only --act leaky_relu --n_layers_G 6 --no_preprocess_pointwise
