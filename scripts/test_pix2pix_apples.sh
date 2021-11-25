set -ex
python test.py  --results_dir /nmnt/x2-hdd/experiments/pulmonary_trunk/new_test/results --checkpoints_dir /nmnt/x2-hdd/experiments/pulmonary_trunk/new_test/checkpoints --dataroot /nmnt/x2-hdd/experiments/pulmonary_trunk/new_test/dset1/test --dataset_mode apples --name apples_pix2pix --model pix2pix --netG unet_64 --norm batch --input_nc 3 --output_nc 3 --nir_channels_only --phase test
