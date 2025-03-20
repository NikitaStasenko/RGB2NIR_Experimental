# RGB2NIR_Experimental
This repository contains several image-to-image translation models, which were tested for RGB to NIR image generation.  The models are Pix2Pix, Pix2PixHD, CycleGAN and PointWise.

## Dataset
The dataset including RGB and NIR images is located here: https://drive.google.com/drive/folders/1uJx_SLi0ePYqhn-lJ8p5whAy5spUsect?usp=sharing, where "RGB_Images" and "MSI_images" containg RGB and mltispectral images acquired with digtial and multispectral 9 band camera at the same time periods, respectively. "RGB_NIR_Decayed_apples" contains RGB and multispectral images acquired with multispectral camera.
The models were trained and tested on images from "RGB_Images" and "MSI_Images", respectively.

## How to read and understand RGB image
20_12_26_22_15_00_Canon_top_all_on.jpg: 20 - year (it was 2020 for current image); 12 - month (December); 26 - date/day (26th); 22 - hours (22:00 or 10 pm), 15 - minutes, 00 - seconds; top_all_on - location.

## How to read and understand multispectral image
set10_20201226_221732_686_00000_channel0.png: set10 - number of set; 2020 - year (it was 2020 for current image); 12 - month (December); 26 - date/day (26th); 22 - hours (22:00 or 10 pm), 17 - minutes, 32 - seconds; 686_00000_ - number of image; channel0 - number of multispectral camera's band.

### The values of each multispectral camera's channels     
channel0 = 561 nm, channel1 = 597 nm, channel2 = 635 nm, channel3 = 635 nm, channel4 = 724 nm, channel5 = 762 nm, channel6 = 802 nm, channel7 = 838 nm; channel8 (panchromatic) = 0 nm.

## Citation
In this project we used the code and methodolgy by:

@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
