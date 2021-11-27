#!/bin/bash

echo "Downloading dataset ..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Un-_cwgPoSikJmW4hT7Y8T8S1f-FUVA3' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Un-_cwgPoSikJmW4hT7Y8T8S1f-FUVA3" -O datasets/apples_RGB_NIR.zip && rm -rf /tmp/cookies.txt

echo "Unzip dataset ..."
unzip -q datasets/apples_RGB_NIR.zip -d datasets/

echo "Remove zip file ..."
rm datasets/apples_RGB_NIR.zip

echo "Convert dataset ..."
python datasets/convert_apple_dataset.py
python datasets/make_dataset_aligned.py --dataset-path datasets/apples_RGB_NIR
