#!/bin/bash

echo "Downloading dataset ..."
# FIXME: change path to zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yEiquJg9inIFgR3F-N5Z3DbFnXJ0aXmA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yEiquJg9inIFgR3F-N5Z3DbFnXJ0aXmA" -O dataset/apples.zip && rm -rf /tmp/cookies.txt

# TODO: unzip

rm dataset/apples.zip 