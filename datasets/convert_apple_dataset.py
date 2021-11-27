import os

import numpy as np

from PIL import Image
from shutil import copyfile

channels = [5, 6, 7]
train_procent = 80

dataset_path = 'datasets/apples_RGB_NIR'
dataset_path_A = os.path.join(dataset_path, 'A')
dataset_path_B = os.path.join(dataset_path, 'B')

images_A = [f for f in os.listdir(dataset_path_A)]
images_B = [f for f in os.listdir(dataset_path_B)]

os.makedirs(os.path.join(dataset_path, 'trainA'), exist_ok=True)
os.makedirs(os.path.join(dataset_path, 'trainB'), exist_ok=True)
os.makedirs(os.path.join(dataset_path, 'testA'), exist_ok=True)
os.makedirs(os.path.join(dataset_path, 'testB'), exist_ok=True)

# Split NIR by name
images_B_dict = {}
for im in images_B:
    split = im.split('_')
    if len(split) < 3 or 'channel' not in im:
        continue
    
    extension = os.path.splitext(im)[1]
    # split = im.split('_')
    channel = int(os.path.splitext(split[-1])[0][7:])
    if channel in channels:
        new_im_name = '{1}_{2:.4}00'.format(*split)
        if new_im_name not in images_B_dict:
            images_B_dict[new_im_name] = {}
        images_B_dict[new_im_name][channel] = im

# Save only channels you need    
for im in images_B_dict:
    images = []
    for ch in channels:
        im_path = os.path.join(dataset_path_B, images_B_dict[im][ch])
        images.append(np.array(Image.open(im_path)))
    images_B_dict[im] = np.stack(images, axis=2)

# Split and rename RGB images
train_size = int(len(images_A) * 0.8)
print(train_size)
count = 0
for im in images_A[:train_size]:
    extension = os.path.splitext(im)[1]
    split = im.split('_')
    new_im_name = '20{0}{1}{2}_{3}{4}{5}'.format(*split)
    if new_im_name in images_B_dict:
        copyfile(os.path.join(dataset_path_A, im), os.path.join(dataset_path, 'trainA', new_im_name + '.png'))
        Image.fromarray(images_B_dict[new_im_name]).save(os.path.join(dataset_path, 'trainB', new_im_name + '.png'))
        count +=1
print(count)
print(len(images_A) - train_size)
count = 0
for im in images_A[train_size:]:
    file_name, extension = os.path.splitext(im)
    split = im.split('_')
    new_im_name = '20{0}{1}{2}_{3}{4}{5}'.format(*split)
    if new_im_name in images_B_dict:
        copyfile(os.path.join(dataset_path_A, im), os.path.join(dataset_path, 'testA', new_im_name + '.png'))
        Image.fromarray(images_B_dict[new_im_name]).save(os.path.join(dataset_path, 'testB', new_im_name + '.png'))
        count +=1
print(count)
