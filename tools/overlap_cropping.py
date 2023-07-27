import os

import cv2
import math
import numpy as np

from pathlib import Path

from PIL import Image

imgs_path = ''
save_path = ''
Path(save_path).mkdir(parents=True, exist_ok=True)

# # block size
# height = 512
# width = 512

height = 256
width = 256

# overlap
over_x = 0
over_y = 0
h_val = height - over_x
w_val = width - over_y

# Set whether to discard an image that does not meet the size
mandatory = False
imgs_list = os.listdir(imgs_path)
print('Total number of images : ', len(imgs_list))
for img_name in imgs_list:
    name = img_name[:-4]
    print(f'image {img_name} is cropping')
    img = Image.open(imgs_path + img_name)
    img = np.array(img)
    # img = cv2.copyMakeBorder(img, 18, 18, 18, 18, cv2.BORDER_CONSTANT, value=255.0)
    # plt.imshow(img)
    # plt.show()

    original_height = img.shape[0]
    original_width = img.shape[1]
    max_row = float((original_height - height) / h_val) + 1
    max_col = float((original_width - width) / w_val) + 1
    max_row = math.ceil(max_row) if mandatory == False else math.floor(max_row)
    max_col = math.ceil(max_col) if mandatory == False else math.floor(max_col)
    images = []
    for i in range(max_row):
        images_temp = []
        for j in range(max_col):
            temp_path = save_path + '/' + name + '_' + str(i) + '_' + str(j)
            if ((width + j * w_val) > original_width and (
                    i * h_val + height) <= original_height):
                temp = img[i * h_val:i * h_val + height, j * w_val:original_width]
                temp_path = save_path + '/' + name + '_' + str(i * h_val) + '_' + str(i * h_val + height)
                temp = Image.fromarray(temp).convert('RGBA')
                temp.save('{}.png'.format(temp_path))
                images_temp.append(temp)
            elif ((height + i * h_val) > original_height and (
                    j * w_val + width) <= original_width):
                temp = img[i * h_val:original_height, j * w_val:j * w_val + width]
                temp = Image.fromarray(temp).convert('RGBA')
                temp.save('{}.png'.format(temp_path))
                images_temp.append(temp)
            elif ((width + j * w_val) > original_width and (
                    i * h_val + height) > original_height):
                temp = img[i * h_val:original_height, j * w_val:original_width]
                temp = Image.fromarray(temp).convert('RGBA')
                temp.save('{}.png'.format(temp_path))
                images_temp.append(temp)
            else:
                temp = img[i * h_val:i * h_val + height, j * w_val:j * w_val + width]
                temp = Image.fromarray(temp).convert('RGBA')
                temp.save('{}.png'.format(temp_path))
                images_temp.append(temp)

        images.append(images_temp)

    print(len(images))