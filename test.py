import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import sys
import logging
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.SAEViT import SAEViT

num_classes = 2

iamges_path = "" # Write the test iamges path of the dataset here
save_path = "" # Write the images save path of the dataset here
checkpoint_path = "" # Write the checkpoint path of the dataset here

if not os.path.exists(save_path):
    os.makedirs(save_path)
iamges_list = os.listdir(iamges_path)
if len(iamges_list) == 0:
    raise RuntimeError(f"=> no image found at '{iamges_path}'\n")
print('Total number of images : ', len(iamges_list))

device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f"Device : {device}\n")

model = SAEViT().to(device)

checkpoint = torch.load(checkpoint_path, map_location='cuda')
model.load_state_dict(checkpoint['model'])

model.eval()
data_transforms = transforms.Compose([transforms.ToTensor()])

init_img = torch.zeros((1, 3, 512, 512), device=device)
model(init_img)

for iamge_name in iamges_list:
    iamge = Image.open(iamges_path + iamge_name).convert('RGB')
    iamge = np.array(iamge)
    iamge = data_transforms(iamge)
    iamge = torch.unsqueeze(iamge, dim=0)
    with torch.no_grad():
        output = model(iamge.to(device))
        prediction = output.cpu().numpy()
        prediction = np.argmax(prediction, axis=1)
        prediction = np.squeeze(prediction, 0) * 255
        prediction = Image.fromarray(prediction.astype('uint8'))
        prediction.save(f'{save_path}{iamge_name}.png')
