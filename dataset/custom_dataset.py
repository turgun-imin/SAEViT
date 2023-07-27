import os
import cv2
import numpy as np
import torch
import torch.utils.data as data

import sys
import logging
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from torchvision import transforms as T
import dataset.transforms as T


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


class CustomDataset(data.Dataset):
    def __init__(self, dataset_root_path: str, rot_flip: bool = True, train: bool = True, transforms=None):
        
        self.rot_flip = rot_flip

        assert os.path.exists(dataset_root_path), f"path '{dataset_root_path}' does not exist."
        if train:
            self.image_root = os.path.join(dataset_root_path, "train/", "image")
            self.mask_root = os.path.join(dataset_root_path, "train/", "label")
        else:
            self.image_root = os.path.join(dataset_root_path, "val/", "image")
            self.mask_root = os.path.join(dataset_root_path, "val/", "label")
        assert os.path.exists(self.image_root), f"path '{self.image_root}' does not exist."
        assert os.path.exists(self.mask_root), f"path '{self.mask_root}' does not exist."

        image_names = [p for p in os.listdir(self.image_root)]
        mask_names = [p for p in os.listdir(self.mask_root)]
        assert len(image_names) > 0, f"not find any images in {self.image_root}."

        # check images and mask
        re_mask_names = []
        for p in image_names:
            mask_name = p
            assert mask_name in mask_names, f"{p} has no corresponding mask."
            re_mask_names.append(mask_name)
        mask_names = re_mask_names

        self.images_path = [os.path.join(self.image_root, n) for n in image_names]
        self.masks_path = [os.path.join(self.mask_root, n) for n in mask_names]

        self.transforms = transforms

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        mask_path = self.masks_path[idx]
        image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        assert image is not None, f"failed to read image: {image_path}"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        target = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        assert target is not None, f"failed to read mask: {mask_path}"

        if self.rot_flip:
            image, target = random_rot_flip(image, target)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images_path)


if __name__ == '__main__':

    dataset_root_path = ""
    train_dataset = CustomDataset(dataset_root_path, train=True, transforms=T.ToTensor())
    print(len(train_dataset))
    val_dataset = CustomDataset(dataset_root_path, train=False, transforms=T.ToTensor())
    print(len(val_dataset))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=8,
                                               shuffle=True,
                                               num_workers=1,
                                               pin_memory=True
                                               )
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=8,
                                               shuffle=True,
                                               num_workers=1,
                                               pin_memory=True)
    print(len(train_loader))
    print(len(val_loader))
    for image, label in train_loader:
        print(image.shape)
        print(label.shape)