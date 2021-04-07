import os
import cv2
from glob import glob
import numpy as np
from PIL import Image, ImageOps
import random
import numpy as np
from collections import Counter

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class CocoStuff164k(Dataset):
    def get_file_names(self, split):
        if split == "train2017":
          file_list = sorted(glob(os.path.join(self.root, "images", self.split, "*", "*.jpg")))
          print(len(file_list))
          file_list = ['/'.join(f.split("/")[-2:]).replace(".jpg", "") for f in file_list]
        else:
          file_list = sorted(glob(os.path.join(self.root, "images", self.split, "*.jpg")))
          print(len(file_list))
          file_list = [f.split("/")[-1].replace(".jpg", "") for f in file_list]  
        self.files = file_list
        
    def transform(self, image, label, h, w, crop):
      resize_image = transforms.Resize(size=(h, w))
      resize_label = transforms.Resize(size=(h, w), interpolation=transforms.InterpolationMode.NEAREST)
      image = resize_image(image)
      label = resize_label(label)

      i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(crop, crop))
      image = TF.crop(image, i, j, h, w)
      label = TF.crop(label, i, j, h, w)

      if random.random() > 0.5:
          image = TF.hflip(image)
          label = TF.hflip(label)
      return image, label

    def __init__(self, root, split, crop=321, scales=(0.5, 0.75, 1.0, 1.25, 1.5), trans=None):
        if split not in ["train2017", "val2017"]:
            raise ValueError("split of Dataset should be train2017 or val2017.")
        self.root = root
        self.split = split
        self.crop = crop
        self.scales = scales
        self.normalize = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.get_file_names(split)

    def __getitem__(self, idx):
        image_id = self.files[idx]
        image_path = os.path.join(self.root, "images", self.split, image_id + ".jpg")
        label_path = os.path.join(self.root, "annotations", self.split, image_id + ".png")

        image = Image.open(image_path)
        label = ImageOps.grayscale(Image.open(label_path))
        
        h, w = label.size
        scale_factor = np.random.choice(self.scales)
        h, w = int(h * scale_factor), int(w * scale_factor)
        crop = int(self.crop * scale_factor)
        image, label = self.transform(image, label, h, w, crop)

        image = self.normalize(image)
        label = np.asarray(label, np.int32)
        label = np.maximum(0, label - 91)

        return image, label

    def __len__(self):
      return len(self.files)

if __name__ == "__main__":
  trans = transforms.Compose([
    transforms.RandomCrop(321),
    transforms.RandomHorizontalFlip(p=0.5),
    #transforms.ToTensor(),
  ])
  data = CocoStuff164k('/content/drive/MyDrive/ML/dataset/', "train2017")
  for image, label in data:
    print(image.shape, label.shape)

