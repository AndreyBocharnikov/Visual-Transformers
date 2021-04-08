import os
import cv2
from glob import glob
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random
import numpy as np
from collections import Counter

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F


def pad_images_and_labels(ignore_index):
  def iner(batch):
    hw = [0, 0]
    #print(type(batch))
    #print(len(batch))
    for image, label in batch:
      hw = np.maximum(hw, image.shape[1:])
    if hw[0] % 2 == 1:
      hw[0] += 1
    if hw[1] % 2 == 1:
      hw[1] += 1
    print(hw)
    images, labels = [], []
    for image, label in batch:
      #print(hw, image.shape[1:])
      left, top = hw - image.shape[1:]
      images.append(F.pad(image, (top, 0, left, 0)))
      #print(image.shape, images[-1].shape)
      labels.append(F.pad(torch.from_numpy(label), (top, 0, left, 0), value=ignore_index))
      #print(label.shape, labels[-1].shape)
    #print(torch.stack(images).shape)
    #print(torch.stack(labels).shape)
    return torch.stack(images), torch.stack(labels)
  return iner

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
        
    def transform(self, image, label, base_size=512, crop_size=512):
        short_size = random.randint(int(base_size * 0.5), int(base_size * 2.0))
        w, h = image.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        image = image.resize((ow, oh), Image.BILINEAR)
        label = label.resize((ow, oh), Image.NEAREST)

        if short_size < crop_size:
          padw = max(0, crop_size - ow)
          padh = max(0, crop_size - oh)
          image = ImageOps.expand(image, border=(0, 0, padw, padh), fill=0)
          label = ImageOps.expand(label, border=(0, 0, padw, padh), fill=self.ignore_index)

        w, h = image.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        image = image.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        label = label.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        return image, label

        """
      i, j, h, w = transforms.RandomResizedCrop.get_params(
        image, scale=(0.75, 1.0), ratio=(0.75, 1.3333333333333333))

      #i, j, h, w = transforms.RandomCrop.get_params(
      #      image, output_size=(crop, crop))
      image = TF.crop(image, i, j, h, w)
      label = TF.crop(label, i, j, h, w)

      resize_image = transforms.Resize(size=(320, 320))
      resize_label = transforms.Resize(size=(320, 320), interpolation=transforms.InterpolationMode.NEAREST)
      image = resize_image(image)
      label = resize_label(label)

      if random.random() > 0.5:
          image = TF.hflip(image)
          label = TF.hflip(label)
      return image, label
      """
      

    def __init__(self, root, split, ignore_index):
        if split not in ["train2017", "val2017"]:
            raise ValueError("split of Dataset should be train2017 or val2017.")
        self.root = root
        self.split = split
        self.ignore_index = ignore_index
        self.normalize = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.get_file_names(split)

    def __getitem__(self, idx):
        image_id = self.files[idx]
        image_path = os.path.join(self.root, "images", self.split, image_id + ".jpg")
        label_path = os.path.join(self.root, "annotations", self.split, image_id + ".png")

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path) #ImageOps.grayscale(Image.open(label_path))
        
        #h, w = label.size
        #scale_factor = np.random.choice(self.scales)
        #h, w = int(h * scale_factor), int(w * scale_factor)
        #crop = int(self.crop * scale_factor)
        if self.split == "train2017":
          image, label = self.transform(image, label)
        else:
          resize_image = transforms.Resize(size=(320, 320))
          resize_label = transforms.Resize(size=(320, 320), interpolation=transforms.InterpolationMode.NEAREST)
          image = resize_image(image)
          label = resize_label(label)

        image = self.normalize(image)
        label = np.asarray(label, np.int64)
        #label = np.maximum(0, label - 91)
        label -= 92
        label[label < 0] = self.ignore_index
        
        #if image.shape != torch.Size([3, 320, 320]):
        #  print(image_path)
        return image, label

    def __len__(self):
      return len(self.files)

if __name__ == "__main__":
  trans = transforms.Compose([
    transforms.RandomCrop(512),
    transforms.RandomHorizontalFlip(p=0.5),
    #transforms.ToTensor(),
  ])
  data = CocoStuff164k('/content/drive/MyDrive/ML/dataset/', "train2017")
  for image, label in data:
    print(image.shape, label.shape)

