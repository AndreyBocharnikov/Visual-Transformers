import os
from glob import glob
from PIL import Image, ImageOps
import random
import numpy as np

from torch.utils.data import Dataset
import torchvision.transforms as transforms


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
        label = Image.open(label_path)

        if self.split == "train2017":
            image, label = self.transform(image, label)
        else:
            resize_image = transforms.Resize(size=(512, 512))
            resize_label = transforms.Resize(size=(512, 512), interpolation=transforms.InterpolationMode.NEAREST)
            image = resize_image(image)
            label = resize_label(label)

        image = self.normalize(image)
        label = np.asarray(label, np.int64)
        label -= 92
        label[label < 0] = self.ignore_index

        return image, label

    def __len__(self):
        return len(self.files)
