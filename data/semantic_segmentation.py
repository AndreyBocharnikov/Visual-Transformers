import os
import cv2
from glob import glob
import numpy as np
from PIL import Image, ImageOps

from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CocoStuff164k(Dataset):
    def get_file_names(self):
        file_list = sorted(glob(os.path.join(self.root, "images", self.split, "*.jpg")))
        print(os.path.join(self.root, "images", self.split, "*.jpg"))
        print(len(file_list))
        file_list = [f.split("/")[-1].replace(".jpg", "") for f in file_list]
        self.files = file_list
        print("n_files =", len(self.files))

    def __init__(self, root, split, trans=None):
        if split not in ["train2017", "val2017"]:
            raise ValueError("split of Dataset should be train2017 or val2017.")
        self.root = root
        self.split = split
        self.get_file_names()

    def __getitem__(self, idx):
        image_id = self.files[idx]
        image_path = os.path.join(self.root, "images", self.split, image_id + ".jpg")
        label_path = os.path.join(self.root, "annotations", self.split, image_id + ".png")

        image = Image.open(image_path)
        label = ImageOps.grayscale(Image.open(label_path))
        #image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        #label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if trans is not None:
          image = trans(image)
        print(image.size, label.size)
        print(np.asarray(image)[:5], np.asarray(label)[:5])
    
    def __len__(self):
      return len(self.files)

if __name__ == "__main__":
  trans = transforms.Compose([
    transforms.RandomCrop(321),
    transforms.RandomHorizontalFlip(p=0.5),
    #transforms.ToTensor(),
  ])
  data = CocoStuff164k('/content/drive/MyDrive/ML/dataset/', "val2017", trans)
  for _ in data:
    break


