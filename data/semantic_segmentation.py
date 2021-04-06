import os
import cv2
from glob import glob
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class CocoStuff164k(Dataset):
    def get_file_names(self):
        file_list = sorted(glob(os.path.join(self.root, "images", self.split, "*.jpg")))
        file_list = [f.split("/")[-1].replace(".jpg", "") for f in file_list]
        self.files = file_list
        print(len(self.files))

    def __init__(self, root, split):
        if split not in ["train2017", "val2017"]:
            raise ValueError("split of Dataset should be train2017 or val2017.")
        self.root = root
        self.split = split

    def __getitem__(self, idx):
        image_id = self.files[idx]
        image_path = os.path.join(self.root, "images", self.split, image_id + ".jpg")
        label_path = os.path.join(self.root, "annotations", self.split, image_id + ".png")

        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        print(image.shape, label.shape)
