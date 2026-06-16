import os
import torch
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
from torch.utils import data
import torchvision.transforms as trans


def getData(mode: str, dataname: str):
    name = dataname.lower()

    if mode == 'train':
        df = pd.read_csv(f'{name}_train.csv')

    elif mode == 'pretrain':
        # df = pd.read_csv(f'{name}_train_single.csv')
        df = pd.read_csv(f'{name}_train+val.csv')

    elif mode == 'val':
        df = pd.read_csv(f'{name}_valid.csv')

    elif mode == 'test':
        df = pd.read_csv(f'{name}_test.csv')

    path = df['img_path'].tolist()
    label = df.drop(columns='img_path').values.tolist()
    return path, label


class NIHChestLoader(data.Dataset):
    def __init__(self, root: str, mode: str, classes: int, dataname: str):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.mode = mode
        self.classes = classes
        self.dataname = dataname

        self.img_name, self.label = getData(mode, dataname)

        if self.mode == 'train':
            self.transformations = trans.Compose([
                                                  trans.Resize((224, 224)),
                                                #   trans.CenterCrop(224),
                                                  trans.RandomHorizontalFlip(),
                                                  trans.RandomRotation(20),
                                                  trans.ToTensor(),
                                                  trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
        elif self.mode == 'pretrain': 
            self.transformations = trans.Compose([trans.Resize((224, 224)),
                                                #   trans.CenterCrop(224),
                                                  trans.RandomHorizontalFlip(),
                                                  trans.RandomRotation(20),
                                                  trans.ToTensor(),
                                                  trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
        else:
            self.transformations = trans.Compose([
                                                  trans.Resize((224, 224)),
                                                #   trans.CenterCrop(224),
                                                  trans.ToTensor(),
                                                  trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        print("> Found %d images..." % (len(self.img_name)))  

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):

        path = os.path.join(self.root, self.img_name[index])
        img = Image.open(path).convert('RGB')
        target = torch.tensor(self.label[index], dtype=torch.float32)

        if self.mode == 'pretrain': 
            img1 = self.transformations(img)
            img2 = self.transformations(img)

            return [img1, img2], target
        else:
            img = self.transformations(img)
            return img, target

    def get_labels(self):
        return self.label