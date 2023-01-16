
import os
import pdb
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import moco.loader
import moco.builder
import numpy as np
import torch
import pickle
from torchvision.utils import save_image
from PIL import Image
FILENAME_LENGTH = 7


class OpenImages(Dataset):
    """
    # Description:
        Dataset for retrieving FGVC Aircraft images and labels

    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, phase='train', resize=500,DATAPATH='',args=''):
        # assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.resize = resize
        self.DATAPATH = DATAPATH

        self.images = []
        self.labels = []
        self.labels_to_idx = {}
        self.images_test = []
        self.labels_test = []
        self.bbox = []

        images_selected_test = np.load('images_selected_test_40k.npy', allow_pickle='TRUE').item()
        if self.phase=='train' or self.phase=='train_sup':
            if args.one_class_dataset:
                images_selected = np.load('filename_dict_1_classes_train.npy', allow_pickle='TRUE').item()
            elif args.multi_class_dataset:
                print("filename_dict_greater_than_1_classes_train.npy")
                images_selected = np.load('filename_dict_greater_than_1_classes_train.npy', allow_pickle='TRUE').item()
            else:
                print("images_selected_new.npy")
                images_selected = np.load('images_selected_new.npy', allow_pickle='TRUE').item()
        else:
            images_selected = np.load('images_selected_test_40k.npy', allow_pickle='TRUE').item()
        cnt = 0
        for all_images in images_selected.keys():

            self.images.append(all_images+'.jpg')
            self.labels.append(images_selected[all_images])
            for l in images_selected[all_images]:
                if l not in self.labels_to_idx.keys():
                    self.labels_to_idx[l] = cnt
                    cnt = cnt + 1
        print("after images_selected ")


        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        self.transform = transforms.Compose([
            # transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        self.test_trasform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

        self.train_transform= transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ])

    def __getitem__(self, item):
        image_orig = Image.open(os.path.join(self.DATAPATH, self.images[item])).convert('RGB')  # (C, H, W)
        labes_one_hot = [0] * len(self.labels_to_idx)

        if self.phase == 'train':
            image = self.transform(image_orig)
            image1 = self.transform(image_orig)
            return image, image1
        if self.phase == 'train_sup':
            for l in self.labels[item]:
                labes_one_hot[self.labels_to_idx[l]] = 1
            labes_one_hot = torch.tensor(labes_one_hot, dtype=torch.float32)
            image = self.train_transform(image_orig)
            return image, labes_one_hot
        else:
            image = self.test_trasform(image_orig)
            for l in self.labels_test[item]:
                labes_one_hot[self.labels_to_idx[l]] = 1
            labes_one_hot = torch.tensor(labes_one_hot, dtype=torch.float32)
            return image,labes_one_hot


    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    ds = AircraftDataset('test', 448)
    # print(len(ds))
    from utils import AverageMeter
    height_meter = AverageMeter('height')
    width_meter = AverageMeter('width')

    for i in range(len(ds)):
        image, label = ds[i]
        avgH = height_meter(image.size(1))
        avgW = width_meter(image.size(2))
        print('H: %.2f, W: %.2f' % (avgH, avgW))
