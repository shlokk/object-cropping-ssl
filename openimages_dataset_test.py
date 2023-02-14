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

DATAPATH = '/fs/vulcan-projects/jigsaw_selfsup_shlokm/openimages/train/test/'
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

    def __init__(self, phase='train', resize=500, args=''):
        # assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.resize = resize
        self.DATAPATH = args.DATAPATH

        self.images = []
        self.labels = []
        self.labels_to_idx = {}
        self.images_test = []
        self.labels_test = []
        images_selected = np.load('images_selected_new.npy', allow_pickle='TRUE').item()

        images_selected_test = np.load('images_selected_test_40k.npy', allow_pickle='TRUE').item()
        cnt = 0
        for images in images_selected.keys():
            self.images.append(images+'.jpg')
            self.labels.append(images_selected[images])
            for l in images_selected[images]:
                if l not in self.labels_to_idx.keys():
                    self.labels_to_idx[l] = cnt
                    cnt = cnt + 1
        for images in images_selected_test.keys():
            self.images_test.append(images+'.jpg')
            self.labels_test.append(images_selected_test[images])

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        self.transform = transforms.Compose([
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

        image_orig = Image.open(os.path.join(DATAPATH, self.images_test[item])).convert('RGB')  # (C, H, W)
        labes_one_hot = [0] * len(self.labels_to_idx)

        image = self.test_trasform(image_orig)
        for l in self.labels_test[item]:
            labes_one_hot[self.labels_to_idx[l]] = 1
        labes_one_hot = torch.tensor(labes_one_hot, dtype=torch.float32)

        return image,labes_one_hot

        # return image and label
        # return image, self.labels[item]  # count begin from zero

    def __len__(self):
        return len(self.images_test)


if __name__ == '__main__':
    ds = OpenImages('test', 448)
    # print(len(ds))
    print(ds.__len__())
