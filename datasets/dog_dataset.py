""" Stanford Dogs (Dog) Dataset
Created: Nov 15,2019 - Yuchong Gu
Revised: Nov 15,2019 - Yuchong Gu
"""
import os
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
# from utils import get_transform

DATAPATH = '/fs/vulcan-projects/jigsaw_selfsup_shlokm/WS-DAN.PyTorch/dogs'


class DogDataset(Dataset):
    """
    # Description:
        Dataset for retrieving Stanford Dogs images and labels

    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, phase='train', resize=500):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.resize = resize
        self.num_classes = 120

        if phase == 'train':
            list_path = os.path.join(DATAPATH, 'train_list.mat')
        else:
            list_path = os.path.join(DATAPATH, 'test_list.mat')

        list_mat = loadmat(list_path)
        self.images = [f.item().item() for f in list_mat['file_list']]
        self.labels = [f.item() for f in list_mat['labels']]

        # transform
        # self.transform = get_transform(self.resize, self.phase)
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

    def __getitem__(self, item):
        # image
        image_id = self.image_id[item]

        # image
        image_orig = Image.open(os.path.join(DATAPATH, 'images', image_path[image_id])).convert('RGB')  # (C, H, W)
        image = self.transform(image_orig)
        image1 = self.transform(image_orig) # count begin from zero
        return image, image1

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    ds = DogDataset('train')
    # print(len(ds))
    for i in range(0, 1000):
        image, label = ds[i]
        # print(image.shape, label)
