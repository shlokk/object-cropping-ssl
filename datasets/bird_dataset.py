""" CUB-200-2011 (Bird) Dataset
Created: Oct 11,2019 - Yuchong Gu
Revised: Oct 11,2019 - Yuchong Gu
"""
import os
import pdb
from PIL import Image
from torch.utils.data import Dataset
# from utils import get_transform
import torchvision.transforms as transforms
import moco.loader
import moco.builder
DATAPATH = '/fs/vulcan-projects/jigsaw_selfsup_shlokm/WS-DAN.PyTorch/CUB_200_2011'



class BirdDataset(Dataset):
    """
    # Description:
        Dataset for retrieving CUB-200-2011 images and labels

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
        self.image_id = []
        self.num_classes = 200
        print(os.path.join(DATAPATH, 'images.txt'))
        self.image_path = {}
        self.image_label = {}

        # get image path from images.txt
        with open(os.path.join(DATAPATH, 'images.txt')) as f:
            for line in f.readlines():
                id, path = line.strip().split(' ')
                self.image_path[id] = path

        # get image label from image_class_labels.txt
        with open(os.path.join(DATAPATH, 'image_class_labels.txt')) as f:
            for line in f.readlines():
                id, label = line.strip().split(' ')
                self.image_label[id] = int(label)

        # get train/test image id from train_test_split.txt
        with open(os.path.join(DATAPATH, 'train_test_split.txt')) as f:
            for line in f.readlines():
                image_id, is_training_image = line.strip().split(' ')
                is_training_image = int(is_training_image)

                if self.phase == 'train' and is_training_image:
                    self.image_id.append(image_id)
                if self.phase in ('val', 'test') and not is_training_image:
                    self.image_id.append(image_id)

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
        # get image id
        # print(self.image_id)

        image_id = self.image_id[item]
        # print(image_path)

        # image
        image_orig = Image.open(os.path.join(DATAPATH, 'images', self.image_path[image_id])).convert('RGB')  # (C, H, W)
        image = self.transform(image_orig)
        image1 =  self.transform(image_orig)

        # return image and label
        # return image, image_label[image_id] - 1  # count begin from zero
        return image, image1

    def __len__(self):
        return len(self.image_id)


if __name__ == '__main__':
    ds = BirdDataset('train')
    print(len(ds))
    for i in range(0, 10):
        image, label = ds[i]
        print(image.shape, label)
