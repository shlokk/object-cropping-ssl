""" FGVC Aircraft (Aircraft) Dataset
Created: Nov 15,2019 - Yuchong Gu
Revised: Nov 15,2019 - Yuchong Gu
"""
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
import random

DATAPATH = '/fs/vulcan-projects/jigsaw_selfsup_shlokm/openimages/train/all_images/'
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

    def __init__(self, phase='train', resize=500,full_dataset=True,DATAPATH='',args=''):
        # assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.resize = resize
        self.DATAPATH = args.DATAPATH

        variants_dict = {}
        # with open(os.path.join(DATAPATH, 'variants.txt'), 'r') as f:
        #     for idx, line in enumerate(f.readlines()):
        #         variants_dict[line.strip()] = idx
        # self.num_classes = len(variants_dict)
        #
        # if self.phase == 'train':
        #     list_path = os.path.join(DATAPATH, 'images_variant_trainval.txt')
        # else:
        #     list_path = os.path.join(DATAPATH, 'images_variant_test.txt')

        self.images = []
        self.labels = []
        self.labels_to_idx = {}
        self.images_test = []
        self.labels_test = []
        self.bbox = []
        self.full_dataset = full_dataset
        images_selected_with_all_features = np.load('images_selected_with_all_features.npy',  allow_pickle='TRUE').item()
        images_selected = np.load('images_selected_new.npy', allow_pickle='TRUE').item()

        images_selected_test = np.load('images_selected_test_40k.npy', allow_pickle='TRUE').item()
        cnt = 0
        for all_images in images_selected.keys():
            # print(images)
            # if len(images_selected[images])==0:
            #     print(images_selected[images])
            cnt_small = 0
            for images in images_selected_with_all_features[all_images]['data']:
                self.images.append(all_images+'.jpg')
                self.labels.append(images_selected[all_images])
                # import pdb
                # pdb.set_trace()
                self.bbox.append(images)
                # print(images)
                cnt_small = cnt_small + 1
                for l in images_selected[all_images]:
                    if l not in self.labels_to_idx.keys():
                        self.labels_to_idx[l] = cnt
                        cnt = cnt + 1
        # with open(list_path, 'r') as f:
        #     for line in f.readlines():
        #         fname_and_variant = line.strip()
        #         self.images.append(fname_and_variant[:FILENAME_LENGTH])
        #         self.labels.append(variants_dict[fname_and_variant[FILENAME_LENGTH + 1:]])

        # transform
        # self.transform = get_transform(self.resize, self.phase)
        for images in images_selected_test.keys():
            self.images_test.append(images+'.jpg')
            self.labels_test.append(images_selected_test[images])

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
        # image
        # print(os.path.join(DATAPATH, self.images[item]))
        image_orig = Image.open(os.path.join(self.DATAPATH, self.images[item])).convert('RGB')  # (C, H, W)
        width, height = image_orig.size
        # print(width, height)
        # import pdb
        # pdb.set_trace()
        # xmin,ymin,xmax,ymax = self.bbox[items][4],self.bbox[items][5],self.bbox[items][6],self.bbox[items][7]
        # print(self.bbox[item][4]*width)
        # print(self.bbox[item])

        # print(crop_box)
        # print(image_orig.size)
        # crop_box = map(int, crop_box)
        # print(crop_box)
        # image = self.transform(image)

        # print(image_orig.size)
        if self.full_dataset==True:
            # if random.random()>0.2:
            #     crop_box = (int(float(float(self.bbox[item][4]) * width)), int(float(self.bbox[item][6]) * height),
            #                 int((float(self.bbox[item][5]) * width)),
            #                 int(float(self.bbox[item][7]) * height))
            #     image_orig = image_orig.crop((crop_box))
            #     width, height = image_orig.size
            width, height = image_orig.size
            crop_box = (int(float(float(self.bbox[item][4]) * width)), int(float(self.bbox[item][6]) * height),
                        int((float(self.bbox[item][5]) * width)),
                        int(float(self.bbox[item][7]) * height))
            image_crop = image_orig.crop((crop_box))
            width_crop, height_crop = image_crop.size
            if self.phase == 'train':
                image = self.transform(image_crop)
                image1 = self.transform(image_orig)
                label = self.labels_to_idx[self.bbox[item][2]]
                return image, image1,label
        else:
            crop_box = (int(float(float(self.bbox[item][4]) * width)), int(float(self.bbox[item][6]) * height),
                        int((float(self.bbox[item][5]) * width)),
                        int(float(self.bbox[item][7]) * height))
            image_orig = image_orig.crop((crop_box))
            width, height = image_orig.size


        if width==0 or height==0:
            item=100
            print('in 100')
            image_orig = Image.open(os.path.join(DATAPATH, self.images[item])).convert('RGB')  # (C, H, W)
            width, height = image_orig.size
            crop_box = (int(float(float(self.bbox[item][4]) * width)), int(float(self.bbox[item][6]) * height),
                        int((float(self.bbox[item][5]) * width)),
                        int(float(self.bbox[item][7]) * height))
            image_crop = image_orig.crop((crop_box))
            if self.phase == 'train':
                image = self.transform(image_crop)
                image1 = self.transform(image_orig)
                return image, image1
            # image_orig = image_orig.crop((crop_box))
        labes_one_hot = [0] * len(self.labels_to_idx)
        # print(self.phase)
        if self.phase == 'train':
            # image = self.transform(image_orig)
            # image1 = self.transform(image_orig)
            # print(image.shape)
            # print(image1.shape)
            image = self.transform(image_crop)
            image1 = self.transform(image_orig)
            # print("here")
            return image, image1
            # if random.random() > 0.5:
            #     return image, image1
            # else:
            #     return image1, image
        if self.phase == 'train_sup':
            # for l in self.bbox[item]:
            # pdb.set_trace()
            # labes_one_hot[self.labels_to_idx[self.bbox[item][2]]] = 1
            # labes_one_hot = torch.tensor(labes_one_hot, dtype=torch.float32)
            # print(labes_one_hot)
            image = self.train_transform(image_orig)
            # label = labes_one_hot[self.labels_to_idx[self.bbox[item][2]]]
            label = self.labels_to_idx[self.bbox[item][2]]
            # print(label)
            # print(image.shape)
            # print(self.labels)
            # torch.tensor(self.labels[item].tolist(), dtype=torch.float32)
            return image, label

        # return image and label
        #     return image, self.labels[item]
        #     return image, image_label[image_id] - 1  # count begin from zero

        else:
            image = self.test_trasform(image_orig)
            # for l in self.labels_test[item]:
            #     labes_one_hot[self.labels_to_idx[l]] = 1
            # pdb.set_trace()
            # labes_one_hot[self.labels_to_idx[self.bbox[item][2]]] = 1
            # labes_one_hot = torch.tensor(labes_one_hot, dtype=torch.float32)
            # print(labes_one_hot)
            label = self.labels_to_idx[self.bbox[item][2]]

            # label = labes_one_hot[self.labels_to_idx[self.bbox[item][2]]]

            return image,label

        # return image and label
        # return image, self.labels[item]  # count begin from zero

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
