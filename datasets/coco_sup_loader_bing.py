
import os

import numpy as np
from PIL import Image
from torchvision import datasets as datasets
import torch
import numpy as np
from torchvision import transforms
import os
import moco.loader
import moco.builder
import random


class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, args, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.root_coco_bing = 'coco_crops/'
        self.dilation = args.dilation

        self.ids = list(self.coco.imgToAnns.keys())
        # self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
        self.transform_single_Crop = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        self.crop_image_dict = np.load('/home/dilipkay_google_com/filestore/shlok/crop_image_dict_coco.npy', allow_pickle='TRUE').item()
        # print(self.cat2cat)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        self.random_crops = 10
        crop_image_dict = self.crop_image_dict

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output


        # path = coco.loadImgs(img_id)[0]['file_name']
        # image_number = np.random.randint(0, 10)
        # path_crop_jpg = path.split('.')  # (C, H, W)
        #
        # path_crop = path_crop_jpg[0] + '_' + str(image_number) + '.' + path_crop_jpg[1]
        # if not os.path.exists(os.path.join(self.root_coco_bing,path_crop)):
        #     path_crop = path_crop_jpg[0] + '_' + str(1) + '.' + path_crop_jpg[1]
        #
        # img = Image.open(os.path.join(self.root, path)).convert('RGB')
        # try:
        #     img_bing = Image.open(os.path.join(self.root_coco_bing,path_crop)).convert('RGB')
        # except:
        #     path_crop = path_crop_jpg[0] + '_' + str(1) + '.' + path_crop_jpg[1]
        #     img_bing = Image.open(os.path.join(self.root_coco_bing, path_crop)).convert('RGB')
        #img1 = self.transform(img)
        #img2 = self.transform_single_Crop(img_bing)
        path = coco.loadImgs(img_id)[0]['file_name']
        image_number = np.random.randint(0, self.random_crops)
        # print(self.random_crops, image_number)
        path_crop_jpg = path.split('.')  # (C, H, W)

        path_crop = path_crop_jpg[0] + '_' + str(image_number) + '.' + path_crop_jpg[1]
        img_id = 'coco_crops_new/' + path_crop
        # if not os.path.exists(os.path.join(self.root_coco_bing,path_crop)):
        #     path_crop = path_crop_jpg[0] + '_' + str(1) + '.' + path_crop_jpg[1]
        if img_id not in crop_image_dict.keys():
            print("not found")
            path_crop = path_crop_jpg[0] + '_' + str(1) + '.' + path_crop_jpg[1]
            img_id = 'coco_crops_new/' + path_crop

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        image_orig = img
        width, height = img.size
        width_crop = width
        height_crop = height
        # try:
        #     img_bing = Image.open(os.path.join(self.root_coco_bing,path_crop)).convert('RGB')
        # except:
        #     path_crop = path_crop_jpg[0] + '_' + str(1) + '.' + path_crop_jpg[1]
        #     img_bing = Image.open(os.path.join(self.root_coco_bing, path_crop)).convert('RGB')
        # img_id = 'coco_crops_new/' + path_crop
        height1, height2, width1, width2 = self.extracy_bbox(crop_image_dict, height, img_id, width)

        crop_box = (int(width1), int(height1), int(width2), int(height2))
        img_bing = img.crop((crop_box))
        #print(self.dilation)
        height1, height2, width1, width2 = self.extracy_bbox(crop_image_dict, height, img_id, width,dilation=self.dilation)

        crop_box = (int(width1), int(height1), int(width2), int(height2))
        img_bing_dilated = img.crop((crop_box))
        # image_crop = img_bing_crop
        img1 = self.transform(img_bing)
        img2 = self.transform(img_bing_dilated)

        if random.random() > 0.5:
            return img1, img2
        else:
            return img2, img1
        # from PIL import Image
        # im = Image.fromarray(img)

        # im.save("your_file.jpeg")
        # print(target)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        # return img1,img2

    def extracy_bbox(self, crop_image_dict, height, img_id, width, dilation = 0.0):
        if dilation>0:
            width1 = int(float(crop_image_dict[img_id][0]))
            height1 = int(float(crop_image_dict[img_id][1]))
            width2 = int(float(crop_image_dict[img_id][2]))
            height2 = int(float(crop_image_dict[img_id][3]))
            width1 = width1 - int(width1 * dilation)
            height1 = height1 - int(height1 * dilation)
            width2 = width2 + int(width2 * dilation)
            height2 = height2 + int(height2 * dilation)
        else:
            width1 = int(float(crop_image_dict[img_id][0]))
            height1 = int(float(crop_image_dict[img_id][1]))
            width2 = int(float(crop_image_dict[img_id][2]))
            height2 = int(float(crop_image_dict[img_id][3]))
        diff = width2 - width1
        final_width = 224
        # import pdb
        # pdb.set_trace()
        # print("initial")
        # print(width1,height1,width2,height2)
        if diff < final_width:
            # pdb.set_trace()
            if width1 - final_width / 2 > 0:
                width1 = width1 - final_width / 2
            else:
                width1 = 0
                width2 = 224

            if width2 + final_width / 2 < width:
                width2 = width2 + final_width / 2
            else:
                # width1 = 0
                # width2 = 224
                width1 = width2 - 224
        diff = height2 - height1
        if diff < 224:
            if height1 - final_width / 2 > 0:
                height1 = height1 - final_width / 2
                # height2 = height1 + final_width
            else:
                height1 = 0
                height2 = 224

            if height2 + final_width / 2 < height:
                height2 = height2 + final_width / 2
            else:
                height1 = height2 - 224
            # img.save("img_orig.jpeg")
            if width2 > width:
                width2 = width
            if height2 > height:
                height2 = height
        if width2 > width:
            width2 = width
        if height2 > height:
            height2 = height
        return height1, height2, width1, width2

    def __len__(self):
        return len(self.ids)
