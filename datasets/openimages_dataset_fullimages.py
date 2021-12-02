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
import random
import math

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

    def __init__(self, phase='train', resize=500,full_dataset=True,rescale_crops_before=False,DATAPATH='',rescale_parameter=0,radius=0,args=''):
        # assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.resize = resize
        self.DATAPATH = args.DATAPATH


        self.images = []
        self.labels = []
        self.labels_to_idx = {}
        self.images_test = []
        self.labels_test = []
        self.bbox = []
        self.full_dataset = full_dataset
        self.rescale_crops_before = rescale_crops_before
        self.rescale_parameter = rescale_parameter
        images_selected_with_all_features = np.load('images_selected_with_all_features.npy',  allow_pickle='TRUE').item()
        images_selected = np.load('images_selected_new.npy', allow_pickle='TRUE').item()
        self.radius = radius
        print(self.radius)
        # print(args)
        self.use_supervised_boxes = args.use_supervised_boxes
        self.use_combination_boxes = args.use_combination_boxes
        self.use_all_three_modalities = args.use_all_three_modalities
        print(self.use_all_three_modalities)


        images_selected_test = np.load('images_selected_test_40k.npy', allow_pickle='TRUE').item()
        cnt = 0
        # print("self.full_dataset"+str(self.full_dataset))
        if self.rescale_crops_before:
            images_selected = np.load('images_selected_new.npy', allow_pickle='TRUE').item()
            cnt = 0
            for all_images in images_selected.keys():
                # print(images)
                # if len(images_selected[images])==0:
                #     print(images_selected[images])
                # cnt_small = 0
                # for images in images_selected_with_all_features[all_images]['data']:
                self.images.append(all_images + '.jpg')
                self.labels.append(images_selected[all_images])
                # self.bbox.append(images)
                # print(images)
                # cnt_small = cnt_small + 1
                for l in images_selected[all_images]:
                    if l not in self.labels_to_idx.keys():
                        self.labels_to_idx[l] = cnt
                        cnt = cnt + 1
        else:
            for all_images in images_selected.keys():
                # print(images)
                # if len(images_selected[images])==0:
                #     print(images_selected[images])
                cnt_small = 0
                for images in images_selected_with_all_features[all_images]['data']:
                    self.images.append(all_images+'.jpg')
                    self.labels.append(images_selected[all_images])
                    self.bbox.append(images)
                    # print(images)
                    cnt_small = cnt_small + 1
                    for l in images_selected[all_images]:
                        if l not in self.labels_to_idx.keys():
                            self.labels_to_idx[l] = cnt
                            cnt = cnt + 1

        for images in images_selected_test.keys():
            self.images_test.append(images+'.jpg')
            self.labels_test.append(images_selected_test[images])

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        self.transform_single_Crop = transforms.Compose([
            transforms.Resize((224,224)),
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
        self.transform_random = transforms.Compose([
            # transforms.RandomCrop((self.rescale_parameter*width,self.rescale_parameter*height)),
            # transforms.RandomResizedCrop((self.rescale_parameter*width,self.rescale_parameter*height), scale=(0, 1.)),
            transforms.RandomResizedCrop(224, scale=(self.rescale_parameter, self.rescale_parameter)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    def is_safe(self,w1,h1,w2,h2,width_crop,height_crop):
        # print(w1,h1,w2,h2,width_crop,height_crop)
        if w1>0 and w1<width_crop and w2>0 and w2<width_crop and h1>0 and h1<height_crop and h2>0 and h2<height_crop:
            return True
        else:
            return False


    def __getitem__(self, item):
        # image
        # print(os.path.join(DATAPATH, self.images[item]))
        image_orig = Image.open(os.path.join(self.DATAPATH, self.images[item])).convert('RGB')  # (C, H, W)
        width, height = image_orig.size
        width_crop = width
        height_crop = height
        pi = math.pi
        #

        # if self.rescale_crops_before:
        #     width = image_orig.size[0]
        #     height = image_orig.size[1]
        #     # (width_crop, height_crop) =  image_orig.size()
        #
        #
        #     # print("in rescale crop")
        use_supervised_boxes =self.use_supervised_boxes
        use_combination_boxes = self.use_combination_boxes
        use_all_three_modalities = self.use_all_three_modalities
        if self.rescale_crops_before==False:
            if self.full_dataset==True:
                if use_supervised_boxes:
                    width, height = image_orig.size
                    # crop_box = (int(float(float(self.bbox[item][4]) * width)), int(float(self.bbox[item][6]) * height),
                    #             int((float(self.bbox[item][5]) * width)),
                    #             int(float(self.bbox[item][7]) * height))
                    # image_crop = image_orig.crop((crop_box))
                    # width_crop, height_crop = image_crop.size
                    width1 = int(float(self.bbox[item][4]) * width)
                    width2 = int(float(self.bbox[item][5]) * width)
                    height1 = int(float(self.bbox[item][6]) * height)
                    height2 = int(float(self.bbox[item][7]) * height)
                    diff = width2 - width1
                    final_width = 224
                    # print("initial")
                    # print(width1,height1,width2,height2)
                    if diff<final_width:
                        # pdb.set_trace()
                        if width1-final_width/2>0:
                            width1 = width1-final_width/2
                        else:
                            width1 = 0
                            width2 = 224

                        if width2+final_width/2<width:
                            width2 = width2+final_width/2
                        else:
                            # width1 = 0
                            # width2 = 224
                            width1 = width2 - 224

                    diff = height2 - height1
                    if diff<224:
                        if height1-final_width/2>0:
                            height1 = height1-final_width/2
                            # height2 = height1 + final_width
                        else:
                            height1 = 0
                            height2 = 224

                        if height2+final_width/2<height:
                            height2 = height2+final_width/2
                        else:
                            height1 = height2 - 224
                            # height2 = 224
                        # pdb.set_trace()
                    # print("final")
                    crop_box = (width1,height1,width2,height2)
                    # print(crop_box)
                    # print(width,height)
                    # pdb.set_trace()
                    image_crop = image_orig.crop((crop_box))
                elif use_combination_boxes:
                    width, height = image_orig.size
                    # crop_box = (int(float(float(self.bbox[item][4]) * width)), int(float(self.bbox[item][6]) * height),
                    #             int((float(self.bbox[item][5]) * width)),
                    #             int(float(self.bbox[item][7]) * height))
                    # image_crop = image_orig.crop((crop_box))
                    # width_crop, height_crop = image_crop.size
                    width1 = int(float(self.bbox[item][4]) * width)
                    width2 = int(float(self.bbox[item][5]) * width)
                    height1 = int(float(self.bbox[item][6]) * height)
                    height2 = int(float(self.bbox[item][7]) * height)
                    diff = width2 - width1
                    final_width = 224
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
                            # height2 = 224
                        # pdb.set_trace()
                    # print("final")
                    crop_box = (width1, height1, width2, height2)
                    # print(crop_box)
                    # print(width,height)
                    # pdb.set_trace()
                    image_crop = image_orig.crop((crop_box))

                    number_of_points = 100
                    radius = np.random.randint(self.radius[0],self.radius[1])
                    crop_size = 224

                    for x in range(0, number_of_points + 1):
                        # self.radius[0]
                        width1_new_point = math.cos(
                            2 * pi / number_of_points * random.randint(0, number_of_points)) * radius
                        height1_new_point = math.sin(
                            2 * pi / number_of_points * random.randint(0, number_of_points)) * radius
                        width1_new_point_final = width1_new_point + width1
                        height1_new_point_final = height1_new_point + height1
                        width2_new_point = width1_new_point_final + crop_size
                        height2_new_point = height1_new_point_final + crop_size
                        # pdb.set_trace()
                        image_list = []
                        if self.is_safe(width1_new_point_final, height1_new_point_final, width2_new_point,
                                        height2_new_point, width_crop, height_crop):
                            # print("here")
                            # pdb.set_trace()

                            crop_box = (
                            int(width1_new_point_final), int(height1_new_point_final), int(width2_new_point),
                            int(height2_new_point))

                            img1 = image_orig.crop((crop_box))
                            image1 = self.transform_single_Crop(img1)
                            image_crop_transform = self.transform(image_crop)
                            image = self.transform(image_orig)

                            # image_crop = image_orig.crop((crop_box))
                            # image_list.append(image1)
                            # image1 = self.transform_single_Crop(image_crop)
                            # print(image.size(),image1.size())
                            if random.random() > 0.5:
                                return image1, image
                            else:
                                return image,image_crop_transform
                            # crop_box = (width1, height1, width2, height2)
                            # img2 = image_orig.crop((crop_box))
                            # print(img1.size,img2.size)
                            # return self.transform_single_Crop(img1), self.transform_single_Crop(img2)
                        width2_new_point = width1 - width1_new_point
                        height2_new_point = height1 - height1_new_point
                        width2_new_point_2 = width2_new_point - crop_size
                        height2_new_point_2 = height2_new_point - crop_size
                        if self.is_safe(width2_new_point, height2_new_point, width2_new_point_2, height2_new_point_2,
                                        width_crop, height_crop):
                            # print("here2")
                            crop_box = (int(width2_new_point_2), int(height2_new_point_2), int(width2_new_point),
                                        int(height2_new_point))
                            # print(crop_box)

                            img1 = image_orig.crop((crop_box))
                            crop_box = (width1, height1, width2, height2)
                            # img2 = image_orig.crop((crop_box))
                            # print(img1.size, img2.size)
                            image2 = self.transform_single_Crop(img1)
                            image_list.append(image2)
                            # image1 = self.transform_single_Crop(image_crop)
                            # print(image.size(),image1.size())
                            img1 = image_orig.crop((crop_box))
                            image1 = self.transform_single_Crop(img1)
                            image_crop_transform = self.transform(image_crop)
                            image = self.transform(image_orig)

                            # image_crop = image_orig.crop((crop_box))
                            # image_list.append(image1)
                            # image1 = self.transform_single_Crop(image_crop)
                            # print(image.size(),image1.size())

                            if use_all_three_modalities:
                                x = random.random()
                                if x < 0.33:
                                    return image1, image
                                elif x > 0.33 and x < 0.66:
                                    return image, image_crop_transform
                                else:
                                    return image, self.transform(image_orig)


                            if random.random() > 0.5:
                                return image1, image
                            else:
                                return image, image_crop_transform

                        # if len(image_list)==2:
                        #     if random.random() > 0.5:
                        #         return image_list[0],  image_list[1]
                        #     else:
                        #         return image_list[1],  image_list[0]
                    # number_of_points = 200
                    # radius = np.random.randint(self.radius[0], self.radius[1])
                    # crop_size = 224
                    # image_list = []
                    # for x in range(0, number_of_points + 1):
                    #     # self.radius[0]
                    #     width1_new_point = math.cos(
                    #         2 * pi / number_of_points * random.randint(0, number_of_points)) * radius
                    #     height1_new_point = math.sin(
                    #         2 * pi / number_of_points * random.randint(0, number_of_points)) * radius
                    #     width1_new_point_final = width1_new_point + width1
                    #     height1_new_point_final = height1_new_point + height1
                    #     width2_new_point = width1_new_point_final + crop_size
                    #     height2_new_point = height1_new_point_final + crop_size
                    #     # pdb.set_trace()
                    #     if self.is_safe(width1_new_point_final, height1_new_point_final, width2_new_point,
                    #                     height2_new_point, width_crop, height_crop):
                    #         # print("here")
                    #         # pdb.set_trace()
                    #
                    #         crop_box = (
                    #             int(width1_new_point_final), int(height1_new_point_final),
                    #             int(width2_new_point),
                    #             int(height2_new_point))
                    #
                    #         img1 = image_orig.crop((crop_box))
                    #         image = self.transform_single_Crop(img1)
                    #         image1 = self.transform_single_Crop(image_crop)
                    #         # print(image.size(),image1.size())
                    #         if random.random() > 0.5:
                    #             return image1, image
                    #         else:
                    #             return image, image1
                    #         # crop_box = (width1, height1, width2, height2)
                    #         # img2 = image_orig.crop((crop_box))
                    #         # print(img1.size,img2.size)
                    #         # return self.transform_single_Crop(img1), self.transform_single_Crop(img2)
                    #     width2_new_point = width1 - width1_new_point
                    #     height2_new_point = height1 - height1_new_point
                    #     width2_new_point_2 = width2_new_point - crop_size
                    #     height2_new_point_2 = height2_new_point - crop_size
                    #     if self.is_safe(width2_new_point, height2_new_point, width2_new_point_2,
                    #                     height2_new_point_2,
                    #                     width_crop, height_crop):
                    #         # print("here2")
                    #         crop_box = (
                    #         int(width2_new_point_2), int(height2_new_point_2), int(width2_new_point),
                    #         int(height2_new_point))
                    #         # print(crop_box)
                    #
                    #         img1 = image_orig.crop((crop_box))
                    #         crop_box = (width1, height1, width2, height2)
                    #         # img2 = image_orig.crop((crop_box))
                    #         # print(img1.size, img2.size)
                    #         image2 = self.transform_single_Crop(img1)
                    #         # image1 = self.transform_single_Crop(image_crop)
                    #         # if random.random() > 0.5:
                    #         #     return image1, image
                    #         # else:
                    #         #     return image, image1
                    #         if random.random() > 0.5:
                    #             return image1, image2
                    #         else:
                    #             return image2, image1
                            # return self.transform_single_Crop(img1), self.transform_single_Crop(img2)


                else:
                    image_crop = image_orig.crop((crop_box))


            else:
                crop_box = (int(float(float(self.bbox[item][4]) * width)), int(float(self.bbox[item][6]) * height),
                            int((float(self.bbox[item][5]) * width)),
                            int(float(self.bbox[item][7]) * height))
                image_orig = image_orig.crop((crop_box))
                width, height = image_orig.size

        if width==0 or height==0 or width_crop==0 or height_crop==0:
            item=100
            print('in 100')
            image_orig = Image.open(os.path.join(DATAPATH, self.images[item])).convert('RGB')  # (C, H, W)
            width, height = image_orig.size
            crop_box = (int(float(float(self.bbox[item][4]) * width)), int(float(self.bbox[item][6]) * height),
                        int((float(self.bbox[item][5]) * width)),
                        int(float(self.bbox[item][7]) * height))
            image_crop = image_orig.crop((crop_box))
            # image_orig = image_orig.crop((crop_box))
        labes_one_hot = [0] * len(self.labels_to_idx)

        if self.phase == 'train':
            # image = self.transform(image_orig)
            # image1 = self.transform(image_orig)
            # print(image.shape)
            # print(image1.shape)
            # pdb.set_trace()
            # print(image_orig.size)
            # print(self.transform_random)
            if self.rescale_crops_before:
                image =  self.transform_random(image_orig)
                # print(image_after_random.size)
                # image = self.transform(image_after_random)
                image1 = self.transform_random(image_orig)
                # image1 = self.transform(image_after_random1)

                # image.save()
                # image1.save('image_after_random_2.png')
                # image_orig.save('image_original_1.png')
                # from torchvision.utils import save_image
                # save_image(image, 'image_after_random_1.png')
                # save_image(image1, 'image_after_random_2.png')
                # save_image(image_orig, 'image_original_1.png')
                # image_orig.save('image_original_1.png')
                # image1 = self.transform(image_orig)
                # print("here")
                #
                return image, image1

            else:
                image = self.transform(image_orig)
                image1 = self.transform_single_Crop(image_crop)
                # print(image.size(),image1.size())
                if random.random() > 0.5:
                    return image1, image
                else:
                    return image,image1
        if self.phase == 'train_sup':
            for l in self.labels[item]:
                labes_one_hot[self.labels_to_idx[l]] = 1
            labes_one_hot = torch.tensor(labes_one_hot, dtype=torch.float32)
            image = self.train_transform(image_orig)
            # print(image.shape)
            # print(self.labels)
            # torch.tensor(self.labels[item].tolist(), dtype=torch.float32)
            return image, labes_one_hot

        # return image and label
        #     return image, self.labels[item]
        #     return image, image_label[image_id] - 1  # count begin from zero

        else:
            image = self.test_trasform(image_orig)
            for l in self.labels_test[item]:
                labes_one_hot[self.labels_to_idx[l]] = 1
            labes_one_hot = torch.tensor(labes_one_hot, dtype=torch.float32)

            return image,labes_one_hot

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
