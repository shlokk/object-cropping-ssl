This repo contains the pytorch implementation of the paper: **Object-Aware Cropping for Self-Supervised Learning (https://openreview.net/forum?id=WXgJN7A69g).**


<p align="center">
  <img src="teasure_figure_tmlr.png" width="650" title="hover text">
</p>

```
@article{
mishra2022objectaware,
title={Object-aware Cropping for Self-Supervised Learning},
author={Shlok Kumar Mishra and Anshul Shah and Ankan Bansal and Janit K Anjaria and Abhyuday Narayan Jagannatha and Abhishek Sharma and David Jacobs and Dilip Krishnan},
journal={Transactions on Machine Learning Research},
year={2022},
url={https://openreview.net/forum?id=WXgJN7A69g},
note={}
}
```
### Preparation
**Creating the OpenImages Hard Multi-Object Subset (OHMS) dataset.**

Download the five dicts from this folder and put them in the main repo.
https://drive.google.com/drive/folders/1IZTZDqcdSbjOFT1rBBGLLdjhZv5ZmthM?usp=sharing

Follow create_data.sh to download openimages dataset.

**Details of the OHMS Subset dataset:**

OHMS subset dataset was created to test the performace of SSL methods on more realistic and harder setup as compared to standard bechmarks such as ImageNet. We sample images with objects from at least 2 distinct classes to create a dataset that better reflects real-world uncurated data. Secondly we only consider class categories with at least 900 images to mitigate effects of imbalanced class distribution. After this processing, we have 212,753 images present across 208 classes and approximately 12 objects per image on average. Further details can be found in the paper.

We provide labels and other information of these images in images_selected_new.npy and images_selected_with_all_features.npy dicts.

**Desciption of the dicts:**

   1) images_selected_new.npy can be loaded as images_selected = np.load('images_selected_new.npy', allow_pickle='TRUE').item().\
      a) The keys in this dict is the image name and the values are the classes.  
      b) This dict can be used for multi-class classification task on OpenImages subset. \
      For ex. the if the key would be 0001cb734adac2ee (image name) and the values would be ['/m/0284d', '/m/040b_t', '/m/0cxn2'] ( class names ). 
      
   2) images_selected_with_all_features.npy contains the information for each object in the image and label. \
      a) images_selected_with_all_features.npy can be loaded as images_selected_with_all_features = np.load('images_selected_with_all_features.npy',      allow_pickle='TRUE').item(). \
      b) The key again would be the image name and values woule be list of all the objects. In every element , element[4] would be x1 poistion ,   element[5] would be x2 poistion,  element[6] would be y1 poistion and element[7] would be y2 poistion.


### State-of-the model performances on OHMS subset.

   | Model             | OpenImages (mAP)   | ImageNet (Top-1 %) |
   |---------------------------|--------|----------|
   | Supervised Performance         | 66.3 | 76.2  |
   | CMC(Tian et al., 2019) | 48.7 (-17.6) | 60.0 (-16.2)  |
   | BYOL(Grill et al., 2020) | 50.2 (-16.1) | 70.7 (-5.5)  |
   | SwAV(Caron et al., 2020) | 51.3 (-15.0) | 72.7 (-3.5)  |
   | MoCo-v2 | 49.8 (-16.5) | 67.5 (-8.7)  |
   | MoCo-v2 (Object-Object+Dilate crop) (Ours) | 58.6 (-7.7) | 68.0 (-8.2)  |



###  Pre-Training

Command to reproduce object aware cropping performance:

```
python main_moco_openimages.py path_to_openimages_dowloaded_dataset -a resnet50 --lr 0.015 --batch-size 128 --dist-url 'tcp://localhost:10005' --world-size 1 --rank 0  -j 8 --moco-t 0.2 --mlp --aug-plus --cos --save_name folder_to_save --multiprocessing-distributed --bing_crops
```

Command to reproduce MoCo-v2 baseline performance:

```
python main_moco_openimages.py path_to_openimages_dowloaded_dataset -a resnet50 --lr 0.015 --batch-size 128 --dist-url 'tcp://localhost:10005' --world-size 1 --rank 0  -j 8 --moco-t 0.2 --mlp --aug-plus --cos --save_name folder_to_save --multiprocessing-distributed
```
###  Linear Probing

Command for linear probing performance:

```
python main_lincls.py -a resnet50 --lr 30.0 --batch-size 256 --epochs 100 --pretrained path_to_model --dist-url 'tcp://localhost:10005' --world-size 1 --rank 0  -j 8 --save_name folder_to_save --multiprocessing-distributed --DATAPATH path_to_data
```

## BibTeX

```
@article{
mishra2022objectaware,
title={Object-aware Cropping for Self-Supervised Learning},
author={Shlok Kumar Mishra and Anshul Shah and Ankan Bansal and Janit K Anjaria and Abhyuday Narayan Jagannatha and Abhishek Sharma and David Jacobs and Dilip Krishnan},
journal={Transactions on Machine Learning Research},
year={2022},
url={https://openreview.net/forum?id=WXgJN7A69g},
note={}
}
```
