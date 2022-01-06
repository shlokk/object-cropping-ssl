This repo contains the code code for paper: Object-Aware Cropping for Self-Supervised Learning (https://arxiv.org/pdf/2112.00319).


Creating the dataset.

Download the five dicts from this folder and put them in the main repo.
https://drive.google.com/drive/folders/1IZTZDqcdSbjOFT1rBBGLLdjhZv5ZmthM?usp=sharing

Follow create_data.sh to download openimages dataset.

Details of the OpenImages Subset dataset:
OpenImages subset dataset was created to test the performace of SSL methods on more realistic and harder setup as compared to standard bechmarks such as ImageNet. We sample images with objects from at least 2 distinct classes to create a dataset that better reflects real-world uncurated data. Secondly we only consider class categories with at least 900 images to mitigate effects of imbalanced class distribution. After this processing, we have 212,753 images present across 208 classes and approximately 12 objects per image on average. Further details can be found in the paper.

We provide labels and other information of these images in images_selected_new.npy and images_selected_with_all_features.npy dicts.
Desciption of the dicts:
1) images_selected_new.npy can be loaded as images_selected = np.load('images_selected_new.npy', allow_pickle='TRUE').item(). 
    * The keys in this dict is the image name and the values are the classes.  
    * This dict can be used for multi-class classification task on OpenImages subset.
      For ex. the if the key would be 0001cb734adac2ee (image name) and the values would be ['/m/0284d', '/m/040b_t', '/m/0cxn2'] ( class names ). 
2) images_selected_with_all_features.npy contains the information for each object in the image and label. 
    *  images_selected_with_all_features.npy can be loaded as images_selected_with_all_features = np.load('images_selected_with_all_features.npy',      allow_pickle='TRUE').item(). 
    *  The key again would be the image name and values woule be list of all the objects. In every element , element[4] would be x1 poistion ,   element[5] would be x2 poistion,  element[6] would be y1 poistion and element[7] would be y2 poistion.

| Model             | OpenImages (mAP)   | ImageNet (Top-1 %) |
|---------------------------|--------|----------|
| Supervised Performance         | 66.3 | 76.2  |
| CMC(Tian et al., 2019) | 48.7 (-17.6) | 60.0 (-16.2)  |
| BYOL(Grill et al., 2020) | 50.2 (-16.1) | 70.7 (-5.5)  |
| SwAV(Caron et al., 2020) | 51.3 (-15.0) | 72.7 (-3.5)  |
| MoCo-v2 | 49.8 (-16.5) | 67.5 (-8.7)  |
| MoCo-v2 (Object-Object+Dilate crop) (Ours) | 58.6 (-7.7) | 68.0 (-8.2)  |




Commands to run:

python main_moco_openimages.py path_to_openimages_dowloaded_dataset -a resnet50 --lr 0.015 --batch-size 128 --dist-url 'tcp://localhost:10005' --world-size 1 --rank 0  -j 8 --moco-t 0.2 --mlp --aug-plus --cos --save_name openimages_bing_crops_temperature_0.2 --multiprocessing-distributed --bing_crops

## BibTeX

```
@misc{mishra2021objectaware,
      title={Object-Aware Cropping for Self-Supervised Learning}, 
      author={Shlok Mishra and Anshul Shah and Ankan Bansal and Abhyuday Jagannatha and Abhishek Sharma and David Jacobs and Dilip Krishnan},
      year={2021},
      eprint={2112.00319},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
