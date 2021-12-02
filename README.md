# object-cropping-ssl
This repo contains the code for the paper "Object-cropping for SSL".


Step1: Creating the dataset.

Download the five dicts from this folder and put them in the main repo.
https://drive.google.com/drive/folders/1IZTZDqcdSbjOFT1rBBGLLdjhZv5ZmthM?usp=sharing


Follow create_data.sh to download openimages dataset.

Commands to run:

python main_moco_openimages.py path_to_openimages_dowloaded_dataset -a resnet50 --lr 0.015 --batch-size 128 --dist-url 'tcp://localhost:10005' --world-size 1 --rank 0  -j 8 --moco-t 0.2 --mlp --aug-plus --cos --save_name openimages_bing_crops_temperature_0.2 --multiprocessing-distributed --bing_crops
