# object-cropping-ssl
This repo contains the code for the paper "Object-cropping for SSL".

Download the five dicts from this folder and put them in the main repo.
https://drive.google.com/drive/folders/1IZTZDqcdSbjOFT1rBBGLLdjhZv5ZmthM?usp=sharing

Data creation: 
mkdir openimages
cd openimages
wget https://datasets.appen.com/appen_datasets/open-images/zip_files_copy/train_00.zip
wget https://datasets.appen.com/appen_datasets/open-images/zip_files_copy/train_01.zip
wget https://datasets.appen.com/appen_datasets/open-images/zip_files_copy/train_02.zip
wget https://datasets.appen.com/appen_datasets/open-images/zip_files_copy/train_03.zip
wget https://datasets.appen.com/appen_datasets/open-images/zip_files_copy/train_04.zip
wget https://datasets.appen.com/appen_datasets/open-images/zip_files_copy/train_05.zip
wget https://datasets.appen.com/appen_datasets/open-images/zip_files_copy/train_06.zip
wget https://datasets.appen.com/appen_datasets/open-images/zip_files_copy/train_07.zip
wget https://datasets.appen.com/appen_datasets/open-images/zip_files_copy/train_08.zip
unzip train_00.zip
rm -rf train_00.zip
unzip train_01.zip
rm -rf train_01.zip
unzip train_02.zip
rm -rf train_02.zip
unzip train_03.zip
rm -rf train_03.zip
unzip train_04.zip
rm -rf train_04.zip
unzip train_05.zip
rm -rf train_05.zip
unzip train_06.zip
rm -rf train_06.zip
unzip train_07.zip
rm -rf train_07.zip
unzip train_08.zip
rm -rf train_08.zip
wget https://datasets.appen.com/appen_datasets/open-images/zip_files_copy/validation.zip
wget https://datasets.appen.com/appen_datasets/open-images/zip_files_copy/test.zip
unzip validation.zip
unzip test.zip
rm -rf validation.zip
rm -rf test.zip
mkdir all_images
mv train_00/* all_images
mv train_01/* all_images
mv train_02/* all_images
mv train_03/* all_images
mv train_04/* all_images
mv train_05/* all_images
mv train_06/* all_images
mv train_07/* all_images
mv train_08/* all_images

update in datasets/init.py DATAPATH = 'path to all images'
