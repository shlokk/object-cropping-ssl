pip install aws-shell
mkdir openimages-dataset
cd openimages-dataset
mkdir all_images
aws s3 --no-sign-request sync s3://open-images-dataset/train all_images
mkdir val
aws s3 --no-sign-request sync s3://open-images-dataset/validation val
mkdir test
aws s3 --no-sign-request sync s3://open-images-dataset/test test
