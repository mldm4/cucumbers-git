#https://github.com/aleju/imgaug
from imgaug import augmenters as iaa
import os
import cv2
import argparse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int,
                    help='number of iterations')
parser.add_argument('--in_path', type=str,
                    help='input path')
parser.add_argument('--out_path', type=str,
                    help='output path')

args = parser.parse_args()

iterations = args.iterations
in_path = args.in_path
out_path = args.out_path

list = os.listdir(in_path)
images = []
for file in list:
    images.append(cv2.imread(os.path.join(in_path,file)))

for iter in range(iterations):
        seq = iaa.Sequential([iaa.GaussianBlur((0, 3.0)),
                              iaa.Affine(scale=(0.7,1.3),mode='reflect', rotate=(-45,45)),
                              iaa.Fliplr(0.5), iaa.Flipud(0.5)])

        images_aug = seq.augment_images(images)

        for image_aug, file in zip(images_aug, list):
            i = file.split('_')[1].split('.')[0]
            now = datetime.datetime.now()
            cv2.imwrite(out_path+'/cucumber_'+str(now.isoformat())+'.jpg', image_aug)

