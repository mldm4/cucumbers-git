import imgaug as ia #https://github.com/aleju/imgaug
from imgaug import augmenters as iaa
import os
import cv2
import xml.etree.cElementTree as ElementTree
import argparse

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

im_path = os.path.join(in_path,'images')
ann_path = os.path.join(in_path,'annotations')

list = os.listdir(im_path)
images = []
for file in list:
    images.append(cv2.imread(os.path.join(im_path,file)))

# Generate random keypoints.
# The augmenters expect a list of imgaug.KeypointsOnImage.
for iter in range(iterations):
    for image, file in zip(images, list):
        keypoints_on_image = []
        height, width = image.shape[0:2]
        keypoints = []
        i = file.split('_')[1].split('.jpg')[0]
        tree = ElementTree.parse(os.path.join(ann_path, 'cucumber_'+i+'.xml'))
        root = tree.getroot()
        for object in root.iter('object'):
            #From the xmin, ymin, xmax, ymax, extrapolate the other two points in the bbox and store them
            #to be augmentated too (ur-upper right, bl-bottom left)
            xmin = int(object[4][0].text)
            ymin = int(object[4][1].text)
            xmax = int(object[4][2].text)
            ymax = int(object[4][3].text)
            xur = xmin
            yur = ymax
            xbl = xmax
            ybl = ymin
            keypoints.append(ia.Keypoint(x=xmin, y=ymin))
            keypoints.append(ia.Keypoint(x=xmax, y=ymax))
            keypoints.append(ia.Keypoint(x=xur, y=yur))
            keypoints.append(ia.Keypoint(x=xbl, y=ybl))


        keypoints_on_image.append(ia.KeypointsOnImage(keypoints, shape=image.shape))

        seq = iaa.Sequential([iaa.GaussianBlur((0, 3.0)),
                              iaa.Affine(scale=(0.7,1.3),mode='reflect', rotate=(-45,45)),
                              iaa.Fliplr(0.5), iaa.Flipud(0.5)])
        seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start

        # augment keypoints and images
        image_aug = seq_det.augment_image(image)
        keypoints_aug = seq_det.augment_keypoints(keypoints_on_image)

        for j, object in zip(range(0,len(keypoints_aug[0].keypoints),4), root.iter('object')):

            # TODO keypoints to pascal valid format:
            x1 = int(keypoints_aug[0].keypoints[j].x)
            y1 = int(keypoints_aug[0].keypoints[j].y)
            x2 = int(keypoints_aug[0].keypoints[j + 1].x)
            y2 = int(keypoints_aug[0].keypoints[j + 1].y)
            x3 = int(keypoints_aug[0].keypoints[j + 2].x)
            y3 = int(keypoints_aug[0].keypoints[j + 2].y)
            x4 = int(keypoints_aug[0].keypoints[j + 3].x)
            y4 = int(keypoints_aug[0].keypoints[j + 3].y)


            object[4][0].text = str(min([x1,x2,x3,x4])) #xmin
            object[4][1].text = str(min([y1,y2,y3,y4])) #ymin
            object[4][2].text = str(max([x1,x2,x3,x4])) #xmax
            object[4][3].text = str(max([x1,x2,x3,x4])) #ymax

        num = int(i) + (len(list) * (iter + 1))
        tree.write(out_path+'/annotations/cucumber_'+str(num)+'.xml')
        cv2.imwrite(out_path+'/images/cucumber_'+str(num)+'.jpg', image_aug)

    # Example code to show each image and print the new keypoints coordinates
    #     image_before = keypoints_on_image[0].draw_on_image(image)
    #     image_after = keypoints_aug[0].draw_on_image(image_aug)
    #     print(file)
    #     misc.imshow(np.concatenate((image_before, image_after), axis=1)) # before and after

