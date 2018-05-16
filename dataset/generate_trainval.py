import os
import argparse
import random
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str,
                    help='input images path')
parser.add_argument('--out_path', type=str,
                    help='output file path')

args = parser.parse_args()

included_extensions = ['jpg', 'jpeg', 'png']
images_list = [fn for fn in os.listdir(args.in_path)
              if any(fn.endswith(ext) for ext in included_extensions)]

# Randomly split for a validation set
random.seed(42)
random.shuffle(images_list)
num_examples = len(images_list)
num_train = int(0.7 * num_examples)
train_list = images_list[:num_train]
val_list = images_list[num_train:]

train_f = open(os.path.join(args.out_path, 'trainset.txt'), 'w')
val_f = open(os.path.join(args.out_path, 'valset.txt'), 'w')

for image in train_list:
    name = image.split('.')[0]
    train_f.write(name+'\n')

for image in val_list:
    name = image.split('.')[0]
    val_f.write(name + '\n')

train_f.close()
val_f.close()
