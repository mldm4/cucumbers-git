import os
import cv2
# dir = '/home/maria/TFM/pepinos/annotations'
# files_list = os.listdir(dir)
# for file in files_list:
#     if file.endswith(".xml"):
#         #im = cv2.imread(os.path.join(dir,file), 1)
#         ext = file.split('pepino')
#         n,_ = ext[1].split('.')
#         new_name = 'cucumber_'+n+'.xml'
#         # print new_name
#         os.rename(os.path.join(dir,file), os.path.join(dir,new_name))
#         # cv2.imwrite(new_name, im)
#         # cv2.imshow('image', im)
#         # cv2.waitKey(0)

# dir = 'images/pepino'
# folders = os.listdir(dir)
# n = 0
# for folder in folders:
#     files = os.listdir(os.path.join(dir,folder))
#     for file in files:
#         if file.endswith(".bmp"):
#             n_format = "{:03}".format(n)
#             new_name = 'cucumber_belt_'+n_format+'.bmp'
#             # print (new_name)
#             n+=1
#             os.rename(os.path.join(dir,folder,file), os.path.join(dir,new_name))

#change format bmp to jpg for belt set
dir = 'images/belt_train_tiny_set'
files = os.listdir(dir)
for file in files:
    if file.endswith(".bmp"):
        im = cv2.imread(os.path.join(dir,file))
        # cv2.imshow('s',im)
        # cv2.waitKey()
        new_name = file.split('.bmp')[0]+'.jpg'
        cv2.imwrite(os.path.join(dir,new_name), im)

#change image name in xml
# import xml.etree.ElementTree as ET
# dir = 'annotations/belt_val_tiny_set/xmls'
# files = os.listdir(dir)
# for file in files:
#     tree = ET.parse(os.path.join(dir, file))
#     root = tree.getroot()
#
#     #change folder
#     root[0].text = 'belt_val_tiny_set'
#
#     #change filename
#     name = root[1].text
#     new_name = name.split('.bmp')[0] + '.jpg'
#     root[1].text = new_name
#
#     #change path
#     name = root[2].text
#     new_name = name.split('.bmp')[0] + '.jpg'
#     root[2].text = new_name
#
#     name = root[2].text
#     new_name = name.split('_train_')[0] + '_val_tiny_' + name.split('_train_')[1]
#     root[2].text = new_name
#
#     tree.write(os.path.join(dir,file))