import os
import cv2
dir = '/home/maria/TFM/pepinos/annotations'
files_list = os.listdir(dir)
for file in files_list:
    if file.endswith(".xml"):
        #im = cv2.imread(os.path.join(dir,file), 1)
        ext = file.split('pepino')
        n,_ = ext[1].split('.')
        new_name = 'cucumber_'+n+'.xml'
        # print new_name
        os.rename(os.path.join(dir,file), os.path.join(dir,new_name))
        # cv2.imwrite(new_name, im)
        # cv2.imshow('image', im)
        # cv2.waitKey(0)