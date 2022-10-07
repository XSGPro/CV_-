import os
import cv2
import numpy as np

from read_img import endwith

def read_file(path):
    img_list = []
    label_list = []
    dir_counter = 0
    IMG_SIZE = 128

    for dir_image in os.listdir(path):
        if endwith(dir_image, 'jpg'):
            img = cv2.imread(os.path.join(path, dir_image))
            resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            recolored_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            img_list.append(recolored_img)
            label_list.append(dir_image.split('_')[1].split('.')[0])
            dir_counter += 1

    img_list = np.array(img_list)

    return img_list, label_list, dir_counter

#读取训练数据集的文件夹，把他们的名字返回给一个list
def read_name_list(path):
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list



if __name__ == '__main__':
    img_list,label_lsit,counter = read_file('dataset')
    print(counter)
    print(label_lsit)


