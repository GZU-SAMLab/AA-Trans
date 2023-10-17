import scipy.io as scio
import collections
import math
import os
import shutil


def copyfile(filename, target_dir):
    """将文件复制到目标目录。"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)


data_dir = '/home/samuel/datasets/Stanford_Dogs/'
train_dir = '/home/samuel/datasets/Stanford_Dogs/train_list.mat'
test_dir = '/home/samuel/datasets/Stanford_Dogs/test_list.mat'

train_list = scio.loadmat(train_dir)
test_list = scio.loadmat(test_dir)

# print(train_list['file_list'])
# print(train_list['labels'])

train_data = train_list['file_list']
test_data = test_list['file_list']
train_label = train_list['labels']
test_label = test_list['labels']

# for i in range(len(train_data)):
#     pic = train_data[i][0].tolist()[0]
#     label = train_label[i][0].tolist() - 1
#     label = str(label)
#     # print(pic)
#     # print(label)
#     line = pic + ' ' + label + '\n'
#     train_file = open('/home/samuel/datasets/Stanford_Dogs/train.txt', 'a')
#     train_file.write(line)

for j in range(len(test_data)):
    pic = test_data[j][0].tolist()[0]
    label = test_label[j][0].tolist() - 1
    label = str(label)
    # print(pic)
    # print(label)
    line = pic + ' ' + label + '\n'
    test_file = open('/home/samuel/datasets/Stanford_Dogs/val.txt', 'a')
    test_file.write(line)

# print(train_data[0][0].tolist()[0])
#
# for i in range(len(train_data)):
#     pic = train_data[i][0].tolist()[0]
#     root = os.path.join(data_dir, 'Images', pic)
#     label, img = pic.split('/')
#     # print(label)
#     copyfile(root, os.path.join(data_dir, 'train', label))
#
# for j in range(len(test_data)):
#     pic = test_data[j][0].tolist()[0]
#     root = os.path.join(data_dir, 'Images', pic)
#     label, img = pic.split('/')
#     # print(root)
#     copyfile(root, os.path.join(data_dir, 'val', label))
