import collections
import math
import os
import shutil

import pandas as pd


def copyfile(filename, target_dir):
    """将文件复制到目标目录。"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)


def reorg_train_valid(data_dir, labels, labels_counts, valid_ratio):
    # # 训练数据集中示例最少的类别中的示例数
    # n = collections.Counter(labels.values()).most_common()[-1][1]
    #
    # print(n)
    # # 验证集中每个类别的示例数
    # n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train_images')):
        label = labels[train_file]
        count = labels_counts[label]
        n_valid_per_label = math.floor(count * valid_ratio)
        fname = os.path.join(data_dir, 'train_images', train_file)

        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir,
                                         'test', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir,
                                         'train', label))
    return n_valid_per_label


def read_txt_labels(fname):
    """读取 `fname` 来给标签字典返回一个文件名。"""
    with open(fname, 'r') as f:
        # 跳过文件头行 (列名)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split() for l in lines]
    return list((name, label) for name, label in tokens)


data_dir = '/home/samuel/datasets/CUB_200_2011/'

train_dir = '/home/samuel/datasets/CUB_200_2011/train.txt'
test_dir = '/home/samuel/datasets/CUB_200_2011/val.txt'
labels1 = read_txt_labels(train_dir)
labels2 = read_txt_labels(test_dir)

for label in labels1:
    fname, img = label[0].split('/')
    root = os.path.join(data_dir, 'images', fname, img)
    copyfile(root, os.path.join(data_dir, 'train', fname))

for label in labels2:
    fname, img = label[0].split('/')
    root = os.path.join(data_dir, 'images', fname, img)
    copyfile(root, os.path.join(data_dir, 'val', fname))


# data_csv = pd.read_csv('data/train.csv', header=0)
# labels_counts = data_csv['labels'].value_counts()
# print(labels_counts)
#
# valid_ratio = 0.3
# reorg_train_valid('./data/', labels, labels_counts, valid_ratio)
