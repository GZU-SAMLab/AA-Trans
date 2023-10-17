import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from PIL import Image


def default_loader(path):
    try:
        img = Image.open(path).convert('RGB')
    except:
        with open('read_error.txt', 'a') as fid:
            fid.write(path + '\n')
        return Image.new('RGB', (224, 224), 'white')
    return img


class RandomDataset(Dataset):
    def __init__(self, transform=None, dataloader=default_loader):
        self.transform = transform
        self.dataloader = dataloader

        with open('/home/samuel/datasets/CUB_200_2011/val.txt', 'r') as fid:
            self.imglist = fid.readlines()

    def __getitem__(self, index):
        image_name, label = self.imglist[index].strip().split()
        image_path = '/home/samuel/datasets/CUB_200_2011/images/{}'.format(image_name)
        img = self.dataloader(image_path)
        img = self.transform(img)
        label = int(label)
        label = torch.LongTensor([label])

        return [img, label]

    def __len__(self):
        return len(self.imglist)


class BatchDataset(Dataset):
    def __init__(self, transform=None, dataloader=default_loader):
        self.transform = transform
        self.dataloader = dataloader

        # 打开train.txt， 读行
        with open('/home/samuel/datasets/CUB_200_2011/train.txt', 'r') as fid:
            self.imglist = fid.readlines()

        self.labels = []
        # 每行读路径和标签
        for line in self.imglist:
            image_path, label = line.strip().split()
            self.labels.append(int(label))
        self.labels = np.array(self.labels)
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, index):
        # 每行读路径和标签
        image_name, label = self.imglist[index].strip().split()
        image_path = '/home/samuel/datasets/CUB_200_2011/images/{}'.format(image_name)
        img = self.dataloader(image_path)
        img = self.transform(img)
        label = int(label)
        label = torch.LongTensor([label])

        return [img, label]

    def __len__(self):
        return len(self.imglist)


class BalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, n_classes, n_samples):
        # dataset这里指train_datasets,
        # labels：即之前求得的tensor([  1,   1,   1,  ..., 200, 200, 200])
        self.labels = dataset.labels
        # 转化为list形式
        self.labels_set = list(set(self.labels.numpy()))
        # 这个操作大致是建立标签和序列对应的字典
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        print(self.label_to_indices)

        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])

        self.used_label_indices_count = {label: 0 for label in self.labels_set}

        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        # self.batch_size = 2
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            # np.random.choice(a, size=None, replace=True, p=None)
            # 从a（只要是ndarray都可以，但必须是一维）中随机抽取数字，并组成指定大小（size）的数组
            # replace:True表示可以取相同数字，False表示不可以取相同数字
            # 数组p：与数组a相对，表示取数组a中每个元素的概率
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
