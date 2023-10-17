import logging
from PIL import Image
import os

import torch

from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, BatchSampler

from .dataset import CUB, CarsDataset, NABirds, dogs, INat2017, IP102
from .autoaugment import AutoAugImageNetPolicy
import numpy as np

logger = logging.getLogger(__name__)


# class BalancedBatchSampler(BatchSampler):
#     def __init__(self, dataset, n_classes, n_samples):
#         # dataset这里指train_datasets,
#         # labels：即之前求得的tensor([  1,   1,   1,  ..., 200, 200, 200])
#
#         # 转化为list形式
#         self.labels_set = dataset.train_label
#
#         self.labels = torch.tensor(dataset.train_label)
#
#         # 这个操作大致是建立标签和序列对应的字典
#         self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}
#         # print(self.labels_set)
#         # print(self.labels)
#
#         for l in self.labels_set:
#             np.random.shuffle(self.label_to_indices[l])
#
#         self.used_label_indices_count = {label: 0 for label in self.labels_set}
#
#         self.count = 0
#         self.n_classes = n_classes
#         self.n_samples = n_samples
#         self.dataset = dataset
#
#         self.batch_size = self.n_samples * self.n_classes
#
#     def __iter__(self):
#         self.count = 0
#         while self.count + self.batch_size < len(self.dataset):
#             # np.random.choice(a, size=None, replace=True, p=None)
#             # 从a（只要是ndarray都可以，但必须是一维）中随机抽取数字，并组成指定大小（size）的数组
#             # replace:True表示可以取相同数字，False表示不可以取相同数字
#             # 数组p：与数组a相对，表示取数组a中每个元素的概率
#             classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
#             indices = []
#             for class_ in classes:
#                 indices.extend(self.label_to_indices[class_][
#                                self.used_label_indices_count[class_]:self.used_label_indices_count[
#                                                                          class_] + self.n_samples])
#                 self.used_label_indices_count[class_] += self.n_samples
#                 if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
#                     np.random.shuffle(self.label_to_indices[class_])
#                     self.used_label_indices_count[class_] = 0
#             yield indices
#             self.count += self.n_classes * self.n_samples
#
#     def __len__(self):
#         return len(self.dataset) // self.batch_size


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.dataset == 'CUB_200_2011':
        train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                              transforms.RandomCrop((448, 448)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                             transforms.CenterCrop((448, 448)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = CUB(root=args.data_root, is_train=True, transform=train_transform)
        testset = CUB(root=args.data_root, is_train=False, transform=test_transform)
    elif args.dataset == 'car':
        trainset = CarsDataset(os.path.join(args.data_root, 'devkit/cars_train_annos.mat'),
                               os.path.join(args.data_root, 'cars_train'),
                               os.path.join(args.data_root, 'devkit/cars_meta.mat'),
                               # cleaned=os.path.join(data_dir,'cleaned.dat'),
                               transform=transforms.Compose([
                                   transforms.Resize((600, 600), Image.BILINEAR),
                                   transforms.RandomCrop((448, 448)),
                                   transforms.RandomHorizontalFlip(),
                                   AutoAugImageNetPolicy(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                               )
        testset = CarsDataset(os.path.join(args.data_root, 'cars_test_annos_withlabels.mat'),
                              os.path.join(args.data_root, 'cars_test'),
                              os.path.join(args.data_root, 'devkit/cars_meta.mat'),
                              # cleaned=os.path.join(data_dir,'cleaned_test.dat'),
                              transform=transforms.Compose([
                                  transforms.Resize((600, 600), Image.BILINEAR),
                                  transforms.CenterCrop((448, 448)),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                              )
    elif args.dataset == 'dog':
        train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                              transforms.RandomCrop((448, 448)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                             transforms.CenterCrop((448, 448)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = dogs(root=args.data_root,
                        train=True,
                        cropped=False,
                        transform=train_transform,
                        download=False
                        )
        testset = dogs(root=args.data_root,
                       train=False,
                       cropped=False,
                       transform=test_transform,
                       download=False
                       )
    elif args.dataset == 'nabirds':
        train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                              transforms.RandomCrop((448, 448)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                             transforms.CenterCrop((448, 448)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = NABirds(root=args.data_root, train=True, transform=train_transform)
        testset = NABirds(root=args.data_root, train=False, transform=test_transform)
    elif args.dataset == 'INat2017':
        train_transform = transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                              transforms.RandomCrop((304, 304)),
                                              transforms.RandomHorizontalFlip(),
                                              AutoAugImageNetPolicy(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                             transforms.CenterCrop((304, 304)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = INat2017(args.data_root, 'train', train_transform)
        testset = INat2017(args.data_root, 'val', test_transform)
    elif args.dataset == 'ip102':
        train_transform = transforms.Compose([transforms.Resize((256, 256), Image.BILINEAR),
                                              transforms.RandomCrop((224, 224)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((256, 256), Image.BILINEAR),
                                             transforms.CenterCrop((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = IP102(root=args.data_root, is_train=True, transform=train_transform)
        testset = IP102(root=args.data_root, is_train=False, transform=test_transform)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=8,
                              drop_last=True,
                              pin_memory=True,
                              )
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=8,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader

