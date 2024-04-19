import glob
from torchvision.io import read_image, ImageReadMode

import torch
import torchvision
from torch.utils import data

import pickle
import numpy as np


def unpickle(file, codetype='bytes'):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding=codetype)
    return dict


class CifarData(data.Dataset):
    def __init__(self, dataset, labels, transform, device):
        self.device = device
        self.dataset = dataset
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx]).to(self.device), self.labels[idx].to(self.device)

    def __len__(self):
        return len(self.labels)


def load_cifar_10(batch_size, device=torch.device("cuda:0"), val=False):
    # 依次加载batch_data_i,并合并到x,y
    x, y = [], []
    for i in range(1, 6):
        batch_path = f'data/cifar-10-batches-py/data_batch_{i}'
        batch_dict = unpickle(batch_path)
        train_batch = batch_dict[b'data']
        train_label = np.array(batch_dict[b'labels'])
        x.append(train_batch)
        y.append(train_label)
    # 将5个训练样本batch合并为50000x3x32x32，标签合并为50000x1
    train_data = np.concatenate(x).reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    train_labels = torch.tensor(np.concatenate(y))

    # 分割训练集和验证集
    if val:
        val_num = int(0.1 * train_labels.shape[0])
        s = np.arange(train_labels.shape[0])
        np.random.shuffle(s)
        val_data = train_data[s[:val_num]]
        val_labels = train_labels[s[:val_num]]
        train_data = train_data[s[val_num:]]
        train_labels = train_labels[s[val_num:]]

    # 创建测试样本
    test_dict = unpickle('data/cifar-10-batches-py/test_batch')
    test_data = test_dict[b'data'].reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = torch.tensor(np.array(test_dict[b'labels']))

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomCrop(size=32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        # 标准化图像的每个通道
        torchvision.transforms.Normalize([0.491, 0.482, 0.447],
                                         [0.247, 0.243, 0.262])])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.491, 0.482, 0.447],
                                         [0.247, 0.243, 0.262])])

    if val:
        return data.DataLoader(CifarData(train_data, train_labels, transform_train, device), batch_size, shuffle=True), \
               data.DataLoader(CifarData(val_data, val_labels, transform_test, device), batch_size), \
               data.DataLoader(CifarData(test_data, test_labels, transform_test, device), batch_size)
    else:
        return data.DataLoader(CifarData(train_data, train_labels, transform_train, device), batch_size, shuffle=True), \
               data.DataLoader(CifarData(test_data, test_labels, transform_test, device), batch_size)


def load_cifar_100(batch_size, device=torch.device("cuda:0"), val=False):
    train_filepath = 'data/cifar-100-python/train'
    train_obj = unpickle(train_filepath, 'latin1')
    train_data = np.array(train_obj["data"]).reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    train_labels = torch.tensor(np.array(train_obj["fine_labels"]))

    # 分割训练集和验证集
    if val:
        val_num = int(0.1 * train_labels.shape[0])
        s = np.arange(train_labels.shape[0])
        np.random.shuffle(s)
        val_data = train_data[s[:val_num]]
        val_labels = train_labels[s[:val_num]]
        train_data = train_data[s[val_num:]]
        train_labels = train_labels[s[val_num:]]

    # 创建测试样本
    test_filepath = 'data/cifar-100-python/test'
    test_obj = unpickle(test_filepath, 'latin1')
    test_data = np.array(test_obj["data"].reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1))
    test_labels = torch.tensor(np.array(test_obj["fine_labels"]))

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomCrop(size=32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        # 标准化图像的每个通道
        torchvision.transforms.Normalize([0.507, 0.487, 0.441],
                                         [0.267, 0.256, 0.276])])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.507, 0.487, 0.441],
                                         [0.267, 0.256, 0.276])])

    if val:
        return data.DataLoader(CifarData(train_data, train_labels, transform_train, device), batch_size, shuffle=True), \
               data.DataLoader(CifarData(val_data, val_labels, transform_test, device), batch_size), \
               data.DataLoader(CifarData(test_data, test_labels, transform_test, device), batch_size)
    else:
        return data.DataLoader(CifarData(train_data, train_labels, transform_train, device), batch_size, shuffle=True), \
               data.DataLoader(CifarData(test_data, test_labels, transform_test, device), batch_size)


class TrainTinyImageNetDataset(data.Dataset):
    def __init__(self, id, device, transform=None):
        self.filenames = glob.glob("data/tiny-imagenet-200/train/*/*/*.JPEG")
        self.device = device
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
            image = read_image(img_path, ImageReadMode.RGB)
        label = torch.tensor(self.id_dict[img_path.split('/')[-3]])
        return self.transform(image).to(self.device), label.to(self.device)


class TestTinyImageNetDataset(data.Dataset):
    def __init__(self, id, device, transform=None):
        self.filenames = glob.glob("data/tiny-imagenet-200/val/images/*.JPEG")
        self.device = device
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open('data/tiny-imagenet-200/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0], a[1]
            self.cls_dic[img] = self.id_dict[cls_id]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
            image = read_image(img_path, ImageReadMode.RGB)
        label = torch.tensor(self.cls_dic[img_path.split('/')[-1]])
        return self.transform(image).to(self.device), label.to(self.device)


def load_tiny_imagenet(batch_size, device=torch.device("cuda:0")):
    id_dict = {}
    for i, line in enumerate(open('data/tiny-imagenet-200/wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomCrop(size=64, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        # 标准化图像的每个通道
        torchvision.transforms.Normalize([0.480, 0.448, 0.397],
                                         [0.276, 0.269, 0.282])])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.480, 0.448, 0.397],
                                         [0.276, 0.269, 0.282])])

    train_set = TrainTinyImageNetDataset(id=id_dict, device=device, transform=transform_train)
    test_set = TestTinyImageNetDataset(id=id_dict, device=device, transform=transform_test)

    return data.DataLoader(train_set, batch_size, shuffle=True), data.DataLoader(test_set, batch_size)


class PruneData(data.Dataset):
    def __init__(self, dataset, device):
        self.device = device
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx].to(self.device)

    def __len__(self):
        return len(self.dataset)


def load_prune_data(size, batch_size, device):
    return data.DataLoader(PruneData(torch.zeros(*size), device), batch_size)
