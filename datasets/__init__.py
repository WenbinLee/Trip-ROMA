import torch
import torchvision

def get_dataset(dataset, data_dir, transform, train=True, train_classifier=False, download=False):
    if dataset == 'stl10':
        if train_classifier:
            split = 'train'
        else:
            split = 'train+unlabeled'
        dataset = torchvision.datasets.STL10(data_dir, split=split if train else 'test', transform=transform, download=download)
    elif dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'imagenet':
        dataset = torchvision.datasets.ImageNet(data_dir, split='train' if train == True else 'val', transform=transform, download=download)
    else:
        raise NotImplementedError

    return dataset