from .simsiam_aug import SimSiamTransform
from .eval_aug import Transform_single
from .simclr_aug import SimCLRTransform

imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]
cifar10_mean_std = [[0.4913, 0.4821, 0.4465],[0.2470, 0.2434, 0.2615]]
cifar100_mean_std = [[0.5071, 0.4867, 0.4408],[0.2675, 0.2565, 0.2761]]
stl10_mean_std = [[0.4914, 0.4822, 0.4465],[0.2471, 0.2435, 0.2616]]


def get_aug(name='simsiam', image_size=224, train=True, train_classifier=None, dataset_name="imagenet"):

    if dataset_name == "imagenet":
        mean_std = imagenet_mean_std
    elif dataset_name == "cifar10":
        mean_std = cifar10_mean_std
    elif dataset_name == "cifar100":
        mean_std = cifar100_mean_std
    elif dataset_name == "stl10":
        mean_std = stl10_mean_std
    else:
        print("Using imagenet mean std")
        mean_std = imagenet_mean_std
    
    if train==True:
        if name == 'simsiam' or name == 'simsiam_roma':
            augmentation = SimSiamTransform(image_size, mean_std)
        elif name == 'simclr' or name == 'simclr_roma':
            augmentation = SimCLRTransform(image_size, mean_std)
        elif name == 'trip' or name == 'trip_roma':
            augmentation = SimCLRTransform(image_size, mean_std)
        else:
            raise NotImplementedError
    elif train==False:
        if train_classifier is None:
            raise Exception
        augmentation = Transform_single(image_size, train_classifier, mean_std)
    else:
        raise Exception

    return augmentation