import os
import scipy
import scipy.io
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle

# ImageNet type datasets, i.e., which support loading with ImageFolder
def imagenette(datadir="/data/data_vvikash/datasets/imagenet_subsets/imagenette", batch_size=128, mode="org", size=224, normalize=False, norm_layer=None, workers=4, distributed=False, **kwargs):
    # mode: base | org
    
    if norm_layer is None:
        if normalize:
            norm_layer = transforms.Normalize(mean=[0.4648, 0.4543, 0.4247], std=[0.2785, 0.2735, 0.2944])
        else:
            norm_layer = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
    
    transform_train = transforms.Compose([transforms.RandomResizedCrop(size),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       norm_layer
                       ])
    transform_test = transforms.Compose([transforms.Resize(int(1.14*size)),
                      transforms.CenterCrop(size),
                      transforms.ToTensor(), 
                      norm_layer])
    
    if mode == "org":
        None
    elif mode == "base":
        transform_train = transform_test
    else:
        raise ValueError(f"{mode} mode not supported")
        
    trainset = datasets.ImageFolder(
        os.path.join(datadir, "train"), 
        transform=transform_train)
    testset = datasets.ImageFolder(
        os.path.join(datadir, "val"), 
        transform=transform_test)
    
    train_sampler, test_sampler = None, None
    if distributed:
        print("Using DistributedSampler")
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
        
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=workers, pin_memory=True)
    
    return train_loader, train_sampler, test_loader, test_sampler, None, None, transform_train


def cifar10(datadir="/data/data_vvikash/datasets/", batch_size=128, mode="org", size=32, normalize=False, norm_layer=None, workers=4, distributed=False, **kwargs):
    # mode: base | org
    if norm_layer is None:
        if normalize:
            norm_layer = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
        else:
            norm_layer = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
    
    trtrain = [transforms.RandomCrop(size, padding=4), transforms.RandomHorizontalFlip(), 
          transforms.ToTensor(), norm_layer]
    if size != 32:
        trtrain = [transforms.Resize(size)] + trtrain
    transform_train = transforms.Compose(trtrain)
    trval = [transforms.ToTensor(), norm_layer]
    if size != 32:
        trval = [transforms.Resize(size)] + trval
    transform_test = transforms.Compose(trval)

    if mode == "org":
        None
    elif mode == "base":
        transform_train = transform_test
    else:
        raise ValueError(f"{mode} mode not supported")
        
    trainset = datasets.CIFAR10(
            root=os.path.join(datadir, "cifar10"),
            train=True,
            download=True,
            transform=transform_train
        )
    testset = datasets.CIFAR10(
            root=os.path.join(datadir, "cifar10"),
            train=False,
            download=True,
            transform=transform_test,
        )
    
    train_sampler, test_sampler = None, None
    if distributed:
        print("Using DistributedSampler")
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
        
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=workers, pin_memory=True)
    
    return train_loader, train_sampler, test_loader, test_sampler, None, None, transform_train


# Oxford-Flowers
class custom_dataset(Dataset):
    def __init__(self, indexes, labels, root_dir, transform=None):
        self.images = []
        self.targets = []
        self.transform = transform

        for i in indexes:
            self.images.append(
                os.path.join(
                    root_dir,
                    "jpg",
                    "image_" + "".join(["0"] * (5 - len(str(i)))) + str(i) + ".jpg",
                )
            )
            self.targets.append(labels[i - 1] - 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        target = self.targets[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, target
    

def flowers(datadir="/data/nvme/datasets/oxford_102_flowers/", batch_size=128, mode="org", size=224, normalize=False, norm_layer=None, workers=4, distributed=False, **kwargs):
    # mode: base | org
    
    if norm_layer is None:
        if normalize:
            norm_layer = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
        else:
            norm_layer = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
    
    transform_train = transforms.Compose([transforms.Resize((int(1.14*size), int(1.14*size))),
                                          transforms.RandomCrop((size, size)),
                                          transforms.ColorJitter(0.4, 0.4, 0.4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.5, 1.5)),
                                          norm_layer
                                         ])
    transform_test = transforms.Compose([transforms.Resize((size, size)),
                                         transforms.ToTensor(), 
                                         norm_layer])
    
    if mode == "org":
        None
    elif mode == "base":
        transform_train = transform_test
    else:
        raise ValueError(f"{mode} mode not supported")
    
    labels = scipy.io.loadmat(os.path.join(datadir, "imagelabels.mat"))["labels"][0]
    splits = scipy.io.loadmat(os.path.join(datadir, "setid.mat"))
    
    trainset = custom_dataset(np.concatenate((splits["trnid"][0], splits["valid"][0]), axis=0),
                              labels, datadir, transform_train)
    testset = custom_dataset(splits["tstid"][0], labels, datadir, transform_test)
    
    train_sampler, test_sampler = None, None
    if distributed:
        print("Using DistributedSampler")
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
        
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=workers, pin_memory=True)
    
    return train_loader, train_sampler, test_loader, test_sampler, None, None, transform_train


def stcars(datadir="/data/data_vvikash/datasets/stanford_cars", batch_size=128, mode="org", size=224, normalize=False, norm_layer=None, workers=4, distributed=False, **kwargs):
    # mode: base | org
    
    if norm_layer is None:
        if normalize:
            norm_layer = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
        else:
            norm_layer = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
    
    transform_train = transforms.Compose([transforms.Resize((int(1.14*size), int(1.14*size))),
                                          transforms.RandomCrop((size, size)),
                                          transforms.ColorJitter(0.4, 0.4, 0.4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.5, 1.5)),
                                          norm_layer
                                         ])
    transform_test = transforms.Compose([transforms.Resize((size, size)),
                                         transforms.ToTensor(), 
                                         norm_layer])
    
    if mode == "org":
        None
    elif mode == "base":
        transform_train = transform_test
    else:
        raise ValueError(f"{mode} mode not supported")

    trainset = datasets.ImageFolder(
        os.path.join(datadir, "train"), 
        transform=transform_train)
    testset = datasets.ImageFolder(
        os.path.join(datadir, "test"), 
        transform=transform_test)
    
    train_sampler, test_sampler = None, None
    if distributed:
        print("Using DistributedSampler")
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
        
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=workers, pin_memory=True)
    
    return train_loader, train_sampler, test_loader, test_sampler, None, None, transform_train

