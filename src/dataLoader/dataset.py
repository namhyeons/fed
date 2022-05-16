from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataset(args):
    
    if args.dataset == 'cifar10':
        apply_transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2616)),
                                  ]
        )
        apply_transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2616)),
                                  ]
        )
        dir = '~/save/cifar10'
        print(dir)
        train_dataset = datasets.CIFAR10(dir, train=True, download=True,
                                         transform=apply_transform_train)
        test_dataset = datasets.CIFAR10(dir, train=False, download=True,
                                        transform=apply_transform_test)
        return train_dataset, test_dataset

    if args.dataset == 'cifar100':
        apply_transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761)),
                                  ]
        )
        apply_transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761)),
                                  ]
        )
        dir = '~/save/cifar100'
        train_dataset = datasets.CIFAR100(dir, train=True, download=True,
                                         transform=apply_transform_train)
        test_dataset = datasets.CIFAR100(dir, train=False, download=True,
                                        transform=apply_transform_test)

        return train_dataset, test_dataset

    if args.dataset == 'tiny-imagenet':
        apply_transform_train = transforms.Compose(
            [transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                std=[0.2302, 0.2265, 0.2262])
                                  ]
        )
        apply_transform_test = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                std=[0.2302, 0.2265, 0.2262])
                                  ]
        )
        dir = '~/FedSD/data/tiny-imagenet/'
        train_dataset = datasets.ImageFolder(dir+'train', transform=apply_transform_train)
        test_dataset = datasets.ImageFolder(dir+'test', transform=apply_transform_test)

        return train_dataset, test_dataset

   
