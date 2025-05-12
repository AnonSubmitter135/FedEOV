import os
import torch
from PIL import Image
from PIL import ImageFilter
import numpy as np
import torch.utils.data as data
from torchvision import datasets, transforms
from datasets import load_dataset
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN, FashionMNIST
import os
import os.path
import logging
from PIL import Image, ImageOps, ImageFilter
from torchvision.transforms.functional import to_pil_image, to_tensor, adjust_brightness
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import random
from torch.utils.data import DataLoader
import json
from pathlib import Path
import time
from collections import defaultdict
import torchvision

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def accimage_loader(path):
    try:
        return Image.open(path).convert('RGB')
    except IOError:
        print(f"Error: Unable to load image at {path}")
        raise

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class MNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        mnist_dataobj = datasets.MNIST(self.root, train=self.train, transform=self.transform, target_transform=self.target_transform, download=self.download)

        data = mnist_dataobj.data.numpy()
        target = np.array(mnist_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):

        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class FashionMNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        fashion_mnist_dataobj = datasets.FashionMNIST(self.root, train=self.train, transform=self.transform, target_transform=self.target_transform, download=self.download)

        data = fashion_mnist_dataobj.data.numpy()
        target = np.array(fashion_mnist_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):

        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class ColoredMNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False, color_flip_prob=0.25, label_noise_prob=0.25):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.color_flip_prob = color_flip_prob
        self.label_noise_prob = label_noise_prob

        self.data, self.target = self.__build_colored_dataset__()

    def __build_colored_dataset__(self):
        mnist_dataobj = datasets.MNIST(
            self.root, train=self.train, download=self.download
        )

        data = mnist_dataobj.data.numpy()
        target = np.array(mnist_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        binary_labels = (target < 5).astype(np.float32)
        flipped_labels = np.random.rand(len(binary_labels)) < self.label_noise_prob
        binary_labels = np.abs(binary_labels - flipped_labels)

        colors = np.abs(binary_labels - (np.random.rand(len(binary_labels)) < self.color_flip_prob))

        colored_data = np.stack([data, data], axis=1)
        for i in range(len(colored_data)):
            colored_data[i, int(colors[i]) ^ 1, :, :] = 0

        return colored_data, binary_labels

    def __getitem__(self, index):

        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(torch.tensor(img, dtype=torch.float32))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        data = cifar_dataobj.data
        target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):

        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class SVHN_custom(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.train is True:
            svhn_dataobj = SVHN(self.root, 'train', self.transform, self.target_transform, self.download)
            data = svhn_dataobj.data
            target = svhn_dataobj.labels
        else:
            svhn_dataobj = SVHN(self.root, 'test', self.transform, self.target_transform, self.download)
            data = svhn_dataobj.data
            target = svhn_dataobj.labels

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target


    def __getitem__(self, index):

        img, target = self.data[index], self.target[index]

        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class CIFAR100_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):

        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0, net_id=None, total=0, apply_noise=False):
    if dataset == 'cifar10':
        dl_obj = CIFAR10_truncated

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    elif dataset == 'cifar100':
        dl_obj = CIFAR100_truncated

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=False)
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=False)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds

color_map = {
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
    'green': (0, 255, 0),
    'yellow': (255, 255, 0),
    'white': (255, 255, 255),
    'purple': (128, 0, 128),
    'black': (0, 0, 0),
    'gold': (255, 215, 0),
    'orange': (255, 165, 0),
    'pink': (255, 192, 203)
}

def apply_rotation(img, angle):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return img.rotate(angle)

def apply_brightness_shift(img, brightness_factor):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return adjust_brightness(img, brightness_factor)

def apply_scaling(img, scale):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return transforms.functional.affine(img, angle=0, translate=(0, 0), scale=scale, shear=0)

def apply_blur(img, radius):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_translation(img, x_offset, y_offset):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return transforms.functional.affine(img, angle=0, translate=(x_offset, y_offset), scale=1.0, shear=0)

def apply_shearing(img, shear_x, shear_y):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return transforms.functional.affine(img, angle=0, translate=(0, 0), scale=1.0, shear=(shear_x, shear_y))

def apply_gaussian_noise(img, std):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    noise = np.random.normal(0, std, (img.height, img.width, len(img.getbands())))
    img = np.clip(np.array(img) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


# Transformation maps
rotation_angles = [0, 20, 340, 40, 320, 60, 300, 80, 280, 90, 180]
brightness_factors = [0.3, 0.6, 1.0]
scaling_factors = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 1.9, 2.0, 2.2]
blur_radii = [0, 0.5, 0.75, 1.0, 1.25, 1.5, 3, 3.25, 3.5, 3.75, 4,2]
translation_offsets = [(0, 0), (2, 2), (4, 4), (6, 6), (8, 8), (10, 10), (-2, -2), (-4, -4), (-6, -6), (-8, -8), (-10, -10)]
shear_values = [(0, 0), (5, 0), (10, 0), (0, 5), (0, 10), (5, 5), (-5, -5), (-10, -10), (10, -10), (-10, 10), (15, 15)]
gaussian_noise_stds = [0, 5, 10, 15, 20, 25, 30, 35, 50, 45, 40]

def convert_to_colored(image, color_pair):
    if isinstance(image, Image.Image):
        image = np.array(image)

    color_1_rgb = color_map[color_pair[0]]
    color_2_rgb = color_map[color_pair[1]]
    colored_image = np.zeros((28, 28, 3), dtype=np.uint8)
    digit_mask = (image > 50)
    colored_image[digit_mask] = color_1_rgb
    colored_image[~digit_mask] = color_2_rgb
    return colored_image

def apply_domain_shift(img, shift_type, parameter):
    img = to_pil_image(img)

    if shift_type == 'rotation':
        img = apply_rotation(img, parameter)
    elif shift_type == 'brightness':
        img = apply_brightness_shift(img, parameter)
    elif shift_type == 'scaling':
        img = apply_scaling(img, parameter)
    elif shift_type == 'blur':
        img = apply_blur(img, parameter)
    elif shift_type == 'translation':
        img = apply_translation(img, parameter[0], parameter[1])
    elif shift_type == 'shearing':
        img = apply_shearing(img, parameter[0], parameter[1])
    elif shift_type == 'gaussian_noise':
        img = apply_gaussian_noise(img, parameter)

    return to_tensor(img)

def save_first_image_of_batch(image, client_id, batch_idx, save_dir):
    image = image.permute(1, 2, 0).cpu().numpy()
    image = np.uint8(image * 255)
    img_pil = Image.fromarray(image)
    filename = os.path.join(save_dir, f'client_{client_id}_batch_{batch_idx}.png')
    img_pil.save(filename)

def get_colored_mnist_loader2(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test, client_id, drop_last=False, color_change=True, rotation_change=True, brightness_change=True, scaling_change=True, blur_change=True, translation_change=True, shearing_change=True, gaussian_noise_change=True):
    # Define color pairs
    color_domains = [
        ('red', 'blue'), ('green', 'yellow'), ('white', 'purple'),
        ('red', 'green'), ('black', 'gold'), ('blue', 'red'),
        ('yellow', 'green'), ('purple', 'white'), ('green', 'red'), ('orange', 'black'),
        ('orange', 'pink'), 
    ]

    # Assign color pair and domain shift parameter
    color_pair = color_domains[client_id % len(color_domains)] if color_change else ('black', 'gold')

    # Define transformations
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] for grayscale images
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] for grayscale images
    ])

    # Load the MNIST dataset
    train_ds = MNIST(datadir, train=True, transform=transform_train, download=False)
    test_ds = MNIST(datadir, train=False, transform=transform_test, download=False)

    train_data = train_ds.data.numpy()[dataidxs_train]
    test_data = test_ds.data.numpy()[dataidxs_test]
    train_labels = np.array(train_ds.targets)[dataidxs_train]
    test_labels = np.array(test_ds.targets)[dataidxs_test]

    # Convert grayscale images to colored and apply domain shifts
    def convert_to_colored_loader(data, labels):
        colored_data = []
        for img in data:
            for domain_shift_type in ['rotation', 'brightness', 'scaling', 'blur', 'translation', 'shearing', 'gaussian_noise']:
                if domain_shift_type == 'rotation' and rotation_change:
                    domain_shift_param = rotation_angles[client_id % len(rotation_angles)]
                    img_colored = apply_rotation(img, domain_shift_param)
                    img_colored = convert_to_colored(img_colored, color_pair)
                elif domain_shift_type == 'brightness' and brightness_change:
                    domain_shift_param = brightness_factors[client_id % len(brightness_factors)]
                    img_colored = convert_to_colored(img, color_pair)
                    img_colored = apply_brightness_shift(img_colored, domain_shift_param)
                elif domain_shift_type == 'scaling' and scaling_change:
                    domain_shift_param = scaling_factors[client_id % len(scaling_factors)]
                    img_colored = apply_scaling(img, domain_shift_param)
                    img_colored = convert_to_colored(img_colored, color_pair)
                elif domain_shift_type == 'blur' and blur_change:
                    domain_shift_param = blur_radii[client_id % len(blur_radii)]
                    img_colored = convert_to_colored(img, color_pair)
                    img_colored = apply_blur(img_colored, domain_shift_param)
                elif domain_shift_type == 'translation' and translation_change:
                    domain_shift_param = translation_offsets[client_id % len(translation_offsets)]
                    img_colored = apply_translation(img, domain_shift_param[0], domain_shift_param[1])
                    img_colored = convert_to_colored(img_colored, color_pair)
                elif domain_shift_type == 'shearing' and shearing_change:
                    domain_shift_param = shear_values[client_id % len(shear_values)]
                    img_colored = apply_shearing(img, domain_shift_param[0], domain_shift_param[1])
                    img_colored = convert_to_colored(img_colored, color_pair)
                elif domain_shift_type == 'gaussian_noise' and gaussian_noise_change:
                    domain_shift_param = gaussian_noise_stds[client_id % len(gaussian_noise_stds)]
                    img_colored = convert_to_colored(img, color_pair)
                    img_colored = apply_gaussian_noise(img_colored, domain_shift_param)
                elif not any([rotation_change, brightness_change, scaling_change, blur_change, translation_change, shearing_change, gaussian_noise_change]):
                    img_colored = convert_to_colored(img, color_pair)
                    break
            img_tensor = to_tensor(img_colored)
            colored_data.append(img_tensor)

        colored_data = torch.stack(colored_data)  # Stack tensors
        return colored_data, torch.tensor(labels, dtype=torch.long)

    train_data_colored, train_labels_colored = convert_to_colored_loader(train_data, train_labels)
    test_data_colored, test_labels_colored = convert_to_colored_loader(test_data, test_labels)

    train_dataset = torch.utils.data.TensorDataset(train_data_colored, train_labels_colored)
    test_dataset = torch.utils.data.TensorDataset(test_data_colored, test_labels_colored)

    # Save directory for visualization
    save_dir = os.path.join(datadir, 'colored-mnist')
    os.makedirs(save_dir, exist_ok=True)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, drop_last=drop_last)
    test_loader = DataLoader(test_dataset, batch_size=test_bs, shuffle=False, drop_last=False)

    # Save sample images for each client
    for batch_idx, (images, labels) in enumerate(train_loader):
        save_first_image_of_batch(images[0], client_id, batch_idx, save_dir)

    return train_loader, test_loader


def get_divided_dataloader(dataset, datadir, train_bs, test_bs, 
                           dataidxs_train, dataidxs_test, noise_level=0, 
                           net_id=None, total=0, drop_last=False, apply_noise=False, domain=None):
    """
    Load dataset with correct transformations and apply domain-specific partitioning for PACS.
    """
    
    if dataset == 'cifar10':
        # CIFAR-10 Dataset
        dl_obj = CIFAR10_truncated
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_ds = dl_obj(datadir, dataidxs=dataidxs_train, transform=transform_train, download=False)
        test_ds = dl_obj(datadir, dataidxs=dataidxs_test, transform=transform_test, download=False)

    elif dataset == 'cifar100':
        # CIFAR-100 Dataset
        dl_obj = CIFAR100_truncated
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])

        train_ds = dl_obj(datadir, dataidxs=dataidxs_train, transform=transform_train, download=False)
        test_ds = dl_obj(datadir, dataidxs=dataidxs_test, transform=transform_test, download=False)

    elif dataset == 'fashion-mnist':
        # Fashion-MNIST Dataset (Grayscale)
        dl_obj = FashionMNIST_truncated
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_ds = dl_obj(datadir, dataidxs=dataidxs_train, transform=transform_train, download=False)
        test_ds = dl_obj(datadir, dataidxs=dataidxs_test, transform=transform_test, download=False)

    elif dataset == 'mnist':
        # MNIST Dataset (Grayscale)
        dl_obj = MNIST_truncated
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_ds = dl_obj(datadir, dataidxs=dataidxs_train, transform=transform_train, download=False)
        test_ds = dl_obj(datadir, dataidxs=dataidxs_test, transform=transform_test, download=False)

    elif dataset == 'svhn':
        # SVHN Dataset
        dl_obj = SVHN_custom
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_ds = dl_obj(datadir, dataidxs=dataidxs_train, transform=transform_train, download=False)
        test_ds = dl_obj(datadir, dataidxs=dataidxs_test, transform=transform_test, download=False)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Create Data Loaders
    train_dl = DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=drop_last)
    test_dl = DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds


def tensor_to_list(d):
    """
    Recursively converts torch.Tensor and numpy.ndarray objects in a dictionary or list to lists.
    """
    if isinstance(d, dict):
        # Recursively apply the conversion to dictionary values
        return {k: tensor_to_list(v) for k, v in d.items()}
    elif isinstance(d, list):
        # Recursively apply the conversion to list elements
        return [tensor_to_list(i) for i in d]
    elif isinstance(d, torch.Tensor):
        # Convert torch.Tensor to a Python list
        return d.detach().cpu().tolist()
    elif isinstance(d, np.ndarray):
        # Convert numpy.ndarray to a Python list
        return d.tolist()
    else:
        # If it's not a tensor or ndarray, return the object as-is
        return d

def save_results(args, global_model, results_dict, similarity_dict=None, properties_dict=None, activations_dict=None, net=None):

    top_level_dir = Path("results_table") / args.dataset / f"seed_{args.init_seed}" / f"{args.model}_{args.classifier_type}_{args.loss_weighting}"  / f"dirichlet_{args.beta}" / f"CR_{args.comm_round}" / f"LE_{args.epochs}" / f"clients_{args.n_parties}" / f"select_{args.sample}" / args.alg
    top_level_dir.mkdir(parents=True, exist_ok=True)

    if args.save_model:
        outfile_gmodel = top_level_dir / 'gmodel_1500.tar'
        torch.save({'epoch': args.comm_round + 1, 'state': global_model.state_dict()}, outfile_gmodel)

    if results_dict:
        json_file_opt = f"{args.alg}_results_{args.comment}.json"
        with open(top_level_dir / json_file_opt, "w") as file:
            json.dump(results_dict, file, indent=4)
        print(f"Results saved to {top_level_dir}")

    # Convert activations to a JSON-serializable format
    if activations_dict:
        activations_dict = tensor_to_list(activations_dict)
        activations_file = top_level_dir / "activations.json"
        with open(activations_file, "w") as file:
            json.dump(activations_dict, file, indent=4)
        print(f"Activations saved to {top_level_dir}")

    # Save similarity data
    if similarity_dict:
        similarity_file = top_level_dir / "similarity.json"
        with open(similarity_file, "w") as file:
            json.dump(similarity_dict, file, indent=4)
        print(f"Similarity data saved to {top_level_dir}")

    # Save properties data
    if properties_dict:
        properties_file = top_level_dir / "properties.json"
        with open(properties_file, "w") as file:
            json.dump(properties_dict, file, indent=4)
        print(f"Properties data saved to {top_level_dir}")

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def load_mnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    print("Loading MNIST data...")
    print(f"datadir: {datadir}")
    # Create the truncated dataset objects
    mnist_train_ds = MNIST_truncated(root=datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(root=datadir, train=False, download=True, transform=transform)
    
    print("MNIST data loaded.")

    # Extract the data and targets from the dataset objects
    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    return X_train, y_train, X_test, y_test, mnist_train_ds, mnist_test_ds

def load_fashion_mnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    print("Loading Fashion-MNIST data...")
    print(f"datadir: {datadir}")
    
    # Create the truncated dataset objects
    fashion_mnist_train_ds = FashionMNIST_truncated(root=datadir, train=True, download=True, transform=transform)
    fashion_mnist_test_ds = FashionMNIST_truncated(root=datadir, train=False, download=True, transform=transform)
    
    print("Fashion-MNIST data loaded.")

    # Extract the data and targets from the dataset objects
    X_train, y_train = fashion_mnist_train_ds.data, fashion_mnist_train_ds.target
    X_test, y_test = fashion_mnist_test_ds.data, fashion_mnist_test_ds.target

    return X_train, y_train, X_test, y_test, fashion_mnist_train_ds, fashion_mnist_test_ds

def load_colored_mnist_data(datadir, color_flip_prob=0.25, label_noise_prob=0.25):
    transform = transforms.Compose([transforms.ToTensor()])

    print("Loading Colored MNIST data...")
    print(f"datadir: {datadir}")

    # Create the truncated dataset objects
    colored_mnist_train_ds = ColoredMNIST_truncated(
        root=datadir,
        train=True,
        download=True,
        transform=transform,
        color_flip_prob=color_flip_prob,
        label_noise_prob=label_noise_prob
    )

    colored_mnist_test_ds = ColoredMNIST_truncated(
        root=datadir,
        train=False,
        download=True,
        transform=transform,
        color_flip_prob=color_flip_prob,
        label_noise_prob=label_noise_prob
    )

    print("Colored MNIST data loaded.")

    # Extract the data and targets from the dataset objects
    X_train, y_train = colored_mnist_train_ds.data, colored_mnist_train_ds.target
    X_test, y_test = colored_mnist_test_ds.data, colored_mnist_test_ds.target

    return X_train, y_train, X_test, y_test, colored_mnist_train_ds, colored_mnist_test_ds

def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    print("Loading CIFAR10 data...")
    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)
    print("CIFAR10 data loaded.")

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target


    return X_train, y_train, X_test, y_test, cifar10_train_ds, cifar10_test_ds

def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train, net_dataidx_map, logdir=None):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    if logdir != None:
        logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def renormalize(weights, index):
    """
    :param weights: vector of non negative weights summing to 1.
    :type weights: numpy.array
    :param index: index of the weight to remove
    :type index: int
    """
    renormalized_weights = np.delete(weights, index)
    renormalized_weights /= renormalized_weights.sum()

    return renormalized_weights


def load_mnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    print("Loading MNIST data...")
    print(f"datadir: {datadir}")
    # Create the truncated dataset objects
    mnist_train_ds = MNIST_truncated(root=datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(root=datadir, train=False, download=True, transform=transform)
    
    print("MNIST data loaded.")

    # Extract the data and targets from the dataset objects
    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    return X_train, y_train, X_test, y_test

def load_svhn_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    svhn_train_ds = SVHN_custom(datadir, train=True, download=True, transform=transform)
    svhn_test_ds = SVHN_custom(datadir, train=False, download=True, transform=transform)
    X_train, y_train = svhn_train_ds.data, svhn_train_ds.target
    X_test, y_test = svhn_test_ds.data, svhn_test_ds.target
    return X_train, y_train, X_test, y_test

def partition_data(dataset, datadir, partition, n_parties, beta=0.4, logdir=None):
    print(f"Data directory in partitioning is {datadir}")
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test, train_dl_obj,test_dl_obj = load_cifar10_data(datadir)
        y = np.concatenate([y_train, y_test], axis=0)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test= load_cifar100_data(datadir)
        y = np.concatenate([y_train, y_test], axis=0)
    elif dataset == 'fashion-mnist':
        X_train, y_train, X_test, y_test, train_dl_obj,test_dl_obj = load_fashion_mnist_data(datadir)
        y = np.concatenate([y_train, y_test], axis=0)
    elif dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
        y = np.concatenate([y_train, y_test], axis=0)
    elif dataset == 'colored-mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
        y = np.concatenate([y_train, y_test], axis=0)
        partition = "homo"
    elif dataset == 'svhn':
        X_train, y_train, X_test, y_test = load_svhn_data(datadir)
        y = np.concatenate([y_train, y_test], axis=0)
    else:
        raise ValueError("Unsupported dataset. Choose from 'cifar10', 'cifar100', 'fashion-mnist', or 'mnist'.")


    n_train = y_train.shape[0]
    n_test = y_test.shape[0]


    if partition == "homo":
        print("the homogenous dataset is being partitioned")
        idxs_train = np.random.permutation(n_train)
        idxs_test = np.random.permutation(n_test)

        batch_idxs_train = np.array_split(idxs_train, n_parties)
        batch_idxs_test = np.array_split(idxs_test, n_parties)
        
        net_dataidx_map_train = {i: batch_idxs_train[i] for i in range(n_parties)}
        net_dataidx_map_test = {i: batch_idxs_test[i] for i in range(n_parties)}

    elif partition == "XtremeHetero":
        print("the extreme heterogeneous dataset is being partitioned")
        """
        Extreme Heterogeneous Partition: Clients get exclusive access to a specific number of classes.
        The number of classes assigned to each client depends on n_parties.
        """

        if dataset == 'cifar10' or dataset == 'fashion-mnist' or dataset == 'mnist' or dataset == 'svhn':
            K = 10 
        elif dataset == "cifar100":
            K = 100 
        else:
            raise ValueError("Unsupported dataset. Choose from 'cifar10', 'cifar100', 'fashion-mnist', or 'mnist'.")

        # Ensure the number of classes is divisible by the number of clients
        if K % n_parties != 0:
            raise ValueError(f"The number of classes ({K}) must be divisible by the number of clients (n_parties).")

        classes_per_client = K // n_parties

        net_dataidx_map_train = {}
        net_dataidx_map_test = {}

        # Create a list of indices for each class
        class_idxs_train = {k: np.where(y_train == k)[0] for k in range(K)}
        class_idxs_test = {k: np.where(y_test == k)[0] for k in range(K)}


        # Distribute classes among clients
        for i in range(n_parties):
            # Determine the classes assigned to this client
            assigned_classes = range(i * classes_per_client, (i + 1) * classes_per_client)
            # Collect indices for the assigned classes
            idxs_train = np.concatenate([class_idxs_train[c] for c in assigned_classes])
            idxs_test = np.concatenate([class_idxs_test[c] for c in assigned_classes])

            # Assign to the current client without shuffling
            net_dataidx_map_train[i] = idxs_train
            net_dataidx_map_test[i] = idxs_test
 

    elif partition == "noniid-labeldir":
        print("the non-iid label directory dataset is being partitioned")
        min_size = 0
        min_require_size = 10
        if dataset == 'cifar10' or dataset == 'fashion-mnist' or dataset == 'mnist' or dataset == 'svhn':
            K = 10
        elif dataset == "cifar100":
            K = 100
        else:
            assert False
            print("Choose Dataset in readme.")

        N_train = y_train.shape[0]
        N_test = y_test.shape[0]

        net_dataidx_map_train = {}
        net_dataidx_map_test = {}

        while min_size < min_require_size:
            idx_batch_train = [[] for _ in range(n_parties)]
            idx_batch_test = [[] for _ in range(n_parties)]
            for k in range(K):
                train_idx_k = np.where(y_train == k)[0]
                test_idx_k = np.where(y_test == k)[0]

                np.random.shuffle(train_idx_k)
                np.random.shuffle(test_idx_k)

                proportions = np.random.dirichlet(np.repeat(beta, n_parties))

                ## Balance
                proportions = np.array([p * (len(idx_j) < N_train / n_parties) for p, idx_j in zip(proportions, idx_batch_train)])

                proportions = proportions / proportions.sum()
                proportions_train = (np.cumsum(proportions) * len(train_idx_k)).astype(int)[:-1]
                proportions_test = (np.cumsum(proportions) * len(test_idx_k)).astype(int)[:-1]

                idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, np.split(train_idx_k, proportions_train))]
                idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, np.split(test_idx_k, proportions_test))]
                
                min_size_train = min([len(idx_j) for idx_j in idx_batch_train])
                min_size_test = min([len(idx_j) for idx_j in idx_batch_test])
                min_size = min(min_size_train, min_size_test)

        for j in range(n_parties):
            np.random.shuffle(idx_batch_train[j])
            np.random.shuffle(idx_batch_test[j])
            net_dataidx_map_train[j] = idx_batch_train[j]
            net_dataidx_map_test[j] = idx_batch_test[j]
    
    elif partition == "XtremeHomo":
        print("the extreme homogenous dataset is being partitioned")
        net_dataidx_map_train = {}
        net_dataidx_map_test = {}

        # Split data indices across clients
        for client_id in range(n_parties):
            net_dataidx_map_train[client_id] = np.random.choice(n_train, size=n_train // n_parties, replace=False)
            net_dataidx_map_test[client_id] = np.random.choice(n_test, size=n_test // n_parties, replace=False)
            
    elif partition == "iid-label100":
        print("the iid label 100 dataset is being partitioned")
        seed = 12345
        n_fine_labels = 100
        n_coarse_labels = 20
        coarse_labels = \
            np.array([
                4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                18, 1, 2, 15, 6, 0, 17, 8, 14, 13
            ])
        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        rng = random.Random(rng_seed)
        np.random.seed(rng_seed)

        n_samples_train = y_train.shape[0]
        n_samples_test = y_test.shape[0]

        selected_indices_train = rng.sample(list(range(n_samples_train)), n_samples_train)
        selected_indices_test = rng.sample(list(range(n_samples_test)), n_samples_test)

        n_samples_by_client_train = int((n_samples_train / n_parties) // 5)
        n_samples_by_client_test = int((n_samples_test / n_parties) // 5)

        indices_by_fine_labels_train = {k: list() for k in range(n_fine_labels)}
        indices_by_coarse_labels_train = {k: list() for k in range(n_coarse_labels)}

        indices_by_fine_labels_test = {k: list() for k in range(n_fine_labels)}
        indices_by_coarse_labels_test = {k: list() for k in range(n_coarse_labels)}

        for idx in selected_indices_train:
            fine_label = y_train[idx]
            coarse_label = coarse_labels[fine_label]

            indices_by_fine_labels_train[fine_label].append(idx)
            indices_by_coarse_labels_train[coarse_label].append(idx)

        for idx in selected_indices_test:
            fine_label = y_test[idx]
            coarse_label = coarse_labels[fine_label]

            indices_by_fine_labels_test[fine_label].append(idx)
            indices_by_coarse_labels_test[coarse_label].append(idx)

        fine_labels_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

        for fine_label, coarse_label in enumerate(coarse_labels):
            fine_labels_by_coarse_labels[coarse_label].append(fine_label)

        net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}

        for client_idx in range(n_parties):
            coarse_idx = client_idx // 5
            fine_idx = fine_labels_by_coarse_labels[coarse_idx]
            for k in range(5):
                fine_label = fine_idx[k]
                sample_idx = rng.sample(list(indices_by_fine_labels_train[fine_label]), n_samples_by_client_train)
                net_dataidx_map_train[client_idx] = np.append(net_dataidx_map_train[client_idx], sample_idx)
                for idx in sample_idx:
                    indices_by_fine_labels_train[fine_label].remove(idx)

        for client_idx in range(n_parties):
            coarse_idx = client_idx // 5
            fine_idx = fine_labels_by_coarse_labels[coarse_idx]
            for k in range(5):
                fine_label = fine_idx[k]
                sample_idx = rng.sample(list(indices_by_fine_labels_test[fine_label]), n_samples_by_client_test)
                net_dataidx_map_test[client_idx] = np.append(net_dataidx_map_test[client_idx], sample_idx)
                for idx in sample_idx:
                    indices_by_fine_labels_test[fine_label].remove(idx)

    elif partition == "noniid-labeldir100":
        print("the non-iid label directory 100 dataset is being partitioned")
        seed = 12345
        alpha = 10
        n_fine_labels = 100
        n_coarse_labels = 20
        coarse_labels = \
            np.array([
                4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                18, 1, 2, 15, 6, 0, 17, 8, 14, 13
            ])

        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        rng = random.Random(rng_seed)
        np.random.seed(rng_seed)

        n_samples = y.shape[0]

        selected_indices = rng.sample(list(range(n_samples)), n_samples)

        n_samples_by_client = n_samples // n_parties

        indices_by_fine_labels = {k: list() for k in range(n_fine_labels)}
        indices_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

        for idx in selected_indices:
            fine_label = y[idx]
            coarse_label = coarse_labels[fine_label]

            indices_by_fine_labels[fine_label].append(idx)
            indices_by_coarse_labels[coarse_label].append(idx)

        available_coarse_labels = [ii for ii in range(n_coarse_labels)]

        fine_labels_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

        for fine_label, coarse_label in enumerate(coarse_labels):
            fine_labels_by_coarse_labels[coarse_label].append(fine_label)

        net_dataidx_map = [[] for i in range(n_parties)]

        for client_idx in range(n_parties):
            coarse_labels_weights = np.random.dirichlet(alpha=beta * np.ones(len(fine_labels_by_coarse_labels)))
            weights_by_coarse_labels = dict()

            for coarse_label, fine_labels in fine_labels_by_coarse_labels.items():
                weights_by_coarse_labels[coarse_label] = np.random.dirichlet(alpha=alpha * np.ones(len(fine_labels)))

            for ii in range(n_samples_by_client):
                coarse_label_idx = int(np.argmax(np.random.multinomial(1, coarse_labels_weights)))
                coarse_label = available_coarse_labels[coarse_label_idx]
                fine_label_idx = int(np.argmax(np.random.multinomial(1, weights_by_coarse_labels[coarse_label])))
                fine_label = fine_labels_by_coarse_labels[coarse_label][fine_label_idx]
                sample_idx = int(rng.choice(list(indices_by_fine_labels[fine_label])))

                net_dataidx_map[client_idx] = np.append(net_dataidx_map[client_idx], sample_idx)

                indices_by_fine_labels[fine_label].remove(sample_idx)
                indices_by_coarse_labels[coarse_label].remove(sample_idx)


                if len(indices_by_fine_labels[fine_label]) == 0:
                    fine_labels_by_coarse_labels[coarse_label].remove(fine_label)

                    weights_by_coarse_labels[coarse_label] = renormalize(weights_by_coarse_labels[coarse_label],fine_label_idx)

                    if len(indices_by_coarse_labels[coarse_label]) == 0:
                        fine_labels_by_coarse_labels.pop(coarse_label, None)
                        available_coarse_labels.remove(coarse_label)

                        coarse_labels_weights = renormalize(coarse_labels_weights, coarse_label_idx)

        random.shuffle(net_dataidx_map)
        net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        for i, index in enumerate(net_dataidx_map):
            net_dataidx_map_train[i] = np.append(net_dataidx_map_train[i], index[index < 50_000]).astype(int)
            net_dataidx_map_test[i] = np.append(net_dataidx_map_test[i], index[index >= 50_000]-50000).astype(int)

    elif partition == "noniid-labeluni":
        print("the non-iid label uniform dataset is being partitioned")
        if dataset == "cifar10":
            num = 2
        elif dataset == "cifar100":
            num = 10
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'cifar10':
            K = 10
        else:
            assert False
            print("Choose Dataset in readme.")

        # -------------------------------------------#
        # Divide classes + num samples for each user #
        # -------------------------------------------#
        assert (num * n_parties) % K == 0, "equal classes appearance is needed"
        count_per_class = (num * n_parties) // K
        class_dict = {}
        for i in range(K):
            # sampling alpha_i_c
            probs = np.random.uniform(0.4, 0.6, size=count_per_class)
            # normalizing
            probs_norm = (probs / probs.sum()).tolist()
            class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

        # -------------------------------------#
        # Assign each client with data indexes #
        # -------------------------------------#
        class_partitions = defaultdict(list)
        for i in range(n_parties):
            c = []
            for _ in range(num):
                class_counts = [class_dict[i]['count'] for i in range(K)]
                max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
                c.append(np.random.choice(max_class_counts))
                class_dict[c[-1]]['count'] -= 1
            class_partitions['class'].append(c)
            class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])

        # -------------------------- #
        # Create class index mapping #
        # -------------------------- #
        data_class_idx_train = {i: np.where(y_train == i)[0] for i in range(K)}
        data_class_idx_test = {i: np.where(y_test == i)[0] for i in range(K)}

        num_samples_train = {i: len(data_class_idx_train[i]) for i in range(K)}
        num_samples_test = {i: len(data_class_idx_test[i]) for i in range(K)}

        # --------- #
        # Shuffling #
        # --------- #
        for data_idx in data_class_idx_train.values():
            random.shuffle(data_idx)
        for data_idx in data_class_idx_test.values():
            random.shuffle(data_idx)

        # ------------------------------ #
        # Assigning samples to each user #
        # ------------------------------ #
        net_dataidx_map_train ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}

        for usr_i in range(n_parties):
            for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
                end_idx_train = int(num_samples_train[c] * p)
                end_idx_test = int(num_samples_test[c] * p)

                net_dataidx_map_train[usr_i] = np.append(net_dataidx_map_train[usr_i], data_class_idx_train[c][:end_idx_train])
                net_dataidx_map_test[usr_i] = np.append(net_dataidx_map_test[usr_i], data_class_idx_test[c][:end_idx_test])

                data_class_idx_train[c] = data_class_idx_train[c][end_idx_train:]
                data_class_idx_test[c] = data_class_idx_test[c][end_idx_test:]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map_train, logdir)
    testdata_cls_counts = record_net_data_stats(y_test, net_dataidx_map_test, logdir)

    return (X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts)
