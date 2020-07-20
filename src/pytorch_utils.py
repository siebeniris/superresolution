from typing import Callable

from PIL import ImageFilter, Image
import PIL

import torch.utils.data as data
import torchvision.transforms as transforms
import torch
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from typing import *
from statistics import mean
import math


def avg_psnr(mses: List[float]):
    return mean(map(lambda mse: 10 * math.log10(1**2 / mse), mses))

def normalize(tensor):
    min = torch.min(tensor)
    range = torch.max(tensor) - min
    if range > 0:
        tensor = (tensor - min) / range
    else:
        tensor = torch.zeros(tensor.size())

    return tensor



def calculate_valid_crop_size(target_size, upscale_factor):
    return int(target_size - (target_size % upscale_factor))


def gauss_blur_transform(radius=2):
    def transform(img):
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    return transform


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath)
    return img


def downscaling_to_tensor(crop_size, upscale_factor) -> transforms:
    return Compose([
        input_downscale_transform(crop_size, upscale_factor),
        ToTensor()
    ])


def input_downscale_transform(crop_size, upscale_factor) -> transforms:
    return Compose([
        CenterCrop(crop_size),
        gauss_blur_transform(radius=upscale_factor),
        Resize(int(crop_size / upscale_factor)),
    ])


def input_upscale_transform(crop_size) -> transforms:
    return Resize(crop_size, interpolation=PIL.Image.BICUBIC)


def input_transform(crop_size, upscale_factor) -> transforms:
    return Compose(
        [
            input_downscale_transform(crop_size, upscale_factor),
            input_upscale_transform(crop_size),
        ]
    )


def target_transform(crop_size) -> transforms:
    return CenterCrop(crop_size)


class PyTorchModelApplier:

    def __init__(
            self,
            null_model: torch.nn.Module,
            model_path,
            preprocessing_transform: transforms,
            use_gpu=True):
        self.model = null_model
        self.model.load_state_dict(torch.load(model_path))
        self.use_gpu = use_gpu
        if use_gpu: self.model.cuda()
        self.input_transform = preprocessing_transform

    def apply(self, image: Image) -> Image:
        if self.input_transform:
            image = self.input_transform(image)
        tensor = ToTensor()(image)
        input_tensor = tensor.view(1, -1, image.size[1], image.size[0])
        if self.use_gpu:
            input_tensor = input_tensor.cuda()

        output = self.model(input_tensor)
        output = output.cpu()
        result_image = transforms.ToPILImage()(output[0]).convert("RGB")
        return result_image

    def lowres(self, image: Image) -> Image:
        if self.input_transform:
            image = self.input_transform(image)
        return image


class DatasetFromFolder(data.Dataset):
    def __init__(
            self,
            file_dir,
            random_transform: transforms = None,
            input_transform: transforms = None,
            target_transform: transforms = None,
    ):
        super(DatasetFromFolder, self).__init__()
        with open(file_dir, "r") as f:
            read_file = [x.strip() for x in f.readlines()]
        self.image_filenames = read_file
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.random_transform = random_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])

        # apply random transform for data augmentation
        if self.random_transform:
            input = self.random_transform(input)

        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        input = ToTensor()(input)
        if self.target_transform:
            target = self.target_transform(target)
        target = ToTensor()(target)
        return input, target

    def __len__(self):
        return len(self.image_filenames)
