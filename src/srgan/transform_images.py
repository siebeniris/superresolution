import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from src.srgan.data_utils import calculate_valid_crop_size
from src.pytorch_utils import downscaling_to_tensor
from src.srgan.srgan_module import Generator

parser = argparse.ArgumentParser(description='Transform images in folder')
parser.add_argument('--upscale_factor', default=2, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_folder', default='data/interim/datasets/test', type=str, help='folder containing lr images')
parser.add_argument('--model_path', default='models/srgan/100_20190717T0025/epochs/netG_epoch_2_76.pth', type=str,
                    help='generator model epoch name')
parser.add_argument('--output_path', default='data/interim/srgan_images/100_20190717T0025/5m', type=str,
                    help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_FOLDER = opt.image_folder
MODEL_NAME = opt.model_path
OUTPUT_PATH = opt.output_path

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load(MODEL_NAME))
else:
    model.load_state_dict(torch.load(MODEL_NAME, map_location=lambda storage, loc: storage))

source_dir = Path(IMAGE_FOLDER)
for file in source_dir.rglob("*.TIF"):
    image = Image.open(file)
    w, h = image.size
    crop_size = calculate_valid_crop_size(min(w, h), UPSCALE_FACTOR)
    lr_scale = downscaling_to_tensor(crop_size, UPSCALE_FACTOR)  # Resize(crop_size //
    image = Variable(ToTensor()(ToPILImage()(lr_scale(image))), volatile=True).unsqueeze(0)
    if TEST_MODE:
        image = image.cuda()

    start = time.clock()
    out = model(image)
    elapsed = (time.clock() - start)
    print('cost' + str(elapsed) + 's')
    out_img = ToPILImage()(out[0].data.cpu())
    out_img.save('{}/out_srf_{}_{}'.format(OUTPUT_PATH, str(UPSCALE_FACTOR), file.name))
