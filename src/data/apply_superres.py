import argparse
import sys
import os
import glob

from PIL import Image
from torchvision.transforms import Compose
import torch

from src.srcnn.srcnn_module import SRCNN
from src.pytorch_utils import PyTorchModelApplier, input_upscale_transform, calculate_valid_crop_size

def create_parser():
    parser = argparse.ArgumentParser(description="apply superresolution on images")
    parser.add_argument('-i','--input', nargs='?', type=argparse.FileType('r'), default =sys.stdin, help="List of input images, will be read by default from stdin.")
    parser.add_argument('-m', '--model_dir', type=str, required=True, )
    parser.add_argument('--checkpoints', type=str, nargs='?', const='::', default=None, help='Slice used checkpoints using start:stop:step python syntax.')
    parser.add_argument('-o', '--output_dir', type=str, help="directory to output images" )
    return parser


class SRCNNPreProcessor (PyTorchModelApplier):
    def __init__(self,model_path, scale_factor,  use_gpu=True, target_size=1024,):
        crop_size = calculate_valid_crop_size(target_size, scale_factor)

        input_transform = Compose([input_upscale_transform(crop_size)])

        super(SRCNNPreProcessor, self).__init__(SRCNN(),model_path,input_transform,use_gpu=use_gpu)
        for m in self.model.modules():
            if "Conv" in str(type(m)):
                setattr(m, "padding_mode", "zeros")


if __name__ == '__main__':
    parser= create_parser()
    config= parser.parse_args()

    print(config)

    images_dict={line.strip(): [Image.open(line.strip())] for line in config.input}

    # choose checkpoints or the model pth
    if config.checkpoints:
        models = sorted(
            glob.glob(config.model_dir + 'checkpoints/*.pth'),
            key=lambda x : int(os.path.basename(x).replace('model_epoch_', '').replace('.pth',''))
        )
        slice = slice(*[None if x is ''else int(x) for x in config.checkpoints.split(':')])
        models = models[slice]

    else:
        models = [config.model_dir + "model.pth"]

    out_dir = config.output_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    for model in models:
        proc= SRCNNPreProcessor(model,scale_factor=2.0,  use_gpu=torch.cuda.is_available(), target_size=1024 )
        for key, images in images_dict.items():
            result_image = proc.apply(images[0])
            out_file= out_dir+ os.path.basename(key)
            result_image.save(out_file)
            print('Save images to {}'.format(out_file))




