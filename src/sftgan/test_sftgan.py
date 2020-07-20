import os.path
import glob
import cv2
import numpy as np
import torch
import argparse

from src.sftgan.data.util import imresize, modcrop
import src.sftgan.utils.util as util
import src.sftgan.models.modules.sft_arch as sft

parser = argparse.ArgumentParser(description="apply superresolution on images")
parser.add_argument('--model', type=str, help="the path of the model")
parser.add_argument('--input', type=str, help='the folder of the images to test')
parser.add_argument('--input_dir')
parser.add_argument('--thirty', action="store_true", default=False,
                    help="if only the 30m images are applied")
parser.add_argument('--format', type=str, default='.JPG', help="either JPG or TIF")
args = parser.parse_args()

model_path= args.model

test_img_folder_name = args.input
test_img_folder = args.input_dir + test_img_folder_name  # HR images
test_prob_path = 'data/processed/sftgan/' + test_img_folder_name + '_segprob'  # probability maps
save_result_path = 'data/processed/sftgan/' + test_img_folder_name + '_result'  # results

# make dirs
util.mkdirs([save_result_path])

if 'torch' in model_path:  # torch version
    model = sft.SFT_Net_torch()
else:
    model = sft.SFT_Net()
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.cuda()

print('sftgan testing...')


idx = 0
for path in glob.glob(test_img_folder + '**/*'+args.format):
    idx += 1
    basename = os.path.basename(path)
    base = os.path.splitext(basename)[0]
    print(idx, base)
    # read image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = modcrop(img, 8)
    img = img * 1.0 / 255

    print(img.shape)

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    # matlab imresize
    if args.thirty:
        img_LR = imresize(img, 1/2, antialiasing=True)
    else:
        img_LR= imresize(img, 1/4, antialiasing=True)

    img_LR = img_LR.unsqueeze(0)
    img_LR = img_LR.cuda()


    # read seg
    seg = torch.load(os.path.join(test_prob_path, base + '_bic.pth'))
    seg = seg.unsqueeze(0)
    seg = seg.cuda()

    output = model((img_LR, seg)).data.squeeze()

    output = util.tensor2img(output)
    util.save_img(output, os.path.join(save_result_path, base + '.png'))
