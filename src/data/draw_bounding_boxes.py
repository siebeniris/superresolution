# output images with overlaid bounding boxes for evaluating models.
import json
import os
import io

import cv2
from PIL import Image, ImageDraw
import numpy as np
import glob

TESTSET = 'data/interim/images_30m_slices'

OUTPUT_DIR = "data/interim/30m_boundingboxes"

with open('data/interim/images_30m_slices_dict.json')as fp:
    coords_dict = json.load(fp)
print(coords_dict)

for image in glob.glob(TESTSET+'**/*.TIF'):
    image_name = os.path.basename(image).replace('.TIF', '')
    print(image_name)

    with open(image, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    img: Image = Image.open(encoded_jpg_io)
    width, height = img.size

    dst = img.convert('RGB')
    xmins, ymins, xmaxs, ymaxs = [], [], [], []

    try:
        for coord in coords_dict[image_name]:
            xmins.append(coord[0])
            ymins.append(coord[1])
            xmaxs.append(coord[2])
            ymaxs.append(coord[3])
    except Exception as ex:
        print(ex)
    print(xmins)
    xmins = list(map(lambda x: x , xmins))
    xmaxs = list(map(lambda x: x, xmaxs))
    ymins = list(map(lambda x: x , ymins))
    ymaxs = list(map(lambda x: x , ymaxs))

    # imshow(np.asarray(dst))
    draw = ImageDraw.Draw(dst)

    for idx, value in enumerate(xmins):
        xmin = value
        xmax = xmaxs[idx]
        ymin = ymins[idx]
        ymax = ymaxs[idx]

        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline='red', width=2)
    if xmins != []:
        # img = Image.fromarray(dst)
        filename = image_name+'.TIF'
        saved_path = os.path.join(OUTPUT_DIR, filename)
        dst.save(saved_path)
