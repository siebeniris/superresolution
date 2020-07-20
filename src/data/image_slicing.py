import os
from PIL import Image
import tensorflow as tf
"""Slicing of images in as much as possible n x m parts."""

def generate_filename(path, x, y):
    dir = os.path.dirname(path)
    image_name = os.path.basename(path)
    image = os.path.splitext(image_name)[0]
    dir_name=dir + '/' + image
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name + '/' + image + '-' + str(x) + '-' + str(y) + '.JPG'


def slice_and_save(path, crop_h, crop_w):
    image = Image.open(path)
    width = image.size[0]
    height = image.size[1]
    print("Width: " + str(width) + "Height: " + str(height))

    x = 0
    y = 0
    while y + crop_h< height:
        while x  + crop_w < width:
            left = x
            right =  left + crop_w
            upper = y
            lower = upper+crop_h
            crop = image.crop((left, upper, right, lower))
            crop.save(generate_filename(path, x , y))
            x = x + crop_w
        y = y + crop_h
        x = 0

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to dataset.')
flags.DEFINE_string('n', '', 'x pixels.')
flags.DEFINE_string('m', '', 'y pixels.')
flags.DEFINE_string('out_dir','', 'Output directory')


FLAGS = flags.FLAGS
data_dir = FLAGS.data_dir
slices=FLAGS.n

for filename in os.listdir(data_dir):
    if filename.upper().endswith('.JPG') or filename.upper().endswith('.TIF') :
        imagepath = data_dir + '//' + filename
        slice_and_save(imagepath, 1024, 1024)