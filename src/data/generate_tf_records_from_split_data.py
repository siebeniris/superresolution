"""
Generate tf.record  for all data in data/interim/separated_data

"""

import json
import io
import os
import time
import argparse
import sys

from typing import List

from joblib import Parallel, delayed
from PIL import Image
from PIL.ImageFilter import GaussianBlur
import tensorflow as tf
import data_augm_utils as augm

parser = argparse.ArgumentParser(description="Generate tf records, choose bicubic, seedling or augment")
parser.add_argument('--input', type=argparse.FileType('r'), nargs='+', required=True,
                    default=sys.stdin, help='the path file consisting of the paths of images to process')
parser.add_argument("--output_dir", type=str, required=True, default="data/processed/tf_records",
                    help="directory of output")
parser.add_argument("--coords_dir", type=str, required=True, default="data/interim/coords_dict.json",
                    help="path of coordinates of seedlings")
parser.add_argument("--bicubic", action="store_true", default=False, help="apply bicubic interpolation")
parser.add_argument("--augment", action="store_true", default=False,
                    help="if images should be flipped and rotated as data augmentation => 8x")
parser.add_argument("--seedling", action="store_true", default=False,
                    help="if only the images with seedlings are applied")
parser.add_argument("--resize", type=int, required=False, default=1024, help="if apply bicubic, resize.")
parser.add_argument('--sr', action='store_true', default=False, help="if apply to superresolued 30m images")
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--downscaling', action='store_true', default=False, help="if apply to downscale images.")

args = parser.parse_args()


def parse_bbox_and_labels(
        image_path,
        coords_dir,
        label_pos="seedling",
        label_neg="no-seedling",
):
    '''
    Parse the bbox parameters and its labels into format expected by tensorflow.
    :param label_pos: string
    :param label_neg: string
    :param image_path: string
    :return:
        xmins: List[int],
        ymins: List[int],
        xmaxs: List[int],
        ymaxs: List[int],
        classes: List
        classes_text: List
    '''
    with open(coords_dir)as fp:
        coords_dict = json.load(fp=fp)

    xmins, ymins, xmaxs, ymaxs = [], [], [], []
    classes, classes_text = [], []

    try:
        image_name = image_path.split('/')[-1].replace('.png', '').replace('.JPG', '').replace('.TIF', '')
        for coord in coords_dict[image_name]:
            xmins.append(coord[0])
            ymins.append(coord[1])
            xmaxs.append(coord[2])
            ymaxs.append(coord[3])

            classes_text.append(label_pos.encode('utf-8'))
            classes.append(1)

        # xmins, ymins, xmaxs, ymaxs
        return xmins, ymins, xmaxs, ymaxs, classes, classes_text

    except KeyError:
        classes_text.append(label_neg.encode('utf-8'))
        classes.append(0)
        return xmins, ymins, xmaxs, ymaxs, classes, classes_text


def create_tf_example(image_path, coords_dir):
    tf_examples = []
    total_seedlings = 0
    filename = image_path.split('/')[-1].encode('utf-8')

    image_format = b'jpg'

    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)

    # apply bicubic
    if args.bicubic:
        print('apply bicubic')
        img: Image = Image.open(encoded_jpg_io).convert('RGB').resize((args.resize, args.resize), Image.BICUBIC)
        width, height = img.size
        print(height, width)

        xmin, ymin, xmax, ymax, classes, classes_text = \
            parse_bbox_and_labels(image_path, coords_dir)

        xmins = list(map(lambda x: x * 2, xmin))
        xmaxs = list(map(lambda x: x * 2, xmax))
        ymins = list(map(lambda x: x * 2, ymin))
        ymaxs = list(map(lambda x: x * 2, ymax))

    # superresolued images
    elif args.sr:
        print('applied to superresolued image')
        img: Image = Image.open(encoded_jpg_io)
        width, height = img.size
        print(height, width)

        xmin, ymin, xmax, ymax, classes, classes_text = \
            parse_bbox_and_labels(image_path, coords_dir)

        resize = args.resize
        xmins = list(map(lambda x: x * resize, xmin))
        xmaxs = list(map(lambda x: x * resize, xmax))
        ymins = list(map(lambda x: x * resize, ymin))
        ymaxs = list(map(lambda x: x * resize, ymax))

    elif args.downscaling:
        print('apply guassian blur and downscaling to the image')
        img0: Image = Image.open(encoded_jpg_io)
        width0, height0 = img0.size

        width, height = int(width0 / 2), int(height0 / 2)
        print(width, height)

        img = img0.convert('RGB').filter(GaussianBlur).resize((width, height))

        xmin, ymin, xmax, ymax, classes, classes_text = \
            parse_bbox_and_labels(image_path, coords_dir)

        xmins = list(map(lambda x: x/2, xmin))
        xmaxs = list(map(lambda x: x/2, xmax))
        ymins = list(map(lambda x: x/2, ymin))
        ymaxs = list(map(lambda x: x/2, ymax))

    else:
        img: Image = Image.open(encoded_jpg_io)
        width, height = img.size
        print(height, width)

        xmins, ymins, xmaxs, ymaxs, classes, classes_text = \
            parse_bbox_and_labels(image_path, coords_dir)

    # if only need images with seedlings or not
    cond = any(classes) if args.seedling else len(classes) != 0

    if cond:
        base_bounding_box = xmins, xmaxs, ymins, ymaxs
        list_bboxes = [base_bounding_box]
        images = [img]

        if args.augment:
            list_bboxes.append(augm.augment_flip(*base_bounding_box, width, height, 'v'))
            images.append(img.transpose(Image.FLIP_TOP_BOTTOM))

            for bboxes, image in list(zip(list_bboxes, images)):
                cur_bboxes = bboxes
                cur_image = image
                for _ in range(3):
                    cur_bboxes = augm.augment_rotate90(*cur_bboxes, width, height)
                    list_bboxes.append(cur_bboxes)
                    # rotate image 270 because it is counterclockwise
                    cur_image = cur_image.rotate(270)
                    images.append(cur_image)

        for bboxes, image in zip(list_bboxes, images):
            encoded_image_io = io.BytesIO()
            image.save(encoded_image_io, 'JPEG')
            encoded_image = encoded_image_io.getvalue()
            tf_example = init_tf_example(
                height, width,
                filename, filename, encoded_image, image_format,
                *normalize_bbox(*bboxes, width, height),
                classes_text, classes,
            )
            tf_examples.append(tf_example)
            total_seedlings += classes.count(1)

    return tf_examples, total_seedlings


def create_tf_examples_parallel(paths_list, coords_dir):
    start_time = time.time()
    tfexamples = []
    total_seedlings = 0
    for tf_example, seedlings in Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(create_tf_example)(path, coords_dir) for path in paths_list):
        tfexamples.extend(tf_example)
        total_seedlings += seedlings
    print('{} seconds for one example'.format(time.time() - start_time))

    return tfexamples, total_seedlings


def normalize_bbox(
        xmins: List[float], xmaxs: List[float], ymins: List[float], ymaxs: List[float],
        width: int, height: int):
    return \
        list(map(lambda x: x / width, xmins)), \
        list(map(lambda x: x / width, xmaxs)), \
        list(map(lambda y: y / height, ymins)), \
        list(map(lambda y: y / height, ymaxs))


def init_tf_example(height: int, width: int,
                    filename: str, source_id: str, encoded_jpg, image_format,
                    xmins: List[int], xmaxs: List[int], ymins: List[int], ymaxs: List[int],
                    classes_text: List[str], classes: List[str], ):
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[filename])),
        'image/source_id': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[source_id])),
        'image/encoded': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[encoded_jpg])),
        'image/format': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_format])),
        'image/object/bbox/xmin': tf.train.Feature(
            float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(
            float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(
            float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(
            float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(
            int64_list=tf.train.Int64List(value=classes)),
    }))
    return tf_example


def create_tf_record(output_filename, examples):
    writer = tf.io.TFRecordWriter(output_filename)
    for example in examples:
        writer.write(example.SerializeToString())
    writer.close()


def read_dataset_filenames(dataset):
    filename_list = []
    for file in dataset:
        for line in file:
            filename_list.append(line.strip())

    return filename_list


def main(_):
    input_list = read_dataset_filenames(args.input)
    coords_dict_path = args.coords_dir

    start_time = time.time()

    examples, annotations = create_tf_examples_parallel(input_list, coords_dict_path)

    print('take {} seconds in total'.format(time.time() - start_time))

    print('{} examples'.format(len(examples)))
    print('{} annotations'.format(annotations))

    aug_str = ""
    if args.augment: aug_str += "_aug"

    seedling_str = ""
    if args.seedling: seedling_str += "_seedling"

    bicubic = ""
    if args.bicubic: seedling_str += "_bicubic"

    output_path = os.path.join(args.output_dir, args.output_file + aug_str + seedling_str + bicubic + '.record')
    create_tf_record(output_path, examples)


if __name__ == '__main__':
    tf.compat.v1.app.run()
