'''
Apply Gaussian Blur and Bicubic Interpolation to Images.

python src/data/bicubic.py --src data/interim/datasets/train --dst data/processed/bicubic_images/train --sizex 1024 --sizey 1024

'''
import pathlib
import argparse

from PIL import Image


def generate_argparser():
    parser = argparse.ArgumentParser(description="Image Preprocessing")
    parser.add_argument('--src', required=True, help="Give an source directory of images")
    parser.add_argument('--dst', required=True, help="Give a destionation directory of images ")
    parser.add_argument('--sizex', required=True, type=int, help="Resize for BICUBIC Interpolation")
    parser.add_argument('--sizey', required=True, type=int, help="Resize for BICUBIC Interpolation")
    return parser


def loading_src_images(source_dir):
    image_root = pathlib.Path(source_dir)
    # iterate over the list of the images '.JPG' format
    image_paths = list()
    for item in image_root.iterdir():
        image_paths.append(str(item))
    return image_paths


def apply_bicubic_interpolation(image_path, dst_dir, size_x=1024, size_y=1024):
    image_name = image_path.rsplit('/', 1)[-1]
    dst_path = dst_dir + '/' + image_name

    im = Image.open(image_path).convert('RGB')
    # apply bicubic interpolat ion tothe RGB image 
    # (1024,1024)
    new_img = im.resize((size_x, size_y), Image.BICUBIC)
    # save image to the path
    new_img.save(dst_path)


if __name__ == '__main__':
    parser = generate_argparser()
    args = parser.parse_args()
    image_paths = loading_src_images(args.src)
    for image_path in image_paths:
        apply_bicubic_interpolation(image_path, args.dst,
                                    args.sizex, args.sizey)
