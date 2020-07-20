'''
Downsacale and apply gaussian filter to Images.

python src/data/gaussian_blur.py --src data/interim/datasets/test/ --dst data/processed/gaussian_blurred/test --size 2
'''
import pathlib
import argparse

from skimage.transform import resize
from skimage import io


def generate_argparser():
    parser = argparse.ArgumentParser(description="Image Preprocessing")
    parser.add_argument('--src', required=True, help="Give an source directory of images")
    parser.add_argument('--dst', required=True, help="Give a destionation directory of images ")
    parser.add_argument('--size', required=True, type=int, help="Downscale size for image")
    return parser


def loading_src_images(source_dir):
    image_root = pathlib.Path(source_dir)
    # iterate over the list of the images '.JPG' format
    image_paths = list()
    for item in image_root.iterdir():
        image_paths.append(str(item))
    return image_paths


def apply_gaussian_blur(image_path, dst_dir, size):
    image_name = image_path.rsplit('/', 1)[-1]
    dst_path = dst_dir + '/' + image_name
    print(dst_path)

    src_image = io.imread(image_path)

    # before downscale, applying gaussian filter (anti-aliasing)it
    dst_image = resize(src_image, (src_image.shape[0] / size, src_image.shape[1] / size), anti_aliasing=True)
    io.imsave(dst_path, dst_image)

    return dst_path


if __name__ == '__main__':
    parser = generate_argparser()
    args = parser.parse_args()
    image_paths = loading_src_images(args.src)
    # with open('data/processed/gaussian_blurred/paths', 'w')as f:
    for image_path in image_paths:
        dst_path = apply_gaussian_blur(image_path, args.dst, args.size)
        #f.write(dst_path + '\n')
