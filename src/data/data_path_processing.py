import json
import pathlib
import argparse
import os



parser = argparse.ArgumentParser(description="Utils for data processing")
parser.add_argument("--format", type=str, default=".JPG",
                    help="suffix for files")
parser.add_argument("--datapath", type=str, help="data path ")
parser.add_argument("--seedling", action="store_true", default=False,
                    help="if only the images with seedlings are applied")
parser.add_argument("--annotation", type=str, help="annotation file path",
                    default="data/interim/coords_dict.json")
args=parser.parse_args()


def get_annotations():
    annotation_file = args.annotation

    with open(annotation_file)as file:
        annotations = json.load(file)

    return annotations


def get_images_from_directory(datapath, format='.JPG'):
    data_root = pathlib.Path(datapath)
    all_images = list(set([str(file) for file in data_root.glob('**/*'+format)]))

    if args.seedling:
        annotations = get_annotations()
        image_paths = []
        for image_path in all_images:
            basename_ = os.path.basename(image_path).replace(format,'').replace('out_srf_2_','')
            if basename_ in annotations:
                image_paths.append(image_path)
        return image_paths

    else:
        return all_images


def output_filenames(filenames):
    head, tail = os.path.split(args.datapath)

    seedling=''
    if args.seedling:
        seedling+='_seedling'
    with open(head+'/'+tail+seedling+'.txt','w')as file:
        for filename in filenames:
            file.write(filename+'\n')


if __name__ == '__main__':
    filenames= get_images_from_directory(datapath=args.datapath, format=args.format)
    output_filenames(filenames)
