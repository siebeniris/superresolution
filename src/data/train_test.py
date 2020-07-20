"""
Author = Igor Frumin
Date last modified: 25/05/2019
"""
import filecmp
import fnmatch
import itertools
import os
import shutil
import sys
from shutil import copyfile
from pathlib import Path
import argparse
from typing import List, Tuple
from datetime import datetime
import glob
import cv2
from sklearn.model_selection import train_test_split
from collections import defaultdict

import multiprocessing as mp

import mlflow as mlf


def create_folder(p, n):
    if not (isinstance(p, str)):
        raise AttributeError('Path must be a string')
    if not os.path.exists(p):
        raise OSError('Path does not exist')

    if os.path.exists(p + '/' + n) and os.path.isdir(p + '/' + n):
        shutil.rmtree(p + '/' + n)
    if not os.path.exists(p + '/' + n):
        os.makedirs(p + '/' + n)


# Delete a folder
def delete_folder(p, n):
    if os.path.exists(p + '/' + n) and os.path.isdir(p + '/' + n):
        shutil.rmtree(p + '/' + n)


# Merge all sub-folder to one main folder created above
def merge(source, target):
    print('Creating a folder with all images:')
    begin = datetime.now()
    for root, dirs, files in os.walk(source):
        for n in files:
            subject = root + "/" + n
            f = target + "/" + n
            shutil.copy(subject, f)
    print('Folder created: Duration: %s' % (datetime.now() - begin))


# Search for files with equal xml-name only
def search_images_with_xml(source, target):
    begin = datetime.now()
    files = os.listdir(source)
    name_list = []
    print('Searching for labeled images with .XML-data:')

    for file1, file2 in itertools.combinations(files, 2):
        first = file1.rsplit('.', 1)[0]
        second = file2.rsplit('.', 1)[0]

        if first == second:
            if fnmatch.fnmatch(file1, '*.JPG') and fnmatch.fnmatch(file2, '*.XML'):
                name_list.append(file1)
                name_list.append(file2)
            elif fnmatch.fnmatch(file1, '*.XML') and fnmatch.fnmatch(file2, '*.JPG'):
                name_list.append(file1)
                name_list.append(file2)

    for file in name_list:
        shutil.copy(os.path.join(source, file), target)
    print('All images with .XML-data found: Duration: %s' % (datetime.now() - begin))


def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func

    return decorate


class Progress:
    def __init__(self, tasks="?", stepsize=1000):
        self.stepsize = stepsize
        self.tasks = tasks
        self.counter = 0
        self.new_iteration = 1

    def update(self):
        self.counter += 1
        cur = self.counter
        if cur >= self.new_iteration * self.stepsize:
            self.new_iteration += 1

            print("{!s} finished {!s}. Estimated total: {!s}.".format(
                mp.current_process().pid,
                cur,
                cur * mp.cpu_count()))


@static_var("progress", Progress())
def _load_image_and_score(path: str):
    _load_image_and_score.progress.update()
    image = cv2.imread(path)
    _, des = cv2.ORB_create().detectAndCompute(image, None)
    return (path, des)


@static_var("progress", Progress())
def _compute_similarity_score(path1, path2, des1, des2, lowe_ratio):
    _compute_similarity_score.progress.update()
    matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < lowe_ratio * n.distance]
    return path1, path2, len(good)


def compute_scores_parallel(
        paths: List[str],
        lowe_ratio: float = 0.89,
        magic_number: int = 95, ):
    """
    :return: List[(str, str)] images that are too similar
    """
    with mp.Pool(mp.cpu_count()) as parallel_pool:
        setattr(_load_image_and_score, "progress", Progress(tasks=len(paths)))
        print("Loading and evaluating {!s} images with {!s} processes.".format(len(paths), mp.cpu_count()))
        scored_images = parallel_pool.map(_load_image_and_score, paths)

    tasks = [(path1, path2, des1, des2, lowe_ratio) for path1, des1 in scored_images for path2, des2 in scored_images]
    tasks = list(filter(lambda tuple4: tuple4[0] != tuple4[1], tasks))  # removing instances where images are the same

    with mp.Pool(mp.cpu_count()) as parallel_pool:
        _compute_similarity_score.progress = Progress(tasks=len(tasks))
        print("Comparing {!s} images with {!s} processes.".format(len(tasks), mp.cpu_count()))
        score_triples = parallel_pool.starmap(_compute_similarity_score, tasks)

    return [(path1, path2) for path1, path2, score in score_triples if score >= magic_number]


def _resolve_transitive_image_similarities(overlaps: List[Tuple[str, str]]):
    """
    :param overlaps:
    :return: Dict[List[str]] list of grouped images
    """
    path_to_group_mapping = dict()

    groupId_counter = 0

    for path1, path2 in overlaps:
        if path1 in path_to_group_mapping and path2 in path_to_group_mapping:
            # if value differs -> merge groups
            if path_to_group_mapping.get(path1) == path_to_group_mapping.get(path2):
                continue
            else:
                group_to_merge = path_to_group_mapping.get(path2)
                for key, value in path_to_group_mapping.items():
                    if group_to_merge == key:
                        path_to_group_mapping[key] = group_to_merge
                continue

        elif path1 in path_to_group_mapping:
            group = path_to_group_mapping.get(path1)
            path_to_group_mapping[path2] = group

        elif path2 in path_to_group_mapping:
            group = path_to_group_mapping.get(path2)
            path_to_group_mapping[path1] = group
            pass
        else:
            groupId_counter += 1
            path_to_group_mapping[path1] = groupId_counter
            path_to_group_mapping[path2] = groupId_counter

    print("Found {!s} groups of transitively similar images".format(groupId_counter))
    groups = defaultdict(list)

    for key, value in path_to_group_mapping.items():
        groups[value].append(key)

    for key, value in groups.items():
        print("Group {!s} has size: {!s}.".format(key, len(value)))

    group_sum = sum(map(len, groups.values()))
    print("Found {!s} files in groups.".format(group_sum))

    mlf.log_metric("transitive_groups", groupId_counter)
    mlf.log_metric("images_groups", group_sum)

    return groups


# Compares the similarity of each image to every other - reduces time by using the function sum(x-n)
# Splits the data in to train/test set
def compare_and_split(
        jpg_paths: List[str],
        out_dir: str,
        file_base_name: str,
        ratio: float,
):  # TODO create folder

    train_set, test_set = set(), set()

    begin = datetime.now()
    print('Comparing and splitting %s images into train and test sets: ' % len(jpg_paths))

    image_list = jpg_paths

    x_train, x_test = train_test_split(image_list, test_size=ratio)
    test_test_set_size = len(x_test)

    # Creates a sorted list with triples (image1, image2, magic number)
    # Magic number defines a number of matches of equal key-points in both images - limitation for the split
    overlaps: List[str, str] = compute_scores_parallel(image_list, lowe_ratio=0.89, magic_number=95)
    print(len(overlaps))

    # Dictionary to group similar images
    grouped = _resolve_transitive_image_similarities(overlaps)

    blacklist = set()  # Images that cannot be added at a later stage to the test set
    for items in sorted(grouped.values(), key=len):
        for i in items:
            blacklist.add(i)
        if len(items) + len(test_set) <= test_test_set_size:
            for i in items:
                test_set.add(i)
        else:
            for i in items:
                train_set.add(i)

    for path in set(image_list).difference(blacklist):
        if len(test_set) < test_test_set_size:
            test_set.add(path)
        else:
            train_set.add(path)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train_set_file_name = out_dir + file_base_name + ".train"
    with open(train_set_file_name, mode='w', encoding="utf-8") as train_set_file:
        for path in train_set:
            train_set_file.write(path + "\n")

    test_set_file_name = out_dir + file_base_name + ".test"
    with open(test_set_file_name, mode='w', encoding="utf-8") as test_set_file:
        for path in test_set:
            test_set_file.write(path + "\n")

    # mlf.log_artifact(test_set_file_name)
    # mlf.log_artifacts(train_set_file_name)

    print('Copied {!s} images to test data and {!s} images to train data.'.format(len(test_set), len(train_set)))
    final_ratio = len(test_set) / len(jpg_paths)
    mlf.log_param("final_ration", final_ratio)
    print('Final ratio: {!s}'.format(final_ratio))
    print('Successful split data in train and test sets : Duration: {!s}'.format(datetime.now() - begin))


# Argparse settings
parser = argparse.ArgumentParser(description='Splits the data in train/test sets')
parser.add_argument('-s', '--set', type=str, metavar='', required=True,
                    help='use "464" for s464_5m or "466" for s466_5m')
parser.add_argument('-sw', '--sumwin', type=str, metavar='', required=True,
                    help='use "s" for summer, "w" for winter, or "sw" for both.')
parser.add_argument('-p', '--parameter', type=float, metavar='', required=True,
                    help='use float parameter between 0.0 and 1.0')
parser.add_argument('-x', '--xml_data', action='store_true', required=False,
                    help='use if you want to work only with labeled data')
args = parser.parse_args()

# Main
if __name__ == '__main__':

    # defines to which value simplified params should be mapped
    param_dict = {
        '464': 's464_5m',
        '466': 's466_5m',
        's': '1708',
        'w': '1710'}

    set_base_paths = ['data/interim/data/' + param_dict.get(set_ident) for set_ident in args.set.split(" ")]
    sources = []
    for set_path in set_base_paths:
        sources.extend(
            [set_path + '/' + param_dict.get(season) + '/gelabelt/' for season in args.sumwin]
        )

    jpg_paths = []
    for base_path in sources:
        jpg_paths.extend(glob.glob(base_path + '**/*.JPG'))

    outputfile_base = "_".join(
        ['data', *args.set.split(" "), args.sumwin, 'xmlonly' if args.xml_data else 'all', str(args.parameter)])

    if args.xml_data:
        jpg_paths = list(filter(
            lambda path: os.path.isfile(path.replace('.JPG', '.xml')),
            jpg_paths))

    compare_and_split(jpg_paths, 'data/interim/sets/', outputfile_base, args.parameter)
