from __future__ import print_function
import argparse
import os
import math

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import mlflow
import torch
from torchvision.transforms import *



from .srcnn_module import SRCNN, SRCNNR, SRCNNBatchNorm
from ..pytorch_utils import input_transform, target_transform, calculate_valid_crop_size, DatasetFromFolder
from src.pytorch_train import PyTorchTrainer, PyTorchRunner


supportedModels = {
    'srcnn' : SRCNN,
    'srcnn-residual': SRCNNR,
    'srcnn-bnorm': SRCNNBatchNorm,

}

def createArgumentParser():
    parser = argparse.ArgumentParser(description="PyTorch SRCNN")
    parser.add_argument(
        "--upscale_factor",
        type=int,
        default=2,
        help="Super Resolution Upscale Factor. Default: 2. Value should be 2, 3 or 4",
    )
    parser.add_argument(
        "--batch_size", type=int, default=46, help="Training Batch Size. Default: 64."
    )
    parser.add_argument('--augm',   nargs='?', const=True, default=False, type=bool, help="augment data?")
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads for data loader. Default: 4",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed to use. Default: 123"
    )
    parser.add_argument(
        "--train_set",
        type=str,
        default="data/interim/datasets/train_paths",
        help="List of images files in training set.",
    )
    parser.add_argument(
        "--test_set",
        type=str,
        default="data/interim/datasets/test_paths",
        help="List of images files in test set.",
    )
    parser.add_argument(
        "--val_set",
        type=str,
        default="data/interim/datasets/val_paths",
        help="List of images files in test set.",
    )
    parser.add_argument(
        '-v','--variant',
        default='srcnn',
        choices=supportedModels.keys(),
    )
    parser.add_argument('--eval', type=str, help="Set model to be evaluated. Sets epochs to 0.")

    return parser

def track_params(config):
    mlflow.log_param("upscale_factor", config.upscale_factor)
    mlflow.log_param("seed", config.seed)
    mlflow.log_param("train_set", config.train_set)
    mlflow.log_param("val_set", config.val_set)
    mlflow.log_param("test_set", config.test_set)
    mlflow.log_param("augm", config.augm)
    mlflow.log_param("type", config.variant)


def augment_transform() -> transforms:
    return RandomChoice([
        RandomGrayscale(p=0.1),
        RandomHorizontalFlip(),
        RandomRotation([-180,180]),
        RandomVerticalFlip(),
    ])



def prepare_dataset(filepath, upscale_factor, target_size, augm):
    crop_size = calculate_valid_crop_size(target_size, upscale_factor)
    return DatasetFromFolder(
        filepath,
        random_transform= augment_transform() if augm else None,
        input_transform=input_transform(crop_size, upscale_factor),
        target_transform=target_transform(crop_size),
    )



if __name__ == "__main__":
    parser = createArgumentParser()
    mfl = parser.add_argument_group("MLFlow")
    mfl.add_argument('-e', '--experiment_name', type=str, default='srcnn', help='Sets under which experiment the run should be created.')

    PyTorchRunner.add_arguments_to(parser)


    config = parser.parse_args()

    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run() as run:
        print("Started mlflow run '{!s}' in experiment '{!s}'.".format(run.info.run_id, config.experiment_name))
        track_params(config)

        torch.manual_seed(config.seed)

        train_set = prepare_dataset(config.train_set, config.upscale_factor, 512, config.augm)
        test_set = prepare_dataset(config.test_set, config.upscale_factor, 512, config.augm)
        val_set = prepare_dataset(config.val_set, config.upscale_factor, 512, config.augm)
        training_data_loader = DataLoader(
            dataset=train_set,
            num_workers=config.threads,
            batch_size=config.batch_size,
            shuffle=True,
        )
        testing_data_loader = DataLoader(
            dataset=test_set, num_workers=config.threads, batch_size=config.batch_size if config.eval else 1, shuffle=False
        )
        val_data_loader = DataLoader(
            dataset=val_set, num_workers=config.threads, batch_size=1, shuffle=False
        )

        srcnn = supportedModels[config.variant]()
        if config.eval:
            config.epochs = 0
            srcnn.load_state_dict(torch.load(config.eval))

        PyTorchRunner.configure_and_run(
            config=config,
            experiment_id=run.info.run_id,
            model=srcnn,
            train_data_loader=training_data_loader,
            val_data_loader=val_data_loader,
            test_data_loader=testing_data_loader,

        )
