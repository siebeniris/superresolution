import os, math, itertools
from statistics import mean
from argparse import ArgumentParser, Namespace
from typing import List, Callable, Dict, Tuple

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.modules.loss import _Loss
from .pytorch_utils import avg_psnr, normalize

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch, mlflow

from .pytorch_argparse import (
    TorchOptimizerArgParse,
    TorchLossArgParse,
    TorchLRSchedulerArgParse,
)


class PyTorchTrainerModel(nn.Module):
    def __init__(self, name):
        super(PyTorchTrainerModel, self).__init__()
        self.ident = name
        if torch.cuda.is_available():
            self.cuda()

    @staticmethod
    def add_arguments_to(parser):
        pass

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def __str__(self):
        return self.ident


class PyTorchTrainer:
    """
    Utility class, which implements a default training process.
    Will track metrics to mlflow. Will track as additional metric PSNR.
    """

    def __init__(
        self,
        config,
        experiment_id,
        model: PyTorchTrainerModel,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader,
        test_data_loader: DataLoader,
        optimizer: Optimizer,
        loss_fn: _Loss,
        lr_schedulers: List[_LRScheduler],
    ):
        """
        Results are safed to models/<model.ident>/<experiment_id>/

        :param config: argparse config, with options added by static method #add_arguments_to
        :param experiment_id: and str identifier for the train run, determines result directory
        :param model: the model to be trained
        :param train_data_loader: DataLoader for training data
        :param val_data_loader: DataLoader for validation data
        :param test_data_loader: DataLoader for test data
        :param optimizer: Optimizer for the training
        :param loss_fn: Loss function
        :param lr_schedulers: learning rate schedulers
        """
        self.optimizer = optimizer
        self.model = model
        self.loss_fn = loss_fn
        self.lr_schedulers = lr_schedulers
        self.experiment_id = experiment_id
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.val_data_loader = val_data_loader

    @staticmethod
    def add_arguments_to(parser: ArgumentParser):
        """
        Adds pytorch trainer relevant confiuration options to ArgumentParser.
        :param parser: ArgumentParser
        """
        trainer = parser.add_argument_group("PyTorchTrainer")
        trainer.add_argument(
            "--epochs", type=int, default=1, help="Number of epochs to train for."
        )
        # trainer.add_argument('--from-checkpoint', type=str, help='Specify checkpoint from which optimizer, loss, model and schedulers should be loaded.')

    def __track_losses_mlflow(
        self,
        epoch: int,
        **kwargs: Dict[str,float],
    ):
        """
        Logs kwargs as k value in mlflow
        :param epoch: current epoch
        """
        for metric, value in kwargs.items():
            mlflow.log_metric(metric, value, step=epoch)

        print_message = (
            "===> Epoch "
            + str(epoch)
            + " Complete:\n\t"
            + "\n\t".join(
                [name + ":\t " + str(value) for name, value in kwargs.items()]
            )
        )
        print(print_message)

    def save_model(self, config, checkpoint_epoch=None):
        """Saves mode to either checkpoint or final model if checkpoint_epoch is None"""

        if checkpoint_epoch is not None:
            model_out_path = "models/{}/{}/checkpoints/model_epoch_{}.pth".format(
                self.model.ident, self.experiment_id, checkpoint_epoch
            )
            # TODO safe parameters and optimizers for retraining
        else:
            model_out_path = "models/{}/{}/model.pth".format(
                self.model.ident, self.experiment_id
            )
            print("Saving final model to: '{}'".format(model_out_path))

        if not os.path.exists(os.path.dirname(model_out_path)):
            os.makedirs(os.path.dirname(model_out_path))
        self.model.save_model(model_out_path)
        return model_out_path



    def train(self, config: Namespace):

        epoch = 0
        """Trains the model as configured."""
        for epoch in range(1, config.epochs + 1):

            lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
            print("Training epoch with learning rate: '{!s}'".format(lr))
            mlflow.log_metric("lr", lr, epoch)

            train_losses, train_mses = self.__train_epoch()
            val_losses, val_mses = self.__evaluate(self.val_data_loader)
            for scheduler in self.lr_schedulers:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(mean(val_losses), epoch)
                else:
                    scheduler.step(epoch)

            self.__track_losses_mlflow(
                epoch,
                train_avg_loss=mean(train_losses), train_avg_psnr=avg_psnr(train_mses),
                val_avg_loss=mean(val_losses), val_avg_psnr=avg_psnr(val_mses),
            )
            self.save_model(config, checkpoint_epoch=epoch)

        test_losses, test_mses = self.__evaluate(self.test_data_loader)
        self.__track_losses_mlflow(
            epoch,
            test_avg_loss=mean(test_losses), test_avg_psnr=avg_psnr(test_losses)
        )
        result_file = self.save_model(config)
        mlflow.log_artifact(result_file)

    def __calculate_single_mses(self, predictions, targets):
        mseLoss = torch.nn.MSELoss()
        mses = []
        for pred_img, img in zip(predictions, targets):
            pred_img = normalize(pred_img)
            mses.append(mseLoss(pred_img,img).detach().item())
        return mses

    def __train_epoch(self) -> Tuple[List[float], List[float]]:
        """Trains the model for one epoch"""
        self.model.train()
        losses = []
        mses = []
        for source, target in self.train_data_loader:
            def closure(source=source, target=target):
                if torch.cuda.is_available():
                    source = source.cuda()
                    target = target.cuda()
                self.optimizer.zero_grad()
                prediction = self.model(source)
                loss = self.loss_fn(prediction, target)
                loss.backward()
                losses.append(loss.detach().item())
                mses.extend(self.__calculate_single_mses(prediction, target))
                source = source.cpu()
                target = target.cpu()
                return loss

            self.optimizer.step(closure)
        return losses, mses

    def __evaluate(self, data_loader) -> Tuple[List[float], List[float]]:
        """Evaluates the model for a declared DataLoader"""
        self.model.eval()
        losses = []
        mses = []
        for source, target in data_loader:
            if torch.cuda.is_available():
                source = source.cuda()
                target = target.cuda()
            prediction = self.model(source)
            loss = self.loss_fn(prediction, target)
            losses.append(loss.detach().item())
            mses.extend(self.__calculate_single_mses(prediction,target))
            source = source.cpu()
            target = target.cpu()
        return losses, mses


class PyTorchRunner:
    """
    Utility class to easily configure and run a PyTorchTrainer with
        TorchLossArgParse,
        TorchOptimizerArgParse,
        TorchLRSchedulerArgParse
    """

    @staticmethod
    def add_arguments_to(parser: ArgumentParser):
        PyTorchTrainer.add_arguments_to(parser)
        TorchLossArgParse.losses().add_arguments_to(parser)
        TorchOptimizerArgParse.optimizers().add_arguments_to(parser)
        TorchLRSchedulerArgParse.schedulers().add_arguments_to(parser)

    @staticmethod
    def configure_and_run(
        config,
        experiment_id,
        model: PyTorchTrainerModel,
        train_data_loader,
        val_data_loader,
        test_data_loader,
    ):

        model.cuda()
        l_config, loss_fn = TorchLossArgParse.losses().from_config(config)
        o_config, optimizer = TorchOptimizerArgParse.optimizers().from_config(
            config, model
        )
        s_config, scheduler = TorchLRSchedulerArgParse.schedulers().from_config(
            config, optimizer
        )
        if torch.cuda.is_available():
            loss_fn = loss_fn.cuda()

        for key, value in itertools.chain(
            l_config.items(), o_config.items(), s_config.items()
        ):
            if value:
                mlflow.log_param(key, value)

        trainer = PyTorchTrainer(
            config=config,
            experiment_id=experiment_id,
            model=model,
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            test_data_loader=test_data_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_schedulers=[scheduler] if scheduler else [],
        )
        trainer.train(config)
