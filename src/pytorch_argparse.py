import argparse, logging
from typing import Dict, Tuple, Any, Callable, Generic, TypeVar
from ast import literal_eval

from torch.optim import Optimizer, Adam, SparseAdam
from torch.optim.lr_scheduler import *
from torch.nn import L1Loss, MSELoss
from torch.nn import Module

# type aliases
Name = str
ArgumentName = str
Type = Callable[[str], Any]
Default = Any
Constructor = Any
T = TypeVar("T")

ArgParseParameters = Dict[str, Any]


class TorchArgParse(Generic[T]):
    """
    This class can be utilized to add a mutually exclusive set of constructors and its parameters as options to an argparse parser.
    And create the selected on from a parsed argparse config.

    The intention is to provide a generic way to make torch optimizers, losses, learning rate schedulers,
    and custom implementations of their interfaces easily available as options on the command line and parsed in a generic way.

    @see TorchOptimizerArgParse, TorchLossArgParse, TorchLRSchedulerArgParse
    """

    def __init__(self, base, description="Choose one."):
        """Creates a generic torch argparse option group.
        At least one constructor must be added, to make this group functional.
        You may use a static generator function of a subclass to create a preconfigured one.

        :param base: The base name of the argument group. It is prepended with 'torch.',
        consequently the fitting torch package name may be the most reasonable choice.
        :param description: The description of the parser argument group.
        """
        self.dict_types: Dict[
            Name, Tuple[Constructor, Dict[ArgumentName, ArgParseParameters]]
        ] = {}
        self.base = "torch." + base
        self.arg_base = "--" + self.base
        self.description = description
        logging.basicConfig(format="%(asctime)-15s %(message)s")
        self.logger = logging.getLogger("TorchArgParse")

    def add_constructor(
        self, name, constructor, parameters: Dict[ArgumentName, ArgParseParameters]
    ):
        """Builder method to add a constructor choice to the parser.

        :param name: The name of the choice, which may be selected.
        :param constructor: The constructor of the object to be created, when parsed from config
        :param parameters: A mapping of constructors argument names to a Mapping of argparse arguments and its values.
            Uses static method args of this class to create the argparse arguments easily.
            The argparse arguments must not contain 'help', 'const' or 'nargs'.

        :return: Updated TorchArgParse
        """
        self.dict_types[name] = (constructor, parameters)
        if "nargs" in parameters or "help" in parameters or "const" in parameters:
            raise KeyError(
                "Argparse parameters must not contain params 'const', 'help' or 'nargs'."
            )
        return self

    @staticmethod
    def args(default: T, type_fun: Callable[[str], T], **kwargs):
        """Utility method to use in conjunction with add_constructor.

        :param default: the default value for the parameter if none is specified on the command line.
        :param type_fun: type parameter for argparse. Converts str into desired type of argument.
        :param kwargs: additional argparse arguments, may not contain 'const', 'help' or 'nargs'
        :return: **kwargs
        """
        kwargs["default"] = default
        kwargs["type"] = type_fun
        return kwargs

    def _from_config(self, config: argparse.Namespace, **kwargs) -> Tuple[Dict, T]:
        """Constructs object from argpars config.
        @Note: This will only work if the argparse options of this instance were added to the parser using #add_arguments_to before parsing.

        :param config: the parse config
        :param kwargs: additional arguments that may be passed to the constructor,
            useful for constructor classes that require a common dependency on creation.
        :return: The configure object.
        """
        config_dict = vars(config)
        opt_name = config_dict[self.base]
        constructor, parameters = self.dict_types[opt_name]
        opt_prefix = self.base + "." + opt_name + "."
        rm_prefix = lambda x: x.replace(opt_prefix, "")
        rm_base_prefix = lambda x: x.replace("torch.", "")

        params_dict = {
            rm_prefix(param): value
            if value
            else parameters[rm_prefix(param)]["default"]
            for param, value in config_dict.items()
            if param.startswith(opt_prefix)
        }
        tracked_params_dict = {
            rm_base_prefix(param): value
            if value
            else parameters[rm_prefix(param)]["default"]
            for param, value in config_dict.items()
            if param.startswith(self.base + "." + opt_name)
        }

        ignored_params = {
            k: v
            for k, v in config_dict.items()
            if k.startswith(self.base + ".") and not k.startswith(opt_prefix) and v
        }
        for k, v in ignored_params.items():
            self.logger.warning(
                "\033[1m\033[1;33mParam '--{}={}' was specified, but will be ignored.\033[0m".format(
                    k, v
                )
            )

        constructed_class = constructor(**params_dict, **kwargs)
        return tracked_params_dict, constructed_class

    def add_arguments_to(self, parser):
        """Add arguments to argparse parse for all configured constructor options.

        :param parser: the parser these are supposed to be added to.
        """
        all_types = parser.add_argument_group(
            title=self.base, description=self.description
        )
        all_types.add_argument(
            self.arg_base, required=True, choices=list(self.dict_types.keys()), type=str
        )

        for name, (constructor, options) in self.dict_types.items():
            type_arguments = parser.add_argument_group(
                self.base + "." + name, description="Available parameters"
            )
            for argument, argparse_params in options.items():
                type_arguments.add_argument(
                    self.arg_base + "." + name + "." + argument,
                    nargs="?",
                    const=argparse_params["default"],
                    help="e.g. '{!s}'".format(argparse_params["default"]),
                    **{
                        k: v
                        for k, v in argparse_params.items()
                        if k not in ["nargs", "help", "const", "default", "nargs"]
                    },
                )


class TorchOptimizerArgParse(TorchArgParse):
    def __init__(self):
        super().__init__(base="optim")

    def from_config(self, config: argparse.Namespace, model: Module) -> Tuple[Dict, T]:
        """
        Constructs torch.optim from argparse config and model.
        :param config: parsed argparse config
        :param model: the model for the optimizer
        :return:
        """
        return super()._from_config(config, params=model.parameters())

    @staticmethod
    def optimizers():
        """
        Generator function for all supported torch optimizers by TorchOptimizerArgParse.
        :return: Preconfigured TorchOptimizerArgParse
        """
        optimizers = TorchOptimizerArgParse()
        optimizers.add_constructor(
            "Adam",
            Adam,
            {
                "lr": TorchArgParse.args(0.001, float),
                "betas": TorchArgParse.args((0.9, 0.999), literal_eval),
                "eps": TorchArgParse.args(1e-08, float),
                "weight_decay": TorchArgParse.args(0, float),
                "amsgrad": TorchArgParse.args(False, bool),
            },
        )
        optimizers.add_constructor(
            "SparseAdam",
            SparseAdam,
            {
                "lr": TorchArgParse.args(0.001, float),
                "betas": TorchArgParse.args((0.9, 0.999), literal_eval),
                "eps": TorchArgParse.args(1e-08, float),
            },
        )
        return optimizers


class TorchLossArgParse(TorchArgParse):
    def __init__(self):
        super().__init__(base="loss")

    def from_config(self, config):
        """
        Constructs torch.loss from argparse config.
        :param config: parsed argparse config
        """
        return super()._from_config(config)

    @staticmethod
    def losses():
        """
        Generator function for all supported torch optimizers by TorchLossArgParse.
        :return: Preconfigured TorchLossArgParse
        """
        loss_functions = TorchLossArgParse()
        loss_functions.add_constructor(
            "L1",
            L1Loss,
            {
                "size_average": TorchArgParse.args(None, bool),
                "reduce": TorchArgParse.args(None, bool),
                "reduction": TorchArgParse.args(
                    "mean", str, choices=["none", "mean", "sum", "none"]
                ),
            },
        )
        loss_functions.add_constructor(
            "MSE",
            MSELoss,
            {
                "size_average": TorchArgParse.args(None, bool),
                "reduce": TorchArgParse.args(None, bool),
                "reduction": TorchArgParse.args(
                    "mean", str, choices=["none", "mean", "sum", "none"]
                ),
            },
        )
        return loss_functions


class TorchLRSchedulerArgParse(TorchArgParse):
    def __init__(self):
        super().__init__(base="lr_sched")

    def from_config(self, config, optimizer):
        """
        Constructs torch.lr_scheduler from argparse config.
        :param config: parsed argparse config
        """
        return super()._from_config(config, optimizer=optimizer)

    @staticmethod
    def schedulers():
        """
        Generator function for all supported torch optimizers by TorchLRSchedulerArgParse.
        :return: Preconfigured TorchLRSchedulerArgParse
        """

        def __none_constructor(**kwargs):
            return None

        scheds = TorchLRSchedulerArgParse()
        scheds.add_constructor("None", __none_constructor, {})
        scheds.add_constructor(
            "StepLR",
            StepLR,
            {
                "step_size": TorchArgParse.args(None, int),
                "gamma": TorchArgParse.args(0.1, float),
                "last_epoch": TorchArgParse.args(-1, int),
            },
        )
        scheds.add_constructor(
            "MultiStepLR",
            StepLR,
            {
                "milestones": TorchArgParse.args([], literal_eval),
                "gamma": TorchArgParse.args(0.1, float),
                "last_epoch": TorchArgParse.args(-1, int),
            },
        )
        scheds.add_constructor(
            "ExponentialLR", ExponentialLR, {"gamma": TorchArgParse.args(0.1, float)}
        )
        scheds.add_constructor(
            "CosineAnnealingLR",
            CosineAnnealingLR,
            {
                "T_max": TorchArgParse.args(None, int),
                "eta_min": TorchArgParse.args(0, float),
                "last_epoch": TorchArgParse.args(-1, int),
            },
        )

        scheds.add_constructor(
            "ReduceLROnPlateau",
            ReduceLROnPlateau,
            {
                "mode": TorchArgParse.args("min", str, choices=["min", "max"]),
                "factor": TorchArgParse.args(0.1, float),
                "patience": TorchArgParse.args(10, int),
                "verbose": TorchArgParse.args(False, bool),
                "threshold": TorchArgParse.args(1e-4, float),
                "threshold_mode": TorchArgParse.args(
                    "rel", str, choices=["rel", "abs"]
                ),
                "cooldown": TorchArgParse.args(0, int),
                "min_lr": TorchArgParse.args(0, literal_eval),
                "eps": TorchArgParse.args(1e-8, float),
            },
        )
        return scheds
