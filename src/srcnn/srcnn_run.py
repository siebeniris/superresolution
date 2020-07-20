from __future__ import print_function
from torchvision.transforms import Compose

from .srcnn_module import SRCNN
from ..pytorch_utils import (
    calculate_valid_crop_size,
    PyTorchModelApplier,
    input_downscale_transform,
    input_upscale_transform,
)


class SRCNNPreProcessor(PyTorchModelApplier):
    def __init__(
        self, model_path, scale_factor, downscale_first, use_gpu=True, target_size=512
    ):
        crop_size = calculate_valid_crop_size(target_size, scale_factor)
        input_transform = (
            Compose(
                [
                    input_downscale_transform(crop_size, scale_factor),
                    input_upscale_transform(crop_size),
                ]
            )
            if downscale_first
            else Compose([input_upscale_transform(crop_size)])
        )

        super(SRCNNPreProcessor, self).__init__(
            SRCNN(), model_path, input_transform, use_gpu=use_gpu
        )
        for m in self.model.modules():
            if "Conv" in str(type(m)):
                setattr(m, "padding_mode", "zeros")


class SRCNNPreProcessorGen(PyTorchModelApplier):
    def __init__(
        self,
        empty_model,
        model_path,
        scale_factor,
        downscale_first,
        use_gpu=True,
        target_size=512,
        srgan=False,
    ):
        crop_size = calculate_valid_crop_size(target_size, scale_factor)
        input_transform = (
            Compose(
                [
                    input_downscale_transform(crop_size, scale_factor),
                    input_upscale_transform(crop_size),
                ]
            )
            if downscale_first
            else Compose([input_upscale_transform(crop_size)])
        ) if not srgan else (
            input_downscale_transform(target_size,scale_factor)
            if downscale_first
            else None
        )

        super(SRCNNPreProcessorGen, self).__init__(
            empty_model, model_path, input_transform, use_gpu=use_gpu
        )
        for m in self.model.modules():
            if "Conv" in str(type(m)):
                setattr(m, "padding_mode", "zeros")
