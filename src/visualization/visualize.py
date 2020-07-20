from __future__ import print_function
import argparse
import os
import sys, glob, itertools, math
from typing import List, Dict
from PIL import Image, ImageFont, ImageDraw
import torch

from ..srcnn.srcnn_run import SRCNNPreProcessor, SRCNNPreProcessorGen
from src.srgan.srgan_module import Generator as SRGANGenerator
from ..srcnn.srcnn_module import *


supportedModels = {
    "srcnn": SRCNN,
    "srcnn-bnorm": SRCNNBatchNorm,
    "srcnn-residual": SRCNNR,
    "srgan": lambda: SRGANGenerator(scale_factor=2),
}


class PytorchSRVisulizator:
    """
    Visualizes pytorch nn.models:

    * model files should contain only the state dict, because the Model classes can't be pickled at runtime.
    * model dir must follow this hierarch:
    modeldir
      | model.pth #final model file
      | checkpoints
          | model_epoch_{}.pth # '{}' is replaced with epoch number

    """

    def __init__(self, config):
        self.config = config

    @staticmethod
    def add_arguments_to(parser: argparse.ArgumentParser):
        parser.add_argument(
            "-i",
            "--input",
            nargs="?",
            type=argparse.FileType("r"),
            default=sys.stdin,
            help="List of input images, will be read by default from stdin.",
        )
        parser.add_argument(
            "-t",
            "--model_type",
            required=True,
            default="srcnn",
            choices=supportedModels.keys(),
        )

        parser.add_argument(
            "-m", "--model_dir", type=str, required=True, help="Model directory to use"
        )
        parser.add_argument(
            "-s",
            "--scale_factor",
            type=int,
            default=2.0,
            help="Factor by which super resolution is needed",
        )

        parser.add_argument(
            "-l",
            "--inc_lowres",
            type=bool,
            const=True,
            nargs="?",
            default=False,
            help="Whether lowres images should be included.",
        )
        parser.add_argument(
            "-o",
            "--inc_highres",
            type=bool,
            const=True,
            nargs="?",
            default=False,
            help="Whether highres images should be included.",
        )

        parser.add_argument(
            "--merge_to",
            type=str,
            default=None,
            help="Merges all generated images to specified filename.",
        )
        parser.add_argument(
            "-c",
            "--columns",
            type=int,
            default=None,
            help="Generated images per line. Defaults to one line per input image.",
        )
        parser.add_argument(
            "--checkpoints",
            type=str,
            nargs="?",
            const="::",
            default=None,
            help="Slice used checkpoints using start:stop:step python syntax.",
        )
        parser.add_argument(
            "-a",
            "--annotate",
            type=bool,
            const=True,
            nargs="?",
            default=False,
            help="Whether images should be labeled.",
        )
        parser.add_argument(
            "-out", "--outdir", type=str, help="output directory of the images"
        )
        parser.add_argument(
            '-no', '--no_downscale', nargs='?', const=True, default=False,
            help='Whether downscaling preprocessing should not be applied.'
        )

        return parser

    def model_final(self, model_dir):
        return model_dir + "model.pth"

    def model_checkpoints(self, model_dir):
        return (glob.glob(model_dir + "checkpoints/*.pth"),)

    def epoch_from_model_checkpoint_name(self):
        return lambda x: int(
            os.path.basename(x).replace("model_epoch_", "").replace(".pth", "")
        )

    def visualize(self):
        images_dict = {
            line.strip(): [Image.open(line.strip()).convert('RGB')] for line in self.config.input
        }

        if self.config.checkpoints:
            models = sorted(
                glob.glob(self.config.model_dir + "checkpoints/*.pth"),
                key=self.epoch_from_model_checkpoint_name(),
            )
            sliced = slice(
                *[
                    None if x is "" else int(x)
                    for x in self.config.checkpoints.split(":")
                ]
            )
            models = models[sliced]

        else:
            models = [self.model_final(self.config.model_dir)]

        model_constr = supportedModels[config.model_type]

        print("Using models with {!s}:".format(model_constr()))
        for m in models:
            print("\t" + m)

        # we assume images have all the same size and need to be downscaled first
        target_size = list(images_dict.values())[0][0].size[0]
        if config.no_downscale:
            target_size = int(target_size*config.scale_factor)

        print("Scaling images to size {!s}:".format(target_size))
        print("\n".join(["\t" + img for img in images_dict.keys()]))

        if self.config.inc_lowres:
            proc = SRCNNPreProcessorGen(
                model_constr(),
                models[0],
                self.config.scale_factor,
                not config.no_downscale,
                use_gpu=torch.cuda.is_available(),
                target_size=target_size,
            )
            for key, images in images_dict.items():
                low_image = proc.lowres(images[0])
                images_dict[key].append((low_image))

        for model in models:
            proc = SRCNNPreProcessorGen(
                model_constr(),
                model,
                self.config.scale_factor,
                not config.no_downscale,
                use_gpu=torch.cuda.is_available(),
                target_size=target_size,
                srgan=config.model_type == 'srgan'
            )
            for key, images in images_dict.items():
                result_image = proc.apply(images[0])
                images_dict[key].append(result_image)

        if not self.config.inc_highres:
            images_dict = {key: value[1:] for key, value in images_dict.items()}

        def combine_to_single_image(images, labels: List[str] = []) -> Image:
            if config.columns:
                columns = config.columns
            else:
                columns = len(models) + len(
                    list(
                        filter(
                            lambda x: x,
                            [self.config.inc_lowres, self.config.inc_highres],
                        )
                    )
                )

            rows = math.ceil(len(images) / float(columns))
            height, width = rows * target_size, columns * target_size

            result = Image.new("RGB", (width, height))
            font = ImageFont.truetype("reports/B612Mono-Regular.ttf", 32)
            draw = ImageDraw.Draw(result)

            image_slots = [
                (w, h)
                for h, w in itertools.product(
                    range(0, target_size * rows, target_size),
                    range(0, target_size * columns, target_size),
                )
            ][
                : len(images)
            ]  # take only n slots

            for image, (w_index, h_index), label in itertools.zip_longest(
                images, image_slots, labels
            ):
                print(*image.size)
                result.paste(image, (w_index, h_index))
                if label:
                    draw.text((w_index, h_index), label, (255, 0, 0), font=font)

            return result

        def gen_labels(file, images):
            labels = list(len(images) * [""])

            labels[0] += "\n".join(os.path.basename(file).split("-")) + "\n"

            index = 0
            if config.inc_highres:
                labels[index] += "high_res "
                index += 1
            if config.inc_lowres:
                labels[index] += "low_res "
                index += 1
            if config.checkpoints:
                for i, model in enumerate(models, index):
                    labels[i] += "e" + str(
                        self.epoch_from_model_checkpoint_name()(model)
                    )

            return labels

        if not self.config.outdir:
            out_dir = self.config.model_dir + "results/"
        else:
            out_dir = self.config.outdir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if self.config.merge_to is not None:

            fin_images = []
            fin_labels = []
            for file, images in images_dict.items():
                fin_images.extend(images)
                if self.config.annotate:
                    fin_labels.extend(gen_labels(file, images))

            result_image = combine_to_single_image(fin_images, labels=fin_labels)
            result_image.save(out_dir + config.merge_to)
            print("Saved image to '{}'.".format(out_dir + config.merge_to))
        else:
            for file, images in images_dict.items():
                out_file = out_dir + os.path.basename(file)

                combine_params = [images]

                if config.annotate:
                    combine_params.append(gen_labels(file, images))

                result_image = combine_to_single_image(*combine_params)
                result_image.save(out_file)
            print("Saved images to '{}'.".format(out_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SRCNN visualizer")

    PytorchSRVisulizator.add_arguments_to(parser)
    config = parser.parse_args()
    print(config)

    visualizer = PytorchSRVisulizator(config)
    visualizer.visualize()
