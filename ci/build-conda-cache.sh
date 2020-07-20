#!/bin/sh
if [-d /opt/conda/envs/sr-gpu/]
then
	/opt/conda/bin/conda env update -f conda.yaml
else
	/opt/conda/bin/conda env create -f conda.yaml
fi
