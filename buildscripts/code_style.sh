#!/bin/bash

source activate $CONDA_ENV

# echo "======================================================================"
# echo Executing clang-format-6.0 style:
# echo "======================================================================"
# python ./setup.py style

echo "======================================================================"
echo Executing python flake8:
echo "======================================================================"
flake8 ./