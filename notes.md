# installation
created a conda environment with python 3.10
installed torch 2.1.2 (latest 2.1) with cuda support for 12.1 (we have cuda 12 here)
then followed the instructions in mmsegmentation docs to install, but had to change __init__.py to be tolerant with a mmcv version (see commits) and pip install a few packages (ftfy, regex)
then the demo worked :)