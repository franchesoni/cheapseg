# installation
created a conda environment with python 3.10
installed torch 2.1.2 (latest 2.1) with cuda support for 12.1 (we have cuda 12 here)
then followed the instructions in mmsegmentation docs to install, but had to change __init__.py to be tolerant with a mmcv version (see commits) and pip install a few packages (ftfy, regex)
then the demo worked :)

note that for mask2former mmdet needs to be installed (as in the readme), however, we will get an error that we solve by modifying: `mmcv_maximum_version = '2.2.1'  # modified` or downgrading mmcv (useless for kernel errors)
when training mask2former we get `error in somecudaoperation: no kernal image is available for execution on the device`, that's fine (for now)



# get sorted list of scores 
run `python extract_tables.py` and look at `ade20_scores.md`
