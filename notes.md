# installation
created a conda environment with python 3.10
installed torch 2.1.2 (latest 2.1) with cuda support for 12.1 (we have cuda 12 here)
then followed the instructions in mmsegmentation docs to install, but had to change __init__.py to be tolerant with a mmcv version (see commits) and pip install a few packages (ftfy, regex)
then the demo worked :)

note that for mask2former mmdet needs to be installed (as in the readme), however, we will get an error that we solve by modifying: `mmcv_maximum_version = '2.2.1'  # modified` or downgrading mmcv (useless for kernel errors)
when training mask2former we get `error in somecudaoperation: no kernal image is available for execution on the device`, that's fine (for now). Well this might be the cause of getting 44 as performance instead of 52. It's too much of a difference. It was run with `CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/mask2former/1gpu_mask2former_swin-s_8xb2-160k_ade20k-512x512.py` 

we also run knet using `python tools/train.py configs/knet/knet-s3_swin-l_upernet_8xb2-adamw-80k_ade20k-512x512_singlegpu.py` and got great performance. Now we'll run with
`CUDA_VISIBLE_DEVICES=X python tools/train.py configs/knet/knet-s3_swin-l_upernet_8xb2-adamw-80k_ade20kdsX-512x512_singlegpu.py`
`CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/knet/knet-s3_swin-l_upernet_8xb2-adamw-80k_ade20kds4-512x512_singlegpu.py`




# get sorted list of scores 
run `python extract_tables.py` and look at `ade20_scores.md`
