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

# pipeline
```
        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            return data
```
`data_info` is `{'img_path': 'data/ade/ADEChallengeData2016_ds_1/images/training/ADE_train_00006996.jpg', 'seg_map_path': 'data/ade/ADEChallengeData2016_ds_1/annotations/training/ADE_train_00006996.png', 'label_map': None, 'reduce_zero_label': True, 'seg_fields': [], 'sample_idx': 6995}`

```
(Pdb) self.transforms
[LoadImageFromFile(ignore_empty=False, to_float32=False, color_type='color', imdecode_backend='cv2', backend_args=None), LoadAnnotations(reduce_zero_label=True, imdecode_backend='pillow', backend_args=None), RandomResize(scale=(2048, 512), ratio_range=(0.5, 2.0), resize_cfg={'type': 'Resize', 'keep_ratio': True}), RandomCrop(crop_size=(512, 512)), RandomFlip(prob=0.5, direction=horizontal), PhotoMetricDistortion(brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18), PackSegInputs(meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'reduce_zero_label'))]
```
1. load image, record shape
2. load ann
3. uniformly sample a ratio in (0.5, 2.0) and multiply the scale (2048, 512) by it
4. bilinear interpolate the image to this scale (same with label bu using nearest)
5. up to 10 times i. create crop, ii. count max class occurrece, iii. go to i if >0.75 iv. crop image and label
6. flip (img and ann) with 1/2 prob
7. define convert as `lambda convert x, alpha, beta: np.clip(x.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)`, then:
    7.1 with 1/2 prob use alpha=1 beta randomly sampled from [-32, 32) (brightness)
    7.2 with 1/2 prob change contrast now, else do it at the last step, which is beta=0, alpha in [0.5, 1.5) (contrast)
    7.3 with 1/2 prob (saturation)
        7.3.1 convert to hsv
        7.3.2 use convert over the saturation channel with alpha in [0.5, 1.5)
        7.3.3 convert back to rgb
    7.4 with 1/2 prob
        7.4.1 convert to hsv
        7.4.2 get hue offset as a random int in [-18, 18), add it to the hue channel
        7.4.3 make the hue channel mod 180 (this is done in the prev step)
        7.4.4 convert back to rgb
    7.5 do contrast if it wasn't done before.
8. convert to tensor
9. move to device
10. convert to float
11. mean std normalize
12. get max input size or max size as coming from the size_divisor param (we do the first in training)
13. pad the image with zeros on the bottom right until max size and the ann with 255
14. stack
15. forward dino + forward head (which)
16. stack gt
17. bilinear resize logits to seg label shape (16, 150, 518, 518)
18. squeeze gt (16, 518, 518)
19. compute loss `loss = F.cross_entropy(
        pred,
        label,
        weight=None,
        reduction='none',
        ignore_index=255)`
20. sum and divide by label.numel() (pytorch uses only non-ignored indices)

## validation
the transforms are:
```[LoadImageFromFile(ignore_empty=False, to_float32=False, color_type='color', imdecode_backend='cv2', backend_args=None), Resize(scale=(2048, 512), scale_factor=None, keep_ratio=True, clip_object_border=True), backend=cv2), interpolation=bilinear), LoadAnnotations(reduce_zero_label=True, imdecode_backend='pillow', backend_args=None), PackSegInputs(meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'reduce_zero_label'))]```

1. resize by a factor up to the given size
2. run data preprocessor as in training (stack padding to max size but with size divisor 14 now)
3. then run forward and upsample as usual, but take the argmax logit for the class prediction
4. evaluation is over the predictions using `IoUMetric.process` as described in `mmseg/evaluation/metrics/iou_metric.py`, but the predictions come one at a time.
        



