_base_ = [
    'dinov2_vits14_ade20kds1_linear.py'
]

n_prototypes = 16
model = dict(
    backbone=dict(
        type='DinoVisionTransformer', 
        backbone_name='dinov2_vitb14_reg'
    ),
    decode_head=dict(type="KNNHead")
    # decode_head=dict(
    #     type="NormedLinear",
    #     in_channels=[768],
    #     channels=768,
    #     num_classes=150*n_prototypes,

    # )
)

WORKERS = 0
data_root = 'data/ade/ADEChallengeData2016_ds_1'
custom_hooks = []
train_dataloader = dict(batch_size=1, dataset=dict(data_root=data_root), num_workers=WORKERS, persistent_workers=False)

# create clean train dataloader
dataset_type = 'ADE20KDataset'
traintest_pipeline = [  # similar to test pipeline in that it has no augmentations but tsimilar to train pipeline in that targets are resized
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),  # min and max sizes
    dict(type='PackSegInputs')
]
test_dataloader = dict(
    batch_size=1,
    num_workers=WORKERS,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'),
        pipeline=traintest_pipeline))