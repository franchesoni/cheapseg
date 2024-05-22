_base_ = [
    "../_base_/datasets/ade20k.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_80k.py",
]

BATCH_SIZE = 16
NUM_WORKERS = 42
norm_cfg = dict(type="SyncBN", requires_grad=True)

crop_size = (518, 518)  # 37 x 37
# add data_preprocessor
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    size=crop_size,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=14),
)


model = dict(  # different
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type="DinoVisionTransformer",
        # out_indices=[8, 9, 10, 11],
        # out_indices=[0, 1, 2, 3],
        backbone_name="dinov2_vits14",
    ),
    decode_head=dict(
        type="BNHead",
        in_channels=[384],
        in_index=[3],
        input_transform="resize_concat",
        channels=384,
        dropout_ratio=0,
        num_classes=150,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)  # add
optim_wrapped = dict(
    _delete_=True,
    type="OptimWrapper",  # custom optimizer for dinov2 from their config
    optimizer=dict(type="AdamW", lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999)),
)
param_scheduler = [  # migrate
    dict(
        type="LinearLR",
        start_factor=1e-06,
        by_epoch=False,
        begin=0,
        end=1500,
    ),
    dict(
        type="PolyLR",
        power=1.0,
        eta_min=0.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    ),
]
# migrated to new format above ->
# lr_config = dict(  # called "param_scheduler"
#     policy='poly',
#     warmup='linear',
#     warmup_iters=1500,
#     warmup_ratio=1e-06,
#     power=1.0,
#     min_lr=0.0,
#     by_epoch=False)

# train pipeline will be inherited from the schedule and dataset base configs

data_root = 'data/ade/ADEChallengeData2016_ds_128'
train_dataloader = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, dataset=dict(data_root=data_root))
val_dataloader = dict(batch_size=1, num_workers=NUM_WORKERS, dataset=dict(data_root=data_root))
test_dataloader = val_dataloader

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=2000)
custom_hooks = [
    dict(type="EarlyStoppingHook", monitor="mIoU", strict=True, patience=6)
]

