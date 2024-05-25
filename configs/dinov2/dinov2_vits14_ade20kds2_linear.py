_base_ = [
    "dinov2_vits14_ade20kds1_linear.py",
]

BATCH_SIZE = 16
NUM_WORKERS = 42
data_root = 'data/ade/ADEChallengeData2016_ds_2'
train_dataloader = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, dataset=dict(data_root=data_root))
val_dataloader = dict(batch_size=1, num_workers=NUM_WORKERS, dataset=dict(data_root=data_root))
test_dataloader = val_dataloader

custom_hooks = [
    dict(type="EarlyStoppingHook", monitor="mIoU", strict=True, patience=3)
]

