_base_ = "knet-s3_swin-l_upernet_8xb2-adamw-80k_ade20k-512x512.py"

# In K-Net implementation we use batch size 2 per GPU as default
data_root = 'data/ade/ADEChallengeData2016_ds_4'
train_dataloader = dict(batch_size=16, num_workers=42, dataset=dict(data_root=data_root))
val_dataloader = dict(batch_size=1, num_workers=42, dataset=dict(data_root=data_root))
test_dataloader = val_dataloader

custom_hooks = [
    dict(type="EarlyStoppingHook", monitor="mIoU", strict=True, patience=3)
]
