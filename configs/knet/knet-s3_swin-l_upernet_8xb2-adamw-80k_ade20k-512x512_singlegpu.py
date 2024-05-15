_base_ = 'knet-s3_swin-l_upernet_8xb2-adamw-80k_ade20k-512x512.py'

# In K-Net implementation we use batch size 2 per GPU as default
train_dataloader = dict(batch_size=16, num_workers=42)
val_dataloader = dict(batch_size=1, num_workers=42)
test_dataloader = val_dataloader
