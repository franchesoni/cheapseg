_base_ = ['./mask2former_swin-s_8xb2-160k_ade20k-512x512.py']

train_dataloader = dict(batch_size=16, num_workers=42)