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

data_root = 'data/ade/ADEChallengeData2016_ds_1'
train_dataloader = dict(batch_size=1, dataset=dict(data_root=data_root), num_workers=42, persistent_workers=False)
custom_hooks = []