_base_ = [
    'dinov2_vits14_ade20k_linear.py'
]

model = dict(
    backbone=dict(
        type='DinoVisionTransformer', 
        backbone_name='dinov2_vitb14'
    ),
    decode_head=dict(
        in_channels=[768],
        channels=768
    )
)
