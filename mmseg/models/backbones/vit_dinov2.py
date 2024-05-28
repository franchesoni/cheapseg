import torch
from mmseg.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class DinoVisionTransformer(BaseModule):
    def __init__(self, backbone_name, init_cfg=None, **kwargs):
        assert backbone_name in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitb14_reg']
        super().__init__(init_cfg=init_cfg)
        self.dino = torch.hub.load('facebookresearch/dinov2', backbone_name)

    def forward(self, x):
        with torch.no_grad():
            out = self.dino.get_intermediate_layers(x=x, n=4, reshape=True)
        return out

    def init_weights(self, pretrained=None):
        pass

