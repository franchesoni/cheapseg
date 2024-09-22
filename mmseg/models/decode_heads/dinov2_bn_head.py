# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from ..utils import resize
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class BNHead(BaseDecodeHead):
    """Just a batchnorm."""

    def __init__(self, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        assert self.in_channels == self.channels
        self.bn = nn.SyncBatchNorm(self.in_channels)
        self.resize_factors = resize_factors

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # print("inputs", [i.shape for i in inputs])
        x = self._transform_inputs(inputs)
        # print("x", x.shape)
        if x.shape[0] > 1:  # batch norm breaks with batch size 1
            feats = self.bn(x)
        else:
            feats = x
        # print("feats", feats.shape)
        return feats

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """
        # we add this extra pathway to process individual feature vectors
        if isinstance(inputs, torch.Tensor) and len(inputs.shape) == 2:  # (B, F)
            return inputs[:, :, None, None]  # (B, F, 1, 1)


        if self.input_transform == "resize_concat":
            # accept lists (for cls token)
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list
            # an image descriptor can be a local descriptor with resolution 1x1
            for i, x in enumerate(inputs):
                if len(x.shape) == 2:
                    inputs[i] = x[:, :, None, None]
            # select indices
            inputs = [inputs[i] for i in self.in_index]
            # Resizing shenanigans
            # print("before", *(x.shape for x in inputs))
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs), (len(self.resize_factors), len(inputs))
                inputs = [
                    resize(input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area")
                    for x, f in zip(inputs, self.resize_factors)
                ]
                # print("after", *(x.shape for x in inputs))
            upsampled_inputs = [
                resize(input=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output


@MODELS.register_module()
class NormedLinear(BaseDecodeHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.in_channels == self.channels
        self.conv_seg = nn.utils.parametrizations.weight_norm(nn.Conv2d(self.channels, self.out_channels, kernel_size=1, bias=False))
        with torch.no_grad():
            self.conv_seg.parametrizations.weight.original0.fill_(1.0)
        self.conv_seg.parametrizations.weight.original0.requires_grad = False
        self.bn = lambda x: x


class DynamicTensor:
    def __init__(self, initial_tensor):
        assert initial_tensor.ndim == 1 or initial_tensor.ndim == 2, "Tensor must be 1D or 2D"
        self.tensor = initial_tensor
        self.current_len = len(self.tensor)
        self.current_buffer_size = 2**torch.log2(torch.tensor(self.current_len)).ceil().int().item() if self.current_len > 0 else 1
        if self.current_buffer_size == 1 and self.current_len == 0:
            self.tensor = torch.empty(1, *self.tensor.shape[1:], dtype=self.tensor.dtype)

    def append(self, x):
        # Ensure `x` has compatible shape
        if self.tensor.ndim == 2:
            assert x.shape == self.tensor.shape[1:], f"Expected shape {self.tensor.shape[1:]}, got {x.shape}"
        elif self.tensor.ndim == 1:
            assert x.shape == (), "Expected scalar for 1D tensor"

        if self.current_len == self.current_buffer_size:
            # Allocate twice the memory
            new_buffer = torch.zeros(self.current_buffer_size * 2, *self.tensor.shape[1:], dtype=self.tensor.dtype)
            new_buffer[:self.current_len] = self.tensor
            self.tensor = new_buffer
            self.current_buffer_size *= 2
        
        self.tensor[self.current_len] = x
        self.current_len += 1

    def get_tensor(self):
        return self.tensor[:self.current_len]

    def __len__(self):
        return self.current_len

    def to(self, *args, **kwargs):
        self.tensor = self.tensor.to(*args, **kwargs)
        return self


@MODELS.register_module()
class KNNHead:
    def __init__(self, k=1, **kwargs):
        self.feats = DynamicTensor(torch.empty((0, 768)))
        self.labels = []
        self.align_corners = False
        self.num_classes = 150
        self.out_channels = 150
        self.k = k

    def _stack_batch_gt(self, *args, **kwargs):
        return BaseDecodeHead._stack_batch_gt(None, *args, **kwargs)

    def forward(self, inputs):
        if len(self.feats) == 0:
            # no knowledge
            return torch.zeros(1, self.num_classes, *inputs.shape[2:], device=inputs.device)
        # inputs is (1, F, H, W)
        B, F, H, W = inputs.shape
        # feats is (N, F)
        similarities = self.feats.get_tensor() @ inputs.permute(1, 0, 2, 3).reshape(F, -1)  # (N, BHW)
        top_k_similarities, top_k_indices = torch.topk(similarities, self.k, dim=0)
        top_k_labels = torch.tensor(self.labels, device=inputs.device)[top_k_indices]  # (k, BHW)
        top_k_labels = top_k_labels.reshape(self.k, B, H, W)
        predicted_labels, _ = torch.mode(top_k_labels, dim=0)  # (B, H, W)

        prediction = torch.nn.functional.one_hot(predicted_labels, num_classes=self.num_classes)  # (B, H, W, C)
        return prediction.permute(0, 3, 1, 2).float()

        # results:
        # 0.4611 err rate at 14k 

    def predict(self, feats, *args):
        feats = feats[3] / torch.norm(feats[3], dim=1, keepdim=True)  # normalize [1, 768, 37, 37]
        prediction = self.forward(feats)
        return torch.nn.functional.interpolate(prediction, size=(518, 518), mode='bilinear', align_corners=False)

    def append(self, x, y):
        self.feats.append(x)
        self.feats.to(x.device)
        self.labels.append(y)

    def init_weights(self, *args, **kwargs):
        pass



