from torch import nn

from typing import Any, Dict

import torch
from mmcv.cnn import build_norm_layer
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import build_backbone
from mmdet3d.models.builder import BACKBONES

__all__ = ["RadarFeatureNet", "RadarEncoder"]


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.
    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]
    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(
        max_num_shape
    )
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class RFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = "RFNLayer"
        self.last_vfe = last_layer
        
        self.units = out_channels

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)
        self.norm_cfg = norm_cfg

        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.norm = build_norm_layer(self.norm_cfg, self.units)[1]

    def forward(self, inputs):

        x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        torch.backends.cudnn.enabled = True
        x = F.relu(x)

        if self.last_vfe:
            x_max = torch.max(x, dim=1, keepdim=True)[0]
            return x_max
        else:
            return x


@BACKBONES.register_module()
class RadarFeatureNet(nn.Module):
    def __init__(
        self,
        in_channels=4,
        feat_channels=(64,),
        with_distance=False,
        voxel_size=(0.2, 0.2, 4),
        point_cloud_range=(0, -40, -3, 70.4, 40, 1),
        norm_cfg=None,
    ):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = "RadarFeatureNet"
        assert len(feat_channels) > 0

        self.in_channels = in_channels
        in_channels += 2
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        feat_channels = [in_channels] + list(feat_channels)
        rfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            rfn_layers.append(
                RFNLayer(
                    in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer
                )
            )
        self.rfn_layers = nn.ModuleList(rfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.pc_range = point_cloud_range

    def forward(self, features, num_voxels, coors):
        dtype = features.dtype
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 1].to(dtype).unsqueeze(1) * self.vx + self.x_offset
        )
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset
        )

        # normalize x,y,z to [0, 1]
        features[:, :, 0:1] = (features[:, :, 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        features[:, :, 1:2] = (features[:, :, 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        features[:, :, 2:3] = (features[:, :, 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        
        # Combine together feature decorations
        features_ls = [features, f_center]
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask
        
        # Forward pass through PFNLayers
        for rfn in self.rfn_layers:
            features = rfn(features)

        return features.squeeze()


@BACKBONES.register_module()
class RadarEncoder(nn.Module):
    def __init__(
        self,
        pts_voxel_encoder: Dict[str, Any],
        pts_middle_encoder: Dict[str, Any],
        pts_bev_encoder=None,
        **kwargs,
    ):
        super().__init__()
        self.pts_voxel_encoder = build_backbone(pts_voxel_encoder)
        self.pts_middle_encoder = build_backbone(pts_middle_encoder)
        if pts_bev_encoder is not None:
            self.pts_bev_encoder = build_backbone(pts_bev_encoder)
        else:
            self.pts_bev_encoder = None

    def forward(self, feats, coords, batch_size, sizes):
        x = self.pts_voxel_encoder(feats, sizes, coords)
        
        x = self.pts_middle_encoder(x, coords, batch_size)
        
        if self.pts_bev_encoder is not None:
            x = self.pts_bev_encoder(x)
        
        return x