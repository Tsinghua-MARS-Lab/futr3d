import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
from mmcv.runner.base_module import BaseModule
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE
from mmdet.models.utils.builder import TRANSFORMER


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@ATTENTION.register_module()
class FUTR3DAttention(BaseModule):
    """An attention module used in Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 use_lidar=True,
                 use_camera=False,
                 use_radar=False,
                 embed_dims=256,
                 radar_dims=64,
                 num_cams=6,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 pc_range=None,
                 rad_cuda=True,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.use_lidar = use_lidar
        self.use_camera = use_camera
        self.use_radar = use_radar
        self.pc_range = pc_range
        self.num_cams = num_cams
        self.rad_cuda = rad_cuda

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.fused_embed = 0
        if self.use_lidar:
            self.sampling_offsets = nn.Linear(
                embed_dims, num_heads * num_levels * num_points * 2)
            self.attention_weights = nn.Linear(embed_dims,
                                            num_heads * num_levels * num_points)
            self.value_proj = nn.Linear(embed_dims, embed_dims)
            self.output_proj = nn.Linear(embed_dims, embed_dims)
            self.fused_embed += embed_dims
        
        if self.use_camera:
            self.img_attention_weights = nn.Linear(embed_dims,
                                           num_cams*num_levels)

            self.img_output_proj = nn.Linear(embed_dims, embed_dims)
        
            self.position_encoder = nn.Sequential(
                nn.Linear(3, self.embed_dims), 
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims), 
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
            )
            self.weight_dropout = nn.Dropout(0.0)
            self.fused_embed += embed_dims
        if self.use_radar:
            self.rad_sampling_offsets = nn.Linear(
                embed_dims, num_heads * num_points * 2)
            self.rad_attention_weights = nn.Linear(embed_dims,
                                            num_heads * num_points)
            self.rad_value_proj = nn.Linear(radar_dims, radar_dims)
            self.rad_output_proj = nn.Linear(radar_dims, radar_dims)
            self.fused_embed += radar_dims

        if self.fused_embed > embed_dims:
            self.modality_fusion_layer = nn.Sequential(
                nn.Linear(self.fused_embed, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=False),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
            )

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        
        device = next(self.parameters()).device
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        if self.use_lidar:
            constant_init(self.sampling_offsets, 0.)
            self.sampling_offsets.bias.data = grid_init.view(-1)
            constant_init(self.attention_weights, val=0., bias=0.)
            xavier_init(self.value_proj, distribution='uniform', bias=0.)
            xavier_init(self.output_proj, distribution='uniform', bias=0.)
        if self.use_camera:
            constant_init(self.img_attention_weights, val=0., bias=0.)
            xavier_init(self.img_output_proj, distribution='uniform', bias=0.)
        if self.use_radar:
            constant_init(self.rad_sampling_offsets, 0.)
            self.rad_sampling_offsets.bias.data = grid_init[:, 0].reshape(-1)
            constant_init(self.rad_attention_weights, val=0., bias=0.)
            xavier_init(self.rad_value_proj, distribution='uniform', bias=0.)
            xavier_init(self.rad_output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                pts_feats=None,
                img_feats=None,
                rad_feats=None,
                rad_key_padding_mask=None,
                rad_spatial_shapes=None,
                rad_level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        if self.use_lidar:
            value = pts_feats
            bs, num_value, _ = value.shape
            
            assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

            value = self.value_proj(value)
            if key_padding_mask is not None:
                value = value.masked_fill(key_padding_mask[..., None], 0.0)
            value = value.view(bs, num_value, self.num_heads, -1)
            sampling_offsets = self.sampling_offsets(query).view(
                bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
            attention_weights = self.attention_weights(query).view(
                bs, num_query, self.num_heads, self.num_levels * self.num_points)
            attention_weights = attention_weights.softmax(-1)

            attention_weights = attention_weights.view(bs, num_query,
                                                    self.num_heads,
                                                    self.num_levels,
                                                    self.num_points)
            ref_points = reference_points.unsqueeze(2).expand(-1, -1, self.num_levels, -1)
            # ref_points = reference_points
            ref_points = ref_points[..., :2]
            if ref_points.shape[-1] == 2:
                offset_normalizer = torch.stack(
                    [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
                sampling_locations = ref_points[:, :, None, :, None, :] \
                    + sampling_offsets \
                    / offset_normalizer[None, None, None, :, None, :]
            else:
                raise ValueError(
                    f'Last dim of reference_points must be'
                    f' 2, but get {reference_points.shape[-1]} instead.')
            if ((IS_CUDA_AVAILABLE and value.is_cuda)
                    or (IS_MLU_AVAILABLE and value.is_mlu)):
                output = MultiScaleDeformableAttnFunction.apply(
                    value, spatial_shapes, level_start_index, sampling_locations,
                    attention_weights, self.im2col_step)
            else:
                output = multi_scale_deformable_attn_pytorch(
                    value, spatial_shapes, sampling_locations, attention_weights)

            pts_output = self.output_proj(output)

        if self.use_camera:
            # (B, 1, num_query, num_cams, num_points, num_levels)
            img_attention_weights = self.img_attention_weights(query).view(
                bs, 1, num_query, self.num_cams, 1, self.num_levels)
            # reference_points: (bs, num_query, 3)
            # output (B, C, num_query, num_cam, num_points, len(lvl_feats))
            reference_points_3d, img_output, mask = feature_sampling(
                img_feats, reference_points, self.pc_range, kwargs['img_metas'])
            img_output = torch.nan_to_num(img_output)
            mask = torch.nan_to_num(mask)

            img_attention_weights = self.weight_dropout(img_attention_weights.sigmoid()) * mask
            img_output = img_output * img_attention_weights
            # output (B, emb_dims, num_query)
            img_output = img_output.sum(-1).sum(-1).sum(-1)
            # output (num_query, B, emb_dims)
            img_output = img_output.permute(0, 2, 1)

            img_output = self.img_output_proj(img_output)

        if self.use_radar:
            value = rad_feats
            bs, num_value, _ = value.shape
            
            assert (rad_spatial_shapes[:, 0] * rad_spatial_shapes[:, 1]).sum() == num_value

            value = self.rad_value_proj(value)
            if rad_key_padding_mask is not None:
                value = value.masked_fill(rad_key_padding_mask[..., None], 0.0)
            value = value.view(bs, num_value, self.num_heads, -1)
            
            sampling_offsets = self.rad_sampling_offsets(query).view(
                bs, num_query, self.num_heads, 1, self.num_points, 2)
            attention_weights = self.rad_attention_weights(query).view(
                bs, num_query, self.num_heads, self.num_points)
            attention_weights = attention_weights.softmax(-1)

            attention_weights = attention_weights.view(bs, num_query,
                                                    self.num_heads,
                                                    1,
                                                    self.num_points)
            ref_points = reference_points.unsqueeze(2).expand(-1, -1, 1, -1)
            # ref_points = reference_points
            ref_points = ref_points[..., :2]
            if ref_points.shape[-1] == 2:
                offset_normalizer = torch.stack(
                    [rad_spatial_shapes[..., 1], rad_spatial_shapes[..., 0]], -1)
                sampling_locations = ref_points[:, :, None, :, None, :] \
                    + sampling_offsets \
                    / offset_normalizer[None, None, None, :, None, :]
            else:
                raise ValueError(
                    f'Last dim of reference_points must be'
                    f' 2, but get {reference_points.shape[-1]} instead.')
            if ((IS_CUDA_AVAILABLE and value.is_cuda)
                    or (IS_MLU_AVAILABLE and value.is_mlu)):
                output = MultiScaleDeformableAttnFunction.apply(
                    value, rad_spatial_shapes, rad_level_start_index, sampling_locations,
                    attention_weights, self.im2col_step)
            else:
                output = multi_scale_deformable_attn_pytorch(
                    value, rad_spatial_shapes, sampling_locations, attention_weights)

            radar_output = self.rad_output_proj(output)
            
        if self.use_lidar and self.use_camera and self.use_radar:
            output = torch.cat((img_output, pts_output, radar_output), dim=2)
            output = self.modality_fusion_layer(output)
        elif self.use_lidar and self.use_camera:
            output = torch.cat((img_output, pts_output), dim=2)
            output = self.modality_fusion_layer(output)
        elif self.use_camera and self.use_radar:
            output = torch.cat((img_output, radar_output), dim=2)
            output = self.modality_fusion_layer(output)
        elif self.use_lidar:
            output = pts_output
        elif self.use_camera:
            output = img_output
        elif self.use_radar:
            output = img_output

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    # reference_points (B, num_queries, 4)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0) 
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > -1.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    mask = torch.nan_to_num(mask)
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B*N, C, H, W)
        reference_points_cam_lvl = reference_points_cam.view(B*N, num_query, 1, 2)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam,  1, len(mlvl_feats))
    return reference_points_3d, sampled_feats, mask

