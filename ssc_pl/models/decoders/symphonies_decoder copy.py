import copy
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from ..layers import (ASPP, DeformableSqueezeAttention, DeformableTransformerLayer,
                      LearnableSqueezePositionalEncoding, TransformerLayer, Upsample)
from ..projections import VoxelProposalLayer
from ..utils import (cumprod, flatten_fov_from_voxels, flatten_multi_scale_feats, generate_grid,
                     get_level_start_index, index_fov_back_to_voxels, interpolate_flatten,
                     nchw_to_nlc, nlc_to_nchw, pix2vox)


class SymphoniesLayer(nn.Module):

    def __init__(self, embed_dims, num_heads=8, num_levels=3, num_points=4, query_update=True):
        super().__init__()
        self.query_image_cross_defrom_attn = DeformableTransformerLayer(
            embed_dims, num_heads, num_levels, num_points)
        self.scene_query_cross_attn = TransformerLayer(embed_dims, num_heads, mlp_ratio=0)
        self.scene_self_deform_attn = DeformableTransformerLayer(
            embed_dims,
            num_heads,
            num_levels=1,
            num_points=num_points * 2,
            attn_layer=DeformableSqueezeAttention)

        self.query_update = query_update
        if query_update:
            self.query_scene_cross_deform_attn = DeformableTransformerLayer(
                embed_dims,
                num_heads,
                num_levels=1,
                num_points=num_points * 2,
                attn_layer=DeformableSqueezeAttention,
                mlp_ratio=0)
            self.query_self_attn = TransformerLayer(embed_dims, num_heads)

    def forward(self,
                scene_embed,
                inst_queries,
                feats,
                scene_pos=None,
                inst_pos=None,
                ref_2d=None,
                ref_3d=None,
                ref_vox=None,
                fov_mask=None):
        scene_embed_fov = flatten_fov_from_voxels(scene_embed, fov_mask)
        scene_pos_fov = flatten_fov_from_voxels(scene_pos,
                                                fov_mask) if scene_pos is not None else None
        scene_embed_flatten, scene_shape = flatten_multi_scale_feats([scene_embed])
        scene_level_index = get_level_start_index(scene_shape)

        feats_flatten, feat_shapes = flatten_multi_scale_feats(feats)
        feats_level_index = get_level_start_index(feat_shapes)

        inst_queries = self.query_image_cross_defrom_attn(
            inst_queries,
            feats_flatten,
            query_pos=inst_pos,
            ref_pts=ref_2d,
            spatial_shapes=feat_shapes,
            level_start_index=feats_level_index)

        scene_embed_fov = self.scene_query_cross_attn(scene_embed_fov, inst_queries, inst_queries,
                                                      scene_pos_fov, inst_pos)
        scene_embed_fov = self.scene_self_deform_attn(
            scene_embed_fov,
            scene_embed_flatten,
            query_pos=scene_pos_fov,
            ref_pts=torch.flip(ref_vox[:, fov_mask.squeeze()], dims=[-1]),  # TODO: assert bs == 1
            spatial_shapes=scene_shape,
            level_start_index=scene_level_index)

        scene_embed = index_fov_back_to_voxels(scene_embed, scene_embed_fov, fov_mask)
        scene_embed_flatten, scene_shape = flatten_multi_scale_feats([scene_embed])
        if not self.query_update:
            return scene_embed, inst_queries

        inst_queries = self.query_scene_cross_deform_attn(
            inst_queries,
            scene_embed_flatten,
            query_pos=inst_pos,
            ref_pts=torch.flip(ref_3d, dims=[-1]),
            spatial_shapes=scene_shape,
            level_start_index=scene_level_index)
        inst_queries = self.query_self_attn(inst_queries, query_pos=inst_pos)
        return scene_embed, inst_queries


class SymphoniesDecoder(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_classes,
                 num_layers,
                 num_levels,
                 scene_shape,
                 project_scale,
                 image_shape,
                 voxel_size=0.2,
                 downsample_z=1):
        super().__init__()
        self.embed_dims = embed_dims
        scene_shape = [s // project_scale for s in scene_shape]
        if downsample_z != 1:
            self.ori_scene_shape = copy.copy(scene_shape)
            scene_shape[-1] //= downsample_z
        self.scene_shape = scene_shape
        self.num_queries = cumprod(scene_shape)
        self.image_shape = image_shape
        self.voxel_size = voxel_size * project_scale
        self.downsample_z = downsample_z

        self.voxel_proposal = VoxelProposalLayer(embed_dims, scene_shape)
        self.layers = nn.ModuleList([
            SymphoniesLayer(
                embed_dims,
                num_levels=num_levels,
                query_update=True if i != num_layers - 1 else False) for i in range(num_layers)
        ])

        self.scene_embed = nn.Embedding(self.num_queries, embed_dims)
        self.scene_pos = LearnableSqueezePositionalEncoding((100, 100, 2),
                                                            embed_dims,
                                                            squeeze_dims=(2, 2, 1))

        image_grid = generate_grid(image_shape)
        image_grid = torch.flip(image_grid, dims=[0]).unsqueeze(0)  # 2(wh), h, w
        self.register_buffer('image_grid', image_grid)
        voxel_grid = generate_grid(scene_shape, normalize=True)
        self.register_buffer('voxel_grid', voxel_grid)

        self.aspp = ASPP(embed_dims, (1, 3))
        assert project_scale in (1, 2)
        self.cls_head = nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose3d(
                    embed_dims,
                    embed_dims,
                    kernel_size=3,
                    stride=(1, 1, downsample_z),
                    padding=1,
                    output_padding=(0, 0, downsample_z - 1),
                ),
                nn.BatchNorm3d(embed_dims),
                nn.ReLU(),
            ) if downsample_z != 1 else nn.Identity(),
            Upsample(embed_dims, embed_dims) if project_scale == 2 else nn.Identity(),
            nn.Conv3d(embed_dims, num_classes, kernel_size=1))

    @autocast(dtype=torch.float32)
    def forward(self, pred_insts, feats, pred_masks, depth, K, E, voxel_origin, projected_pix,
                fov_mask):
        ###### Create Voxel pts from depth info (bs, HW*Ncam, 3)
        bs = pred_insts[0]['queries'].shape[0]
        vol_pts = []
        inst_queries = []
        inst_pos = []
        ref_2d = []
        ref_3d = []

        for i in range(len(depth)):
            vol_pts_cam = pix2vox(
                self.image_grid,
                depth[i].unsqueeze(1),
                K[i],
                E[i],
                voxel_origin[i],
                self.voxel_size,
                downsample_z=self.downsample_z).long()
            inst_queries_cam = pred_insts[i]['queries']    # bs, n*ncam, c
            inst_pos_cam = pred_insts[i].get('query_pos', None)
            ref_2d_cam = pred_insts[i]['pred_pts'].unsqueeze(2).expand(-1, -1, len(feats[i]), -1)
            ref_3d_cam = self.generate_vol_ref_pts_from_masks(
                pred_insts[i]['pred_boxes'], pred_masks[i],
                vol_pts).unsqueeze(2) if pred_masks[i] else self.generate_vol_ref_pts_from_pts(
                    pred_insts[i]['pred_pts'], vol_pts_cam).unsqueeze(2)

            if self.downsample_z != 1:
                projected_pix[i] = interpolate_flatten(
                    projected_pix[i], self.ori_scene_shape, self.scene_shape, mode='trilinear')
                fov_mask[i] = interpolate_flatten(
                    fov_mask[i], self.ori_scene_shape, self.scene_shape, mode='trilinear')
                
            ref_pix_cam = (torch.flip(projected_pix[i], dims=[-1]) + 0.5) / torch.tensor(
                self.image_shape).to(projected_pix[i])
            ref_pix_cam = torch.flip(ref_pix_cam, dims=[-1])
            ref_vox_cam = nchw_to_nlc(self.voxel_grid.unsqueeze(0)).unsqueeze(2)
            
            vol_pts.append(vol_pts_cam)
            inst_queries.append(inst_queries_cam)
            inst_pos.append(inst_pos_cam)
            ref_2d.append(ref_2d_cam)
            ref_3d.append(ref_3d_cam)
        
        
        vol_pts = torch.cat(vol_pts, dim=1)
        inst_queries = torch.cat(inst_queries, dim=1)
        inst_pos = torch.cat(inst_pos, dim=1)
        ref_2d = torch.cat(ref_2d, dim=1)
        ref_3d = torch.cat(ref_3d, dim=1)
        projected_pix = torch.cat(projected_pix, dim=1)
        fov_mask = torch.cat(fov_mask, dim=1)

        # Adjust floor level for multi-view
        z_min = torch.min(vol_pts[:, :, 2])
        vol_pts[:, :, 2] -= z_min





        scene_embed = self.scene_embed.weight.repeat(bs, 1, 1)
        scene_pos = self.scene_pos().repeat(bs, 1, 1)
        #######################################################
        scene_embed = self.voxel_proposal(scene_embed, feats, scene_pos, vol_pts, ref_pix)
        #######################################################
        scene_pos = nlc_to_nchw(scene_pos, self.scene_shape)

        outs = []
        for i, layer in enumerate(self.layers):
            scene_embed, inst_queries = layer(scene_embed, inst_queries, feats, scene_pos, inst_pos,
                                              ref_2d, ref_3d, ref_vox, fov_mask)
            if i == 2:
                scene_embed = self.aspp(scene_embed)
            if self.training or i == len(self.layers) - 1:
                outs.append(self.cls_head(scene_embed))
        return outs




    def generate_vol_ref_pts_from_masks(self, pred_boxes, pred_masks, vol_pts):
        pred_boxes *= torch.tensor((self.image_shape + self.image_shape)[::-1]).to(pred_boxes)
        pred_pts = pred_boxes[..., :2].int()
        cx, cy, w, h = pred_boxes.split((1, 1, 1, 1), dim=-1)
        pred_boxes = torch.cat([(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)],
                               dim=-1).int()
        pred_boxes[0::2] = pred_boxes[0::2].clamp(0, self.image_shape[1] - 1)
        pred_boxes[1::2] = pred_boxes[1::2].clamp(1, self.image_shape[1] - 1)

        pred_masks = F.interpolate(
            pred_masks.float(), self.image_shape, mode='bilinear').to(pred_masks.dtype)
        bs, n = pred_masks.shape[:2]

        for b, i in product(range(bs), range(n)):
            if pred_masks[b, i].sum().item() != 0:
                continue
            boxes = pred_boxes[b, i]
            pred_masks[b, i, boxes[1]:boxes[3], boxes[0]:boxes[2]] = True
            if pred_masks[b, i].sum().item() != 0:
                continue
            pred_masks[b, i, pred_pts[b, i, 1], pred_pts[b, i, 0]] = True
        pred_masks = pred_masks.flatten(2).unsqueeze(-1).to(vol_pts)  # bs, n, hw, 1
        vol_pts = vol_pts.unsqueeze(1) * pred_masks  # bs, n, hw, 3
        vol_pts = vol_pts.sum(dim=2) / pred_masks.sum(dim=2) / torch.tensor(
            self.scene_shape).to(vol_pts)
        return vol_pts.clamp(0, 1)

    def generate_vol_ref_pts_from_pts(self, pred_pts, vol_pts):
        pred_pts = pred_pts * torch.tensor(self.image_shape[::-1]).to(pred_pts)
        pred_pts = pred_pts.long()
        pred_pts = pred_pts[..., 1] * self.image_shape[1] + pred_pts[..., 0]
        assert pred_pts.size(0) == 1
        ref_pts = vol_pts[:, pred_pts.squeeze()]
        ref_pts = ref_pts / (torch.tensor(self.scene_shape) - 1).to(pred_pts)
        return ref_pts.clamp(0, 1)
