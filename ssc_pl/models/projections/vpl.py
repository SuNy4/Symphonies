import torch
import torch.nn as nn

from ..layers import DeformableTransformerLayer
from ..utils import (flatten_multi_scale_feats, get_level_start_index, index_fov_back_to_voxels,
                     nlc_to_nchw)


class VoxelProposalLayer(nn.Module):

    def __init__(self, embed_dims, scene_shape, num_heads=8, num_levels=3, num_points=4):
        super().__init__()
        self.attn = DeformableTransformerLayer(embed_dims, num_heads, num_levels, num_points)
        self.scene_shape = scene_shape

    def forward(self, scene_embed, feats, scene_pos=None, vol_pts=None, ref_pix=None, fov_mask=None):
        bs, ncam, voxel_size = fov_mask.shape
        # Generate voxel map based on depth information
        keep = ((vol_pts[..., 0] >= 0) & (vol_pts[..., 0] < self.scene_shape[0]) &
                (vol_pts[..., 1] >= 0) & (vol_pts[..., 1] < self.scene_shape[1]) &
                (vol_pts[..., 2] >= 0) & (vol_pts[..., 2] < self.scene_shape[2]))
        #assert vol_pts.shape[0] == 1

        #keep = keep.reshape(bs, -1)
        #geom = vol_pts.squeeze()[keep.squeeze()]
        pts_mask = []
        for i in range(len(keep)):
            geom = vol_pts[i, keep[i]]
            pts_mask_batch = torch.zeros(self.scene_shape, device=scene_embed.device, dtype=torch.bool)
            pts_mask_batch[geom[:, 0], geom[:, 1], geom[:, 2]] = True
            pts_mask.append(pts_mask_batch.unsqueeze(0))
        pts_mask = torch.cat(pts_mask, dim=0)
        pts_mask = pts_mask.flatten(1)

        # bs*ncam, x*y*z
        proposal = (pts_mask.unsqueeze(1) * fov_mask).flatten(start_dim=0, end_dim=1)

        val_index = []
        for ex_bs in range(bs*ncam):
            valid_indices = torch.nonzero(proposal[ex_bs], as_tuple=False).squeeze()
            val_index.append(valid_indices)
        max_len = max([len(view) for view in val_index])

        # for batch in range(bs):
        #     view_indexes = [view[view] for _, view in enumerate(proposal[batch])]
        #     index.append(view_indexes)
        # max_len = max([len(view) for batch in index for view in batch])
        
        # # bs*ncam, x*y*z, dims
        # scene_embed = scene_embed.unsqueeze(1).expand(-1, ncam, -1, -1).flatten(start_dim=0, end_dim=1)
        # if scene_pos is not None:
        #     scene_pos = scene_pos.unsqueeze(1).expand(-1, ncam, -1, -1).flatten(start_dim=0, end_dim=1)
        # else:
        #     scene_pos = None

        # bs, ncam, x*y*z, 2(pixel coord)
        ref_pix = ref_pix.flatten(start_dim=0, end_dim=1)

        scene_embed_cam = scene_embed.new_zeros(
            [bs*ncam, max_len, scene_embed.shape[-1]]
        )
        scene_pos_cam = scene_pos.new_zeros(
            [bs*ncam, max_len, scene_pos.shape[-1]]
        )
        ref_pix_cam = ref_pix.new_zeros(
            [bs*ncam, max_len, ref_pix.shape[-1]]
        )

        count = 0
        for batch in range(bs):
            for _ in range(ncam):
                scene_embed_cam[count, :len(val_index[count])] = scene_embed[batch, val_index[count]]
                scene_pos_cam[count, :len(val_index[count])] = scene_pos[batch, val_index[count]]
                ref_pix_cam[count, :len(val_index[count])] = ref_pix[batch, val_index[count]]
                count += 1
        assert count == bs*ncam

        # bs*Ncam, sum(HW), embed_dim*6 ->  bs, Ncam, sum(HW), embed_dim
        feat_flatten, shapes = flatten_multi_scale_feats(feats)
        #feat_flatten = feat_flatten.reshape(bs, -1, feat_flatten.shape[-2], feat_flatten.shape[-1])


        ###############################################################

        pts_embed_cam = self.attn(
            scene_embed_cam, #[:, pts_mask],
            feat_flatten,
            query_pos=scene_pos_cam if scene_pos_cam is not None else None, #[:, pts_mask] 
            ref_pts=ref_pix_cam.unsqueeze(2).expand(-1, -1, len(feats), -1), #[:, pts_mask]
            spatial_shapes=shapes,
            level_start_index=get_level_start_index(shapes))
        
        count = 0
        pts_embed = pts_embed_cam.new_zeros(bs, voxel_size, pts_embed_cam.shape[-1])
        for batch in range(bs):
            for _ in range(ncam):
                ### Aggregation of multi view, 어떻게 합칠건지 더 봐야함
                pts_embed[batch, val_index[count]] = pts_embed_cam[count, :len(val_index[count])]
                count += 1
        assert count == bs*ncam

        x3d = nlc_to_nchw(scene_embed, self.scene_shape)
        proposal = torch.where(pts_mask.unsqueeze(1), pts_embed.transpose(1,2), x3d.flatten(2)).reshape(*x3d.shape)
        
        return proposal
    
        return index_fov_back_to_voxels(
            nlc_to_nchw(scene_embed, self.scene_shape), pts_embed, pts_mask)
