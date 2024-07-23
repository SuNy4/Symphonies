import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ... import build_from_configs
from .. import encoders
from ..decoders import SymphoniesDecoder
from ..losses import ce_ssc_loss, frustum_proportion_loss, geo_scal_loss, sem_scal_loss


class Symphonies(nn.Module):

    def __init__(
        self,
        encoder,
        embed_dims,
        scene_size,
        view_scales,
        volume_scale,
        num_classes,
        num_layers=3,
        num_queries=100,
        image_shape=(900, 1600),
        voxel_size=0.2,
        downsample_z=2,
        class_weights=None,
        criterions=None,
        **kwargs,
    ):
        super().__init__()
        self.volume_scale = volume_scale
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.criterions = criterions

        self.encoder = build_from_configs(
            encoders, encoder, embed_dims=embed_dims, num_queries=num_queries, scales=view_scales)
        self.decoder = SymphoniesDecoder(
            embed_dims,
            num_classes,
            num_layers=num_layers,
            num_levels=len(view_scales),
            scene_shape=scene_size,
            project_scale=volume_scale,
            image_shape=image_shape,
            voxel_size=voxel_size,
            downsample_z=downsample_z)
        
        #self.transforms = transforms.Compose([
        #to_pil = transforms.ToPILImage(),  # Convert numpy array to PIL image
        #resize = transforms.Resize((800, 800)),  # Resize image
        #tensor = transforms.ToTensor()  # Convert PIL image to PyTorch tensor and normalize
        #])
            
    def forward(self, inputs):
                #self.transforms = transforms.Compose([
        to_pil = transforms.ToPILImage(),  # Convert numpy array to PIL image
        resize = transforms.Resize((800, 800)),  # Resize image
        tensor = transforms.ToTensor()  # Convert PIL image to PyTorch tensor and normalize
        #])
        # For NuScenes Occ3D
        model_memory = torch.cuda.memory_allocated() / 1024**2
        if inputs['cam_channels']:
            # pred_insts = []
            # pred_masks = []
            # feats = []
            input_img = []
            depth = []
            K = []
            E = []
            voxel_origin = []
            projected_pix = []
            fov_mask = []
            scenes = []
            frames = []

            ncam = 0
            for cam in inputs['cam_channels']:
                # model_memory = torch.cuda.memory_allocated() / 1024**2
                # input_img = inputs['cam_channels'][cam]['img']
                # print(input_img.shape)
                # for bs in range(input_img.shape[0]):
                #     print(input_img[bs].shape)
                #     input_img[bs] = resize(input_img[bs ])
                # pred_insts_cam = self.encoder(input_img)
                # encoder_memory = torch.cuda.memory_allocated() / 1024**2
                # print(encoder_memory - model_memory)
                # pred_masks_cam = pred_insts_cam.pop('pred_masks', None)
                # feats_cam = pred_insts_cam.pop('feats')
                cam_img = inputs['cam_channels'][cam]['img']
                depth_cam, K_cam, E_cam, voxel_origin_cam, projected_pix_cam, fov_mask_cam = list(
                    map(lambda k: inputs['cam_channels'][cam][k], 
                        ('depth', 'cam_K', 'cam_pose', 'voxel_origin', 
                        f'projected_pix_{self.volume_scale}', f'fov_mask_{self.volume_scale}')))
                ###################
                scene = inputs['scene']
                frame_id = inputs['frame_id']
                #################
                scenes.append(scene)
                frames.append(frame_id)
                ###################

                # pred_insts.append(pred_insts_cam)
                # pred_masks.append(pred_masks_cam)
                # feats.append(feats_cam)
                input_img.append(cam_img)
                depth.append(depth_cam.unsqueeze(1))
                K.append(K_cam.unsqueeze(1))
                E.append(E_cam.unsqueeze(1))
                voxel_origin.append(voxel_origin_cam.unsqueeze(1))
                projected_pix.append(projected_pix_cam.unsqueeze(1))
                fov_mask.append(fov_mask_cam.unsqueeze(1))
                ncam += 1

            input_img = torch.cat(input_img, dim=0)
            # encoder1_memory = torch.cuda.memory_allocated() / 1024**2
            # print(encoder1_memory - model_memory)
            pred_insts = self.encoder(input_img)
            # encoder2_memory = torch.cuda.memory_allocated() / 1024**2
            # print(encoder2_memory - encoder1_memory)
            pred_masks = pred_insts.pop('pred_masks', None)
            feats = pred_insts.pop('feats')

            # Pass variables to decoder (bs*Ncam, _ _) forms
            pred_new = {}
            for i in pred_insts.keys():
                pred = pred_insts[i].unsqueeze(1)
                pred = pred.reshape(ncam, -1, *pred.shape[2:])
                pred_insts[i] = pred.permute(1, 0, *range(2, pred.dim())).flatten(start_dim=0, end_dim=1)
            for i in range(len(feats)):
                pred = feats[i].unsqueeze(1)
                pred = pred.reshape(ncam, -1, *pred.shape[2:])
                feats[i] = pred.permute(1, 0, *range(2, pred.dim())).flatten(start_dim=0, end_dim=1)
            if pred_masks is not None:
                pred = pred_masks.unsqueeze(1)
                pred = pred.reshape(ncam, -1, *pred.shape[2:])
                pred_masks = pred.permute(1, 0, *range(2, pred.dim())).flatten(start_dim=0, end_dim=1)
            else:
                pred_masks = None

            # for j in pred_insts.keys():
            #     merged = torch.cat([pred_insts[i][j].unsqueeze(1) for i in range(len(pred_insts))], dim=1)  # bs x ncam x 50 x 2 or 128
            #     pred_new[j] = merged.flatten(start_dim=0, end_dim=1) # bs, Ncam*50, 2), (bs*Ncam, 50, 128)
            # pred_insts = pred_new
            
            # bs*Ncam, _, _ shape
            # feats_new=[]
            # for j in range(len(feats[0])):
            #     feats_merged = torch.cat([feats[i][j].unsqueeze(1) for i in range(len(feats))], dim=1)
            #     feats_new.append(feats_merged.flatten(start_dim=0, end_dim=1))
            # feats = feats_new

            ####pred masks 가 none으로 나오는 거 handling필요

            # for i, pred in enumerate(pred_masks):
            #     if pred_masks:
            #         pred
            #         print("Not None mask exists")
            #     else:
            #         pred_masks = None
            
            depth = torch.cat(depth, dim=1).flatten(start_dim=0, end_dim=1)
            K = torch.cat(K, dim=1).flatten(start_dim=0, end_dim=1)
            E = torch.cat(E, dim=1).flatten(start_dim=0, end_dim=1)
            voxel_origin = torch.cat(voxel_origin, dim=1).flatten(start_dim=0, end_dim=1)
            projected_pix = torch.cat(projected_pix, dim=1).flatten(start_dim=0, end_dim=1)
            fov_mask = torch.cat(fov_mask, dim=1).flatten(start_dim=0, end_dim=1)
            

        # For SemanticKITTI, KITTI360
        else:
            pred_insts = self.encoder(inputs['img'])
            pred_masks = pred_insts.pop('pred_masks', None)
            feats = pred_insts.pop('feats')

            depth, K, E, voxel_origin, projected_pix, fov_mask = list(
                map(lambda k: inputs[k],
                    ('depth', 'cam_K', 'cam_pose', 'voxel_origin', f'projected_pix_{self.volume_scale}',
                    f'fov_mask_{self.volume_scale}')))
        outs = self.decoder(pred_insts, feats, pred_masks, depth, K, E, voxel_origin, projected_pix,
                            fov_mask, ncam)
        return {'ssc_logits': outs[-1], 'aux_outputs': outs}

    def loss(self, preds, target):
        loss_map = {
            'ce_ssc': ce_ssc_loss,
            'sem_scal': sem_scal_loss,
            'geo_scal': geo_scal_loss,
            'frustum': frustum_proportion_loss
        }

        target['class_weights'] = self.class_weights.type_as(preds['ssc_logits'])
        losses = {}
        if 'aux_outputs' in preds:
            for i, pred in enumerate(preds['aux_outputs']):
                scale = 1 if i == len(preds['aux_outputs']) - 1 else 0.5
                for loss in self.criterions:
                    losses['loss_' + loss + '_' + str(i)] = loss_map[loss]({
                        'ssc_logits': pred
                    }, target) * scale
        else:
            for loss in self.criterions:
                losses['loss_' + loss] = loss_map[loss](preds, target)
        return losses
