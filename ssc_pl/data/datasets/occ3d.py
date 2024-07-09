# Init Datasloader
from nuscenes import NuScenes as nusc
from nuscenes.utils import splits
import glob
import os.path as osp
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms as T
from ssc_pl.utils.helper import compute_CP_mega_matrix, compute_local_frustums, vox2pix
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from PIL import Image

OCC3D_CLASS_FREQ = torch.tensor([])
normalize_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], 
                     to_rgb=True, debug=False)

version = "v1.0-trainval"

cam_list = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

NUSPLITS = {
    'train': splits.train,
    'val': splits.val,
    'test': splits.test,
    }

NUSCENES = nusc(version=version, dataroot='./data/nuscenes', verbose=True)

class Occ3D(Dataset):
    META_INFO = {
        'class_weights':
        1 / torch.log(OCC3D_CLASS_FREQ + 1e-6),
        'class names':
        ('others', 'barrier', 'bicycle' 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
        'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation', 'free')
    }
    
    def __init__(
        self,
        split,
        data_root,
        project_scale=2,
        frustum_size=4,
        # context_prior=False,
        flip=False,
        load_pose=False,
        # load_only_with_target=True,
    ):
        super().__init__()
        self.data_root = data_root
        self.scene = NUSPLITS[split]
        self.split = split

        self.frustum_size = frustum_size
        self.project_scale = project_scale
        self.output_scale = int(self.project_scale / 2)
        # self.context_prior = context_prior
        self.flip = flip
        self.load_pose = load_pose
        self.num_classes = 18

        self.voxel_origin = np.array((-2, -40, -40))
        self.voxel_size = 0.4
        self.scene_size = (6.4, 80, 80)
        self.img_shape = (1600, 900)

        self.scans = []
        self.nusc = NUSCENES
        
        print(f"Preparing Data for {split}...")
        
        for scene_number in tqdm(self.scene):
            for scene_data in self.nusc.scene:
                if scene_data['name'] == scene_number:
                    scene = scene_data
            sample_token = scene['first_sample_token']

            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                voxel_path = osp.join(self.data_root, 'gts', scene_number, sample_token, 'labels.npz')
                sample_data = {
                    'scene': scene_number,
                    'sample': sample_token,
                    'voxel_path': voxel_path,
                    'cam_channels': {},
                }

                # # Load lidar data
                # lid_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                # lid_calib = self.nusc.get('calibrated_sensor', lid_data['calibrated_sensor_token'])
                # lid_quat = lid_calib['rotation']
                # lid_rot = R.from_quat(lid_quat[1:]+[lid_quat[0]]).as_matrix()
                # lid_trans = lid_calib['translation']
                # T_lid2veh = np.eye(4)
                # T_lid2veh[:3, :3] = lid_rot
                # T_lid2veh[:3, 3] = lid_trans

                # Load Camera data for each view
                for cam in cam_list:
                    cam_data = self.nusc.get('sample_data', sample['data'][cam])

                    file_path = osp.join(self.data_root, cam_data['filename'])
                    cam_calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                    cam_intrinsic = np.array(cam_calib['camera_intrinsic'])  ## P
                    cam_quat = cam_calib['rotation']
                    rot = R.from_quat(cam_quat[1:]+[cam_quat[0]]).as_matrix()
                    trans = cam_calib['translation']
                    T_cam2veh = np.eye(4)
                    T_cam2veh[:3, :3] = rot
                    T_cam2veh[:3, 3] = trans
                    T_veh2cam = np.linalg.inv(T_cam2veh)
    
                    # T_velo_2_cam = T_veh2cam #@ T_lid2veh # (4x4)
                    # T_velo_2_cam = T_velo_2_cam[:3, :] # (3x4)

                    proj_matrix = cam_intrinsic @ T_veh2cam[:3, :]
                    sample_data['cam_channels'][cam] = {
                        'P': cam_intrinsic,
                        'T_veh_2_cam': T_veh2cam,
                        'proj_matrix': proj_matrix,
                        'image_path': file_path,
                    }

                self.scans.append(sample_data)

                sample_token = sample['next']

        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.scans)
    
    def __getitem__(self, idx):
        flip = random.random() > 0.5 if self.flip and self.split == 'train' else False
        scan = self.scans[idx]
        scene = scan['scene']
        frame = scan['sample']
        scale_3ds = (self.output_scale, self.project_scale)
        
        data = {'frame_id': frame,
                'scene': scene,
                'scale_3ds': scale_3ds,
                'cam_channels': {},
                }
        label = {'frame_id': frame,
                'scene': scene,
                'cam_channels': {},
                }

        target_path = scan['voxel_path']
        with_target = self.split != 'test' and osp.exists(target_path)
        if with_target:
            target = np.load(target_path)
            mask_camera = target['mask_camera']
            gt = target['semantics']
            if flip:
                gt = np.flip(gt, axis=1).copy()
            gt[gt == 17] = 0
            gt = gt * mask_camera
            label['mask_camera'] = mask_camera
            label['target'] = gt

        for camera in cam_list:
            label_cam = {}
            cam_scan = scan['cam_channels'][camera]
            P = cam_scan['P']
            T_veh_2_cam = cam_scan['T_veh_2_cam']
            proj_matrix = cam_scan['proj_matrix']
            data_cam = {
                'P': P,
                'cam_pose': T_veh_2_cam,
                'proj_matrix': proj_matrix,
                'voxel_origin': self.voxel_origin,
                }
            cam_K = P[:3, :3]

            data_cam['cam_K'] = cam_K
            for scale_3d in scale_3ds:
                # compute the 3D-2D mapping
                projected_pix, fov_mask, pix_z = vox2pix(T_veh_2_cam, cam_K, self.voxel_origin,
                                                        self.voxel_size * scale_3d, self.img_shape,
                                                        self.scene_size)
                data_cam[f'projected_pix_{scale_3d}'] = projected_pix
                data_cam[f'pix_z_{scale_3d}'] = pix_z
                data_cam[f'fov_mask_{scale_3d}'] = fov_mask

            if self.data_root is not None:
                depth_path = osp.join(self.data_root, 'depth', scene, frame, camera+'.npz')
                depth = np.load(depth_path)
                depth = depth['arr_0'][:self.img_shape[1], :self.img_shape[0]].astype(np.float32)
                if flip:
                    depth = np.flip(depth, axis=1).copy()

                ## Divide 4 for Depth Scaling
                data_cam['depth'] = depth / 4
            
            img_path = cam_scan['image_path']
            img = Image.open(img_path).convert('RGB')
            img = np.array(img, dtype=np.float32) / 255.0
            if flip:
                img = np.flip(img, axis=1).copy()
            img = img[:self.img_shape[1], :self.img_shape[0]]
            data_cam['img'] = self.transforms(img)
            
            ## This is for frustrum loss, which will not be used for Occ3D dataset
            # if with_target:
            #     frustums_masks, frustums_class_dists = compute_local_frustums(
            #         data_cam[f'projected_pix_{self.output_scale}'],
            #         data_cam[f'pix_z_{self.output_scale}'],
            #         gt,
            #         self.img_shape,
            #         n_classes=self.num_classes,
            #         size=self.frustum_size,
            #     )
            #     label_cam['frustums_masks'] = frustums_masks
            #     label_cam['frustums_class_dists'] = frustums_class_dists

            data_cam = ndarray_to_tensor(data_cam)
            label_cam = ndarray_to_tensor(label_cam)
            data['cam_channels'][camera] = data_cam
            label['cam_channels'][camera] = label_cam

        return data, label
    
def ndarray_to_tensor(data: dict):
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            if v.dtype == np.float64:
                v = v.astype('float32')
            data[k] = torch.from_numpy(v)
    return data

