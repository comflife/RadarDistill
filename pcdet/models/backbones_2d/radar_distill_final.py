import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pcdet.utils.box_utils import center_to_corner_box2d
from ...ops.basicblock.modules.Basicblock_convn import ConvNeXtBlock
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from functools import partial
from .base_bev_backbone import BaseBEVBackboneV2






    

def clip_sigmoid(x, eps=1e-4):
    
    # FIXME change back!
    # y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    y = torch.clamp(x.sigmoid(), min=eps, max=1 - eps)
    return y



            

class Radar_Distill(BaseBEVBackboneV2):
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg, **kwargs)
        self.model_cfg = model_cfg
        
        self.encoder_1 = nn.Sequential(
            ConvNeXtBlock(dim=256,downsample=True),
            ConvNeXtBlock(dim=256,downsample=False),
        )
        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(256,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.agg_1 = nn.Sequential(
            nn.Conv2d(512,256,1,1,0),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        self.encoder_2 = nn.Sequential(
            ConvNeXtBlock(dim=256,downsample=True),
            ConvNeXtBlock(dim=256,downsample=False),
        )
        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(256,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.agg_2 = nn.Sequential(
            nn.Conv2d(512,256,1,1,0),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        
        self.encoder_3 = nn.Sequential(
            ConvNeXtBlock(dim=256,downsample=True),
            ConvNeXtBlock(dim=256,downsample=False),
        )
        self.decoder_3 = nn.Sequential(
            nn.ConvTranspose2d(256,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.agg_3 = nn.Sequential(
            nn.Conv2d(512,256,1,1,0),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.voxel_size = self.model_cfg.VOXEL_SIZE
        self.point_cloud_range = self.model_cfg.POINT_CLOUD_RANGE
    
    def _get_bev_world_grid(self, H, W, device, dtype):
        x_min, y_min = self.point_cloud_range[0], self.point_cloud_range[1]
        x_max, y_max = self.point_cloud_range[3], self.point_cloud_range[4]

        # 각 픽셀은 동일 간격으로 다운샘플된 BEV 셀을 의미하므로 전체 범위/해상도로 스케일을 계산
        step_x = (x_max - x_min) / max(W, 1)
        step_y = (y_max - y_min) / max(H, 1)

        x_coords = torch.linspace(
            x_min + 0.5 * step_x, x_max - 0.5 * step_x, steps=W, device=device, dtype=dtype
        )
        y_coords = torch.linspace(
            y_min + 0.5 * step_y, y_max - 0.5 * step_y, steps=H, device=device, dtype=dtype
        )
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        return torch.stack([xx, yy], dim=-1)  # (H, W, 2)

    @torch.no_grad()
    def _build_gt_radar_distribution_object_aligned(self, batch_dict, device, dtype=torch.float32):
        """
        Object-aligned Radar Distribution (RICCARDO 논문 방식, 현재 코드베이스에 맞춤)
        
        - per-object local grid (heading-aligned)
        - 객체 크기의 2~3배 커버리지
        - BEV voxel_size (0.075m)를 기준으로 동적 grid 크기 계산
        - per-object 정규화 (sum → probability distribution)
        
        Returns:
            density_maps: List[List[Tensor]] 
                          batch_size개의 리스트, 각 배치 안에는 num_boxes개의 (grid_size, grid_size) tensor
        """
        batch_size = batch_dict['batch_size']
        density_maps = []

        radar_points = batch_dict.get('radar_points', None)
        if radar_points is None or radar_points.shape[0] == 0:
            for _ in range(batch_size):
                density_maps.append([])
            return density_maps

        if not torch.is_tensor(radar_points):
            radar_points = torch.from_numpy(radar_points)

        radar_points = radar_points.to(device=device, dtype=dtype)
        batch_indices = radar_points[:, 0].long()

        gt_boxes = batch_dict['gt_boxes']
        if not torch.is_tensor(gt_boxes):
            gt_boxes = torch.from_numpy(gt_boxes).to(device=device, dtype=dtype)
        else:
            gt_boxes = gt_boxes.to(device=device, dtype=dtype)

        # 현재 BEV voxel_size 사용 (0.075m for RadarDistill)
        pixel_size = float(self.voxel_size[0])  # x 방향 voxel size

        for b in range(batch_size):
            batch_density = []

            valid_mask = gt_boxes[b].sum(dim=1) != 0
            boxes = gt_boxes[b][valid_mask]

            if boxes.shape[0] == 0:
                density_maps.append([])
                continue

            point_mask = batch_indices == b
            batch_pts = radar_points[point_mask][:, 1:4]  # (P, 3)

            if batch_pts.shape[0] == 0:
                for _ in range(boxes.shape[0]):
                    # 빈 grid 추가 (크기는 나중에 계산)
                    batch_density.append(None)
                density_maps.append(batch_density)
                continue

            # roiaware_pool3d로 각 box별 inside points 빠르게 추출
            pts_exp = batch_pts.unsqueeze(0)  # (1, P, 3)
            boxes_exp = boxes[:, :7].unsqueeze(0).contiguous()

            box_idx = roiaware_pool3d_utils.points_in_boxes_gpu(pts_exp, boxes_exp).squeeze(0)  # (P,)

            for i in range(boxes.shape[0]):
                box = boxes[i]
                center = box[0:2]
                size = box[3:5]   # l, w
                angle = box[6]

                # === 동적 grid 크기 계산 ===
                # 객체를 충분히 커버하면서 주변 context도 포함 (2.5~3배)
                max_dim = max(size).item()
                coverage = max_dim * 2.5  # 객체 + 주변 context
                grid_size = int(coverage / pixel_size)
                
                # 최소/최대 크기 제한 (메모리 효율)
                grid_size = max(64, min(grid_size, 256))
                # 홀수로 만들어서 중심이 정확히 가운데 오도록
                if grid_size % 2 == 0:
                    grid_size += 1

                grid = torch.zeros(grid_size, grid_size, device=device, dtype=dtype)

                # 이 box에 속한 point만
                in_this_box = (box_idx == i)
                if not in_this_box.any():
                    batch_density.append(grid)  # 0 map
                    continue

                pts = batch_pts[in_this_box]  # (K, 3)

                # === Object-aligned 변환 ===
                rel_pts = pts[:, :2] - center.unsqueeze(0)  # (K, 2)

                cos_a, sin_a = torch.cos(angle), torch.sin(angle)
                rot_mat = torch.stack([torch.stack([cos_a, -sin_a]),
                                       torch.stack([sin_a, cos_a])])  # (2,2)

                local_pts = torch.matmul(rel_pts, rot_mat)  # (K, 2)

                # local grid index 계산 (center가 grid 중심)
                half_grid_world = (grid_size * pixel_size) / 2.0
                local_x = (local_pts[:, 0] + half_grid_world) / pixel_size
                local_y = (local_pts[:, 1] + half_grid_world) / pixel_size

                x_idx = local_x.long().clamp(0, grid_size - 1)
                y_idx = local_y.long().clamp(0, grid_size - 1)

                flat_idx = y_idx * grid_size + x_idx

                grid.view(-1).scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=dtype))

                # === Per-object 정규화 (probability distribution) ===
                total_hits = grid.sum()
                if total_hits > 0:
                    grid = grid / total_hits

                batch_density.append(grid)

            density_maps.append(batch_density)

        return density_maps
    

    
    
    def low_loss(self, lidar_bev, radar_bev, batch_dict):

        B, _, H, W = radar_bev.shape
        device = lidar_bev.device
        dtype = lidar_bev.dtype

        lidar_mask = (lidar_bev.sum(1).unsqueeze(1) > 0).float()
        radar_mask = radar_bev.sum(1, keepdim=True)

        # === Object-aligned Radar Distribution 생성 ===
        density_maps = self._build_gt_radar_distribution_object_aligned(
            batch_dict, device, dtype
        )

        # === Object-aligned grids → Global BEV grid로 projection ===
        radar_distribution = torch.zeros((B, 1, H, W), device=device, dtype=dtype)
        gt_boxes = batch_dict['gt_boxes']
        
        pixel_size = float(self.voxel_size[0])
        pc_range = self.point_cloud_range
        x_min, y_min = pc_range[0], pc_range[1]
        x_max, y_max = pc_range[3], pc_range[4]
        x_range = x_max - x_min
        y_range = y_max - y_min

        for b in range(B):
            if b >= len(density_maps) or len(density_maps[b]) == 0:
                continue
            
            valid_mask = gt_boxes[b].sum(dim=1) != 0
            boxes = gt_boxes[b][valid_mask]
            
            for i, obj_grid in enumerate(density_maps[b]):
                if obj_grid is None or i >= boxes.shape[0]:
                    continue
                
                box = boxes[i]
                center = box[0:2]
                angle = box[6]
                
                grid_size = obj_grid.shape[0]
                
                # Local grid 좌표 생성 (object center 기준)
                half_grid_world = (grid_size * pixel_size) / 2.0
                local_x = torch.linspace(-half_grid_world, half_grid_world, grid_size, device=device, dtype=dtype)
                local_y = torch.linspace(-half_grid_world, half_grid_world, grid_size, device=device, dtype=dtype)
                yy, xx = torch.meshgrid(local_y, local_x, indexing='ij')
                local_coords = torch.stack([xx, yy], dim=-1)  # (grid_size, grid_size, 2)
                
                # Rotate to world frame
                cos_a, sin_a = torch.cos(angle), torch.sin(angle)
                rot_mat = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=device, dtype=dtype)
                
                local_flat = local_coords.reshape(-1, 2)  # (grid_size^2, 2)
                world_flat = torch.matmul(local_flat, rot_mat.T) + center.unsqueeze(0)  # (grid_size^2, 2)
                
                # World → Global BEV grid index
                bev_x_idx = ((world_flat[:, 0] - x_min) / x_range * W).long().clamp(0, W - 1)
                bev_y_idx = ((world_flat[:, 1] - y_min) / y_range * H).long().clamp(0, H - 1)
                
                # Object-aligned density 값을 global BEV에 누적
                density_values = obj_grid.reshape(-1)  # (grid_size^2,)
                
                # scatter_add로 누적 (여러 객체가 겹치는 경우 합산)
                flat_indices = bev_y_idx * W + bev_x_idx
                radar_distribution[b, 0].view(-1).scatter_add_(0, flat_indices, density_values)
        
        # 최종 정규화 (per-batch)
        for b in range(B):
            max_val = radar_distribution[b].max()
            if max_val > 0:
                radar_distribution[b] = radar_distribution[b] / max_val

        # === 기존 Loss 계산 로직 ===
        gt_box_mask = (radar_distribution > 0).float()
        guided_lidar_mask = lidar_mask * (1 - gt_box_mask) + lidar_mask * radar_distribution

        activate_map = (radar_mask > 0).float() + guided_lidar_mask * 0.5

        mask_radar_lidar = torch.zeros_like(activate_map, dtype=torch.float)
        mask_radar_de_lidar = torch.zeros_like(activate_map, dtype=torch.float)
        mask_radar_lidar[activate_map == 1.5] = 1
        mask_radar_de_lidar[activate_map == 1.0] = 1

        if mask_radar_de_lidar.sum() > 0:
            mask_radar_de_lidar *= (mask_radar_lidar.sum() / (mask_radar_de_lidar.sum() + 1e-6))

        loss_radar_lidar = F.mse_loss(radar_bev, lidar_bev, reduction='none')
        loss_radar_lidar = torch.sum(loss_radar_lidar * mask_radar_lidar) / max(B, 1)

        loss_radar_de_lidar = F.mse_loss(radar_bev, lidar_bev, reduction='none')
        loss_radar_de_lidar = torch.sum(loss_radar_de_lidar * mask_radar_de_lidar) / max(B, 1)

        feature_loss = 3e-4 * loss_radar_lidar + 5e-5 * loss_radar_de_lidar
        mask_loss = nn.L1Loss()(radar_mask.sigmoid(), guided_lidar_mask)

        return feature_loss, mask_loss
    
    def high_loss(self, radar_bev, radar_bev2, lidar_bev, lidar_bev2, heatmaps, radar_preds, batch_dict):
        B, _, H, W = radar_bev.shape
        device = radar_bev.device
        dtype = radar_bev.dtype
        
        # === 기존 heatmap 기반 mask 계산 ===
        thres = 0.1
        gt_thres = 0.1
        gt_batch_hm = torch.cat(heatmaps, dim=1)
        gt_batch_hm_max = torch.max(gt_batch_hm, dim=1, keepdim=True)[0]
        
        radar_batch_hm = [(clip_sigmoid(radar_pred_dict['hm'])) for radar_pred_dict in radar_preds]
        radar_batch_hm = torch.cat(radar_batch_hm, dim=1)
        radar_batch_hm_max = torch.max(radar_batch_hm, dim=1, keepdim=True)[0]
        
        radar_fp_mask = torch.logical_and(gt_batch_hm_max < gt_thres, radar_batch_hm_max > thres)
        radar_fn_mask = torch.logical_and(gt_batch_hm_max > gt_thres, radar_batch_hm_max < thres)
        radar_tp_mask = torch.logical_and(gt_batch_hm_max > gt_thres, radar_batch_hm_max > thres)
        
        # 기본 weight 계산
        wegiht = torch.zeros_like(radar_batch_hm_max)
        tp_fn_sum = (radar_tp_mask + radar_fn_mask).sum()
        fp_sum = radar_fp_mask.sum()
        
        if tp_fn_sum > 0:
            wegiht[radar_tp_mask + radar_fn_mask] = 5.0 / tp_fn_sum
        if fp_sum > 0:
            wegiht[radar_fp_mask] = 1.0 / fp_sum
        
        # === Loss 계산 ===
        scaled_radar_bev = radar_bev.softmax(1)
        scaled_lidar_bev = lidar_bev.softmax(1)
        
        scaled_radar_bev2 = radar_bev2.softmax(1)
        scaled_lidar_bev2 = lidar_bev2.softmax(1)
        
        high_loss = F.l1_loss(scaled_radar_bev, scaled_lidar_bev, reduction='none') * wegiht
        high_loss = high_loss.sum()
        high_8x_loss = F.l1_loss(scaled_radar_bev2, scaled_lidar_bev2, reduction='none') * wegiht
        high_8x_loss = high_8x_loss.sum()
        high_loss = 0.5 * (high_loss + high_8x_loss)
        return high_loss

    def get_loss(self, batch_dict):
        low_lidar_bev =  batch_dict['multi_scale_2d_features']['x_conv4']
        low_radar_bev = batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_2']
        low_radar_de_8x = batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_1']
        high_radar_bev = batch_dict['radar_spatial_features_2d']
        high_lidar_bev = batch_dict['spatial_features_2d']
        high_radar_bev_8x = batch_dict['radar_spatial_features_2d_8x']
        high_lidar_bev_8x = batch_dict['spatial_features_2d_8x']
        radar_pred_dicts = batch_dict['radar_pred_dicts']
        gt_heatmaps = batch_dict['target_dicts']['heatmaps']
        gt_boxes = batch_dict['gt_boxes']
        
        B, _, H, W = low_radar_bev.shape
        
        feature_loss, mask_loss = self.low_loss(low_lidar_bev, low_radar_bev, batch_dict)
        de_8x_feature_loss, de_8x_mask_loss = self.low_loss(low_lidar_bev, low_radar_de_8x, batch_dict)

        
        high_distill_loss = self.high_loss(high_radar_bev,high_radar_bev_8x, high_lidar_bev,high_lidar_bev_8x, gt_heatmaps, radar_pred_dicts, batch_dict)
        high_distill_loss *= 25
        low_distill_loss = 0.5 * (feature_loss + de_8x_feature_loss) + 0.5 * (mask_loss + de_8x_mask_loss)
        low_distill_loss *= 5
        distill_loss = low_distill_loss + high_distill_loss

        tb_dict={
            'low_feature_loss' : low_distill_loss.item(),
            'high_distill_loss' : high_distill_loss.item(),
            'distll_loss' : distill_loss.item(),
            'low_distill_de_8x_loss' : de_8x_feature_loss.item(),
            'low_distill_loss' : feature_loss.item(),
            'mask_loss' : mask_loss.item(),
            'mask_de_8x_loss': de_8x_mask_loss.item(),
        }
        return distill_loss, tb_dict
    
    def forward(self, data_dict):
        
        spatial_features = data_dict['radar_multi_scale_2d_features']['x_conv4']
        ups = []
        ret_dict = {}
        
        en_16x = self.encoder_1(spatial_features) #(B, 256, 90, 90)
        de_8x = torch.cat((self.decoder_1(en_16x), spatial_features), dim=1)#(B,512,180,180)
        de_8x = self.agg_1(de_8x)#(B,256,180,180)
        
        en_32x = self.encoder_2(en_16x)#(B,256,45,45)
        de_16x = torch.cat((self.decoder_2(en_32x), self.encoder_3(de_8x)), dim=1)#(B,512,90,90)
        de_16x = self.agg_2(de_16x)#(B,256,90,90)

        x = torch.cat((self.decoder_3(de_16x), de_8x), dim=1)#(B, 512, 180, 180)
        x_conv4 = self.agg_3(x)

        data_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_2'] = x_conv4
        data_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_1'] = de_8x

        
        x_conv5 = data_dict['radar_multi_scale_2d_features']['x_conv5']
        
        ups = [x_conv4]
        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[0](x))
        data_dict['radar_spatial_features_2d_8x'] = ups[-1]


        x = torch.cat(ups, dim=1)
        x = self.blocks[0](x)
        
        data_dict['radar_spatial_features_2d'] = x
        
                
        return data_dict
    
