import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pcdet.utils.box_utils import center_to_corner_box2d
from ...ops.basicblock.modules.Basicblock_convn import ConvNeXtBlock
from functools import partial
import cv2
from .base_bev_backbone import BaseBEVBackboneV2


def extract_keypoint_features_from_bev(
    bev_features, gt_boxes, point_cloud_range, voxel_size, num_keypoints=256, enlarge_factor=1.0
):
    """
    주어진 GT 박스 영역 내에서 BEV 피처를 추출합니다. (TiGDistill-BEV 논문 참고)
    
    Args:
        bev_features: (B, C, H, W) BEV feature map
        gt_boxes: (B, N, 8) GT boxes [x, y, z, dx, dy, dz, heading, class_id]
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        voxel_size: [vx, vy, vz]
        num_keypoints: number of keypoints to sample per box
        enlarge_factor: factor to enlarge the box region
        
    Returns:
        keypoint_features_list: list of (N_boxes, num_keypoints, C) per batch
    """
    batch_size, num_channels, H, W = bev_features.shape
    keypoint_features_list = []

    pc_range = torch.tensor(point_cloud_range, device=bev_features.device)
    voxel_size_tensor = torch.tensor(voxel_size, device=bev_features.device)

    for b in range(batch_size):
        batch_bev_features = bev_features[b]  # (C, H, W)
        batch_gt_boxes = gt_boxes[b]  # (N, 8)

        # 유효한 GT 박스만 선택 (padding 제거)
        mask = batch_gt_boxes.sum(dim=1) != 0
        valid_gt_boxes = batch_gt_boxes[mask]

        if valid_gt_boxes.shape[0] == 0:
            continue

        box_keypoints_list = []
        for i in range(valid_gt_boxes.shape[0]):
            box = valid_gt_boxes[i]
            box_center = box[0:2]  # (x, y)
            box_dims = box[3:5] * enlarge_factor  # (dx, dy) * enlarge_factor
            box_angle = box[6]  # heading

            # 박스 내부에 균등하게 keypoints 샘플링 (local coordinates)
            keypoints_local = torch.rand(num_keypoints, 2, device=bev_features.device) - 0.5
            keypoints_local *= box_dims.unsqueeze(0)  # scale to box size

            # Rotate keypoints according to box heading
            rot_sin, rot_cos = torch.sin(box_angle), torch.cos(box_angle)
            rot_mat = torch.tensor([[rot_cos, -rot_sin], [rot_sin, rot_cos]], device=bev_features.device)
            keypoints_rotated = torch.matmul(keypoints_local, rot_mat.T)
            
            # Transform to world coordinates
            world_coords = box_center.unsqueeze(0) + keypoints_rotated

            # Convert world coordinates to BEV grid coordinates
            bev_coords_x = (world_coords[:, 0] - pc_range[0]) / voxel_size_tensor[0]
            bev_coords_y = (world_coords[:, 1] - pc_range[1]) / voxel_size_tensor[1]

            # Normalize to [-1, 1] for grid_sample
            normalized_x = (bev_coords_x / (W - 1)) * 2.0 - 1.0
            normalized_y = (bev_coords_y / (H - 1)) * 2.0 - 1.0
            
            grid = torch.stack([normalized_x, normalized_y], dim=1).unsqueeze(0).unsqueeze(0)  # (1, 1, num_keypoints, 2)

            # Sample features at keypoint locations
            sampled_features = F.grid_sample(
                batch_bev_features.unsqueeze(0),  # (1, C, H, W)
                grid,
                mode='bilinear',
                align_corners=True
            ).squeeze(0).squeeze(1).permute(1, 0)  # (num_keypoints, C)
            
            box_keypoints_list.append(sampled_features)
        
        if len(box_keypoints_list) > 0:
            keypoint_features_list.append(torch.stack(box_keypoints_list))  # (N_boxes, num_keypoints, C)

    return keypoint_features_list


def extract_dual_region_keypoints(
    bev_features, gt_boxes, point_cloud_range, voxel_size,
    num_keypoints_inner=256, num_keypoints_outer=256,
    inner_factor=1.0, outer_factor=2.0,
    x_sample_num: int = None, y_sample_num: int = None,
    outer_x_sample_num: int = None, outer_y_sample_num: int = None,
):
    """
    GT 박스에서 두 개의 영역(inner, outer)에서 keypoint를 추출합니다.
    
    Args:
        bev_features: (B, C, H, W) BEV feature map
        gt_boxes: (B, N, 8) GT boxes [x, y, z, dx, dy, dz, heading, class_id]
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        voxel_size: [vx, vy, vz]
        num_keypoints_inner: inner region에서 샘플링할 keypoint 수
        num_keypoints_outer: outer region에서 샘플링할 keypoint 수
        inner_factor: inner box의 enlarge factor (기본 1.0 = GT box)
        outer_factor: outer box의 enlarge factor (기본 2.0)
        
    Returns:
        inner_keypoints_list: list of (N_boxes, Nin, C) per batch (GT box 내부)
        outer_keypoints_list: list of (N_boxes, Nout, C) per batch (1.0~2.0 사이)
    """
    batch_size, num_channels, H, W = bev_features.shape
    inner_keypoints_list = []
    outer_keypoints_list = []

    pc_range = torch.tensor(point_cloud_range, device=bev_features.device)
    voxel_size_tensor = torch.tensor(voxel_size, device=bev_features.device)

    for b in range(batch_size):
        batch_bev_features = bev_features[b]  # (C, H, W)
        batch_gt_boxes = gt_boxes[b]  # (N, 8)

        # 유효한 GT 박스만 선택
        mask = batch_gt_boxes.sum(dim=1) != 0
        valid_gt_boxes = batch_gt_boxes[mask]

        if valid_gt_boxes.shape[0] == 0:
            continue

        inner_box_keypoints_list = []
        outer_box_keypoints_list = []
        
        for i in range(valid_gt_boxes.shape[0]):
            box = valid_gt_boxes[i]
            box_center = box[0:2]  # (x, y)
            box_dims_inner = box[3:5] * inner_factor  # (dx, dy) * 1.0
            box_dims_outer = box[3:5] * outer_factor  # (dx, dy) * 2.0
            box_angle = box[6]  # heading

            # Rotation matrix
            rot_sin, rot_cos = torch.sin(box_angle), torch.cos(box_angle)
            rot_mat = torch.tensor([[rot_cos, -rot_sin], [rot_sin, rot_cos]], device=bev_features.device)

            # ============= Inner Region (GT box 내부) =============
            if x_sample_num is not None and y_sample_num is not None:
                # Deterministic grid sampling (e.g., 24x24)
                xs = torch.linspace(-0.5, 0.5, steps=x_sample_num, device=bev_features.device)
                ys = torch.linspace(-0.5, 0.5, steps=y_sample_num, device=bev_features.device)
                yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # (y_sample_num, x_sample_num)
                grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (N, 2) in [-0.5, 0.5]
                keypoints_local_inner = grid * box_dims_inner.unsqueeze(0)
            else:
                # Random sampling fallback
                keypoints_local_inner = torch.rand(num_keypoints_inner, 2, device=bev_features.device) - 0.5
                keypoints_local_inner *= box_dims_inner.unsqueeze(0)
            keypoints_rotated_inner = torch.matmul(keypoints_local_inner, rot_mat.T)
            world_coords_inner = box_center.unsqueeze(0) + keypoints_rotated_inner

            bev_coords_x_inner = (world_coords_inner[:, 0] - pc_range[0]) / voxel_size_tensor[0]
            bev_coords_y_inner = (world_coords_inner[:, 1] - pc_range[1]) / voxel_size_tensor[1]
            normalized_x_inner = (bev_coords_x_inner / (W - 1)) * 2.0 - 1.0
            normalized_y_inner = (bev_coords_y_inner / (H - 1)) * 2.0 - 1.0
            grid_inner = torch.stack([normalized_x_inner, normalized_y_inner], dim=1).unsqueeze(0).unsqueeze(0)

            sampled_features_inner = F.grid_sample(
                batch_bev_features.unsqueeze(0),
                grid_inner,
                mode='bilinear',
                align_corners=True
            ).squeeze(0).squeeze(1).permute(1, 0)  # (num_keypoints_inner, C)
            
            # ============= Outer Region (1.0 ~ 2.0 사이) =============
            # outer box 영역에서 샘플링하되, inner box 영역은 제외
            if outer_x_sample_num is not None and outer_y_sample_num is not None:
                xs_o = torch.linspace(-0.5, 0.5, steps=outer_x_sample_num, device=bev_features.device)
                ys_o = torch.linspace(-0.5, 0.5, steps=outer_y_sample_num, device=bev_features.device)
                yy_o, xx_o = torch.meshgrid(ys_o, xs_o, indexing='ij')
                grid_o = torch.stack([xx_o.reshape(-1), yy_o.reshape(-1)], dim=-1)  # (No, 2)
                # Convert to local coords by scaling with outer dims
                kp_local_outer_all = grid_o * box_dims_outer.unsqueeze(0)
                # Mask out points that fall inside the inner box
                inside_x = torch.abs(kp_local_outer_all[:, 0]) <= (box_dims_inner[0] / 2)
                inside_y = torch.abs(kp_local_outer_all[:, 1]) <= (box_dims_inner[1] / 2)
                mask_outer = ~(inside_x & inside_y)
                keypoints_local_outer = kp_local_outer_all[mask_outer]
                # Optionally downsample to num_keypoints_outer if provided and smaller
                if num_keypoints_outer is not None and keypoints_local_outer.shape[0] > num_keypoints_outer:
                    idx = torch.randperm(keypoints_local_outer.shape[0], device=bev_features.device)[:num_keypoints_outer]
                    keypoints_local_outer = keypoints_local_outer[idx]
            else:
                # Random sampling fallback
                max_attempts = num_keypoints_outer * 3 if num_keypoints_outer is not None else 2048
                outer_keypoints_local = []
                attempts = 0
                while (num_keypoints_outer is None or len(outer_keypoints_local) < num_keypoints_outer) and attempts < max_attempts:
                    candidate = torch.rand(1, 2, device=bev_features.device) - 0.5
                    candidate = candidate * box_dims_outer.unsqueeze(0)
                    inside_x = torch.abs(candidate[:, 0]) <= (box_dims_inner[0] / 2)
                    inside_y = torch.abs(candidate[:, 1]) <= (box_dims_inner[1] / 2)
                    if not (inside_x & inside_y).item():
                        outer_keypoints_local.append(candidate)
                    attempts += 1
                if len(outer_keypoints_local) == 0:
                    # Fallback to a single outer corner if extremely degenerate
                    keypoints_local_outer = (torch.tensor([[0.5, 0.5]], device=bev_features.device) * box_dims_outer.unsqueeze(0))
                else:
                    keypoints_local_outer = torch.cat(outer_keypoints_local, dim=0)
            keypoints_rotated_outer = torch.matmul(keypoints_local_outer, rot_mat.T)
            world_coords_outer = box_center.unsqueeze(0) + keypoints_rotated_outer

            bev_coords_x_outer = (world_coords_outer[:, 0] - pc_range[0]) / voxel_size_tensor[0]
            bev_coords_y_outer = (world_coords_outer[:, 1] - pc_range[1]) / voxel_size_tensor[1]
            normalized_x_outer = (bev_coords_x_outer / (W - 1)) * 2.0 - 1.0
            normalized_y_outer = (bev_coords_y_outer / (H - 1)) * 2.0 - 1.0
            grid_outer = torch.stack([normalized_x_outer, normalized_y_outer], dim=1).unsqueeze(0).unsqueeze(0)

            sampled_features_outer = F.grid_sample(
                batch_bev_features.unsqueeze(0),
                grid_outer,
                mode='bilinear',
                align_corners=True
            ).squeeze(0).squeeze(1).permute(1, 0)  # (num_keypoints_outer, C)
            
            inner_box_keypoints_list.append(sampled_features_inner)
            outer_box_keypoints_list.append(sampled_features_outer)
        
        if len(inner_box_keypoints_list) > 0:
            inner_keypoints_list.append(torch.stack(inner_box_keypoints_list))
            outer_keypoints_list.append(torch.stack(outer_box_keypoints_list))

    return inner_keypoints_list, outer_keypoints_list


def clip_sigmoid(x, eps=1e-4):
    """Sigmoid function for input feature.

    Args:
        x (torch.Tensor): Input feature map with the shape of [B, N, H, W].
        eps (float): Lower bound of the range to be clamped to. Defaults
            to 1e-4.

    Returns:
        torch.Tensor: Feature map after sigmoid.
    """
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
    
    
    def low_loss(self, lidar_bev, radar_bev):

        B, _, H, W = radar_bev.shape
        lidar_mask = (lidar_bev.sum(1).unsqueeze(1) > 0).float()
        
        radar_mask = (radar_bev.sum(1).unsqueeze(1))
        
        activate_map = (radar_mask > 0).float() + lidar_mask * 0.5

        mask_radar_lidar = torch.zeros_like(activate_map, dtype=torch.float)
        mask_radar_de_lidar = torch.zeros_like(activate_map, dtype=torch.float)
        mask_radar_lidar[activate_map==1.5] = 1
        mask_radar_de_lidar[activate_map==1.0] = 1

        mask_radar_de_lidar *= (mask_radar_lidar.sum() / mask_radar_de_lidar.sum())

        loss_radar_lidar = F.mse_loss(radar_bev, lidar_bev, reduction='none')
        loss_radar_lidar = torch.sum(loss_radar_lidar * mask_radar_lidar) / B
        
        loss_radar_de_lidar = F.mse_loss(radar_bev, lidar_bev, reduction='none')
        loss_radar_de_lidar = torch.sum(loss_radar_de_lidar * mask_radar_de_lidar) / B

        # breakpoint()
        feature_loss = 3e-4 * loss_radar_lidar + 5e-5 * loss_radar_de_lidar
        loss = nn.L1Loss()
        mask_loss = loss(radar_mask.sigmoid(), lidar_mask)

        return feature_loss, mask_loss
    
    def high_loss(self, radar_bev,radar_bev2, lidar_bev,lidar_bev2, heatmaps, radar_preds):
        thres = 0.1
        gt_thres = 0.1
        gt_batch_hm = torch.cat(heatmaps, dim=1)
        gt_batch_hm_max = torch.max(gt_batch_hm, dim=1, keepdim=True)[0]
        
        #[1, 2, 2, 1, 2, 2]
        radar_batch_hm = [(clip_sigmoid(radar_pred_dict['hm'])) for radar_pred_dict in radar_preds]
        radar_batch_hm = torch.cat(radar_batch_hm, dim=1)
        radar_batch_hm_max = torch.max(radar_batch_hm, dim=1, keepdim=True)[0]
        
        radar_fp_mask = torch.logical_and(gt_batch_hm_max < gt_thres, radar_batch_hm_max > thres)
        radar_fn_mask = torch.logical_and(gt_batch_hm_max > gt_thres, radar_batch_hm_max < thres)
        radar_tp_mask = torch.logical_and(gt_batch_hm_max > gt_thres, radar_batch_hm_max > thres)
        # radar_tn_mask = torch.logical_and(gt_batch_hm_max < gt_thres, radar_batch_hm_max < thres)
        wegiht = torch.zeros_like(radar_batch_hm_max)
        wegiht[radar_tp_mask + radar_fn_mask] = 5 /(radar_tp_mask + radar_fn_mask).sum()
        wegiht[radar_fp_mask] = 1 / (radar_fp_mask).sum()
        
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
        
        B, _, H, W = low_radar_bev.shape
        
        feature_loss, mask_loss = self.low_loss(low_lidar_bev, low_radar_bev)
        de_8x_feature_loss, de_8x_mask_loss = self.low_loss(low_lidar_bev, low_radar_de_8x)

        
        high_distill_loss = self.high_loss(high_radar_bev,high_radar_bev_8x, high_lidar_bev,high_lidar_bev_8x, gt_heatmaps, radar_pred_dicts)
        high_distill_loss *= 25
        low_distill_loss = 0.5 * (feature_loss + de_8x_feature_loss) + 0.5 * (mask_loss + de_8x_mask_loss)
        low_distill_loss *= 5
        distill_loss = low_distill_loss + high_distill_loss
        
        # ================== TiGDistill-BEV Inter-channel Distillation Loss with Dual Region ==================
        # Extract TiGDistill configuration
        distill_cfg = self.model_cfg.get('TIG_DISTILL', {})
        tig_distill_dict = {}
        if distill_cfg:
            num_keypoints_inner = distill_cfg.get('NUM_KEYPOINTS', 256)
            num_keypoints_outer = distill_cfg.get('NUM_KEYPOINTS_OUTER', 256)
            inner_factor = distill_cfg.get('INNER_FACTOR', 1.0)
            outer_factor = distill_cfg.get('OUTER_FACTOR', 2.0)
            x_sample_num = distill_cfg.get('X_SAMPLE_NUM', None)
            y_sample_num = distill_cfg.get('Y_SAMPLE_NUM', None)
            outer_x_sample_num = distill_cfg.get('OUTER_X_SAMPLE_NUM', None)
            outer_y_sample_num = distill_cfg.get('OUTER_Y_SAMPLE_NUM', None)
            bev_ic_weight = distill_cfg.get('BEV_IC_WEIGHT', 1.0)
            bev_ik_weight = distill_cfg.get('BEV_IK_WEIGHT', 1.0)
            bev_contrastive_weight = distill_cfg.get('BEV_CONTRASTIVE_WEIGHT', 0.5)
            contrastive_margin = distill_cfg.get('CONTRASTIVE_MARGIN', 1.0)
            
            # Get GT boxes and BEV features
            gt_boxes = batch_dict['gt_boxes']
            
            # Use high-resolution BEV features for inter-channel distillation
            teacher_bev_features = high_lidar_bev
            student_bev_features = high_radar_bev
            
            # Extract dual region keypoint features from both teacher and student
            teacher_inner_list, teacher_outer_list = extract_dual_region_keypoints(
                teacher_bev_features, gt_boxes, 
                self.point_cloud_range, self.voxel_size, 
                num_keypoints_inner, num_keypoints_outer,
                inner_factor, outer_factor,
                x_sample_num, y_sample_num,
                outer_x_sample_num, outer_y_sample_num
            )
            student_inner_list, student_outer_list = extract_dual_region_keypoints(
                student_bev_features, gt_boxes, 
                self.point_cloud_range, self.voxel_size, 
                num_keypoints_inner, num_keypoints_outer,
                inner_factor, outer_factor,
                x_sample_num, y_sample_num,
                outer_x_sample_num, outer_y_sample_num
            )

            # Compute inter-channel and inter-keypoint correlation loss
            loss_bev_ic = 0.0  # inter-channel loss (inner region only)
            loss_bev_ik = 0.0  # inter-keypoint loss (inner region only)
            loss_bev_contrastive = 0.0  # contrastive loss (push outer region away)
            num_objects = 0

            if len(teacher_inner_list) > 0 and len(student_inner_list) > 0:
                for idx in range(len(teacher_inner_list)):
                    teacher_inner = teacher_inner_list[idx]
                    student_inner = student_inner_list[idx]
                    teacher_outer = teacher_outer_list[idx]
                    student_outer = student_outer_list[idx]
                    
                    num_boxes = min(teacher_inner.shape[0], student_inner.shape[0])
                    for obj_idx in range(num_boxes):
                        # ============= Inner Region: Alignment Loss =============
                        f_teacher_inner = teacher_inner[obj_idx]  # (num_keypoints_inner, C)
                        f_student_inner = student_inner[obj_idx]  # (num_keypoints_inner, C)

                        # Compute inter-channel correlation matrices
                        A_teacher = f_teacher_inner.T @ f_teacher_inner  # (C, C)
                        A_student = f_student_inner.T @ f_student_inner  # (C, C)
                        loss_bev_ic += F.mse_loss(A_teacher, A_student, reduction='mean')
                        
                        # Compute inter-keypoint correlation matrices
                        B_teacher = f_teacher_inner @ f_teacher_inner.T  # (num_keypoints_inner, num_keypoints_inner)
                        B_student = f_student_inner @ f_student_inner.T  # (num_keypoints_inner, num_keypoints_inner)
                        loss_bev_ik += F.mse_loss(B_teacher, B_student, reduction='mean')
                        
                        # ============= Outer Region: Contrastive Loss =============
                        # Goal: Push outer region features away from inner region features
                        # Method 1: Maximize distance between inner and outer region
                        # Method 2: Minimize similarity between inner and outer region
                        # Method 3: Contrastive loss with margin
                        
                        f_student_outer = student_outer[obj_idx]  # (num_keypoints_outer, C)
                        
                        # L2 normalize features for better distance computation
                        f_inner_norm = F.normalize(f_student_inner, p=2, dim=1)  # (num_keypoints_inner, C)
                        f_outer_norm = F.normalize(f_student_outer, p=2, dim=1)  # (num_keypoints_outer, C)
                        
                        # Compute cosine similarity between inner and outer keypoints
                        # similarity: (num_keypoints_outer, num_keypoints_inner)
                        similarity = torch.mm(f_outer_norm, f_inner_norm.T)
                        
                        # Contrastive loss: push outer features away from inner features
                        # We want low similarity (ideally negative or zero)
                        # Use margin-based loss: max(0, similarity - margin)
                        contrastive_loss = torch.clamp(similarity + contrastive_margin, min=0.0).mean()
                        loss_bev_contrastive += contrastive_loss
                        
                        num_objects += 1

            if num_objects > 0:
                loss_bev_ic = loss_bev_ic / num_objects
                loss_bev_ik = loss_bev_ik / num_objects
                loss_bev_contrastive = loss_bev_contrastive / num_objects
                
                # Add all losses to total distill_loss
                distill_loss = distill_loss + \
                               loss_bev_ic * bev_ic_weight + \
                               loss_bev_ik * bev_ik_weight + \
                               loss_bev_contrastive * bev_contrastive_weight
                
                # Store in dict for logging
                tig_distill_dict['loss_bev_ic'] = loss_bev_ic.item()
                tig_distill_dict['loss_bev_ik'] = loss_bev_ik.item()
                tig_distill_dict['loss_bev_contrastive'] = loss_bev_contrastive.item()
                
                # Store in batch_dict for pillarnet to use (for backward compatibility)
                batch_dict['loss_bev_ic'] = loss_bev_ic
                batch_dict['loss_bev_ik'] = loss_bev_ik
                batch_dict['loss_bev_contrastive'] = loss_bev_contrastive
                batch_dict['loss_bev_combined'] = loss_bev_ic + loss_bev_ik + loss_bev_contrastive
            else:
                tig_distill_dict['loss_bev_ic'] = 0.0
                tig_distill_dict['loss_bev_ik'] = 0.0
                tig_distill_dict['loss_bev_contrastive'] = 0.0
                
                batch_dict['loss_bev_ic'] = torch.tensor(0.0, device=high_radar_bev.device)
                batch_dict['loss_bev_ik'] = torch.tensor(0.0, device=high_radar_bev.device)
                batch_dict['loss_bev_contrastive'] = torch.tensor(0.0, device=high_radar_bev.device)
                batch_dict['loss_bev_combined'] = torch.tensor(0.0, device=high_radar_bev.device)
        
        tb_dict={
            'low_feature_loss' : low_distill_loss.item(),
            'high_distill_loss' : high_distill_loss.item(),
            'distll_loss' : distill_loss.item(),
            'low_distill_de_8x_loss' : de_8x_feature_loss.item(),
            'low_distill_loss' : feature_loss.item(),
            'mask_loss' : mask_loss.item(),
            'mask_de_8x_loss': de_8x_mask_loss.item(),
            **tig_distill_dict  # TiGDistill losses 포함
        }
        return distill_loss, tb_dict
    
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
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
    
