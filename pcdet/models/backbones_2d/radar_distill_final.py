import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pcdet.utils.box_utils import center_to_corner_box2d
from ...ops.basicblock.modules.Basicblock_convn import ConvNeXtBlock
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from functools import partial
from .base_bev_backbone import BaseBEVBackboneV2


def extract_keypoint_features_from_bev(
    bev_features, gt_boxes, point_cloud_range, voxel_size, num_keypoints=256, enlarge_factor=1.0
):
    
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
    # 추가 옵션 (Config로 관리 권장)
    chunk_size: int = 8192,
    oversample_rate: float = 3.0,
    deterministic: bool = False,
    precomputed_coords=None,
    return_coords: bool = False
):
    batch_size, num_channels, H, W = bev_features.shape
    inner_keypoints_list = []
    outer_keypoints_list = []
    stored_inner_coords = [] if return_coords else None
    stored_outer_coords = [] if return_coords else None

    device = bev_features.device
    dtype = bev_features.dtype
    pc_range = torch.tensor(point_cloud_range, device=device, dtype=dtype)
    voxel_size_tensor = torch.tensor(voxel_size, device=device, dtype=dtype)
    
    target_outer = num_keypoints_outer if num_keypoints_outer is not None else max(num_keypoints_inner, 1)

    # [Helper] 좌표 정규화 함수
    def get_normalized_grid(world_coords):
        bev_coords_x = (world_coords[..., 0] - pc_range[0]) / voxel_size_tensor[0]
        bev_coords_y = (world_coords[..., 1] - pc_range[1]) / voxel_size_tensor[1]
        norm_x = (bev_coords_x / (W - 1)) * 2.0 - 1.0
        norm_y = (bev_coords_y / (H - 1)) * 2.0 - 1.0
        return torch.stack([norm_x, norm_y], dim=-1)

    coord_cursor = 0
    for b in range(batch_size):
        batch_bev = bev_features[b:b+1]
        batch_boxes = gt_boxes[b]

        mask = batch_boxes.sum(dim=1) != 0
        valid_boxes = batch_boxes[mask]
        num_valid = valid_boxes.shape[0]

        if num_valid == 0:
            continue

        use_precomputed = precomputed_coords is not None

        centers = valid_boxes[:, 0:2]
        dims = valid_boxes[:, 3:5]
        angles = valid_boxes[:, 6]
        
        dims_inner = dims * inner_factor
        dims_outer = dims * outer_factor
        
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        rot_mat = torch.stack([
            torch.stack([cos_a, -sin_a], dim=-1),
            torch.stack([sin_a, cos_a], dim=-1)
        ], dim=1)

        if use_precomputed:
            try:
                world_inner = precomputed_coords['inner'][coord_cursor].to(device=device, dtype=dtype)
                final_world_outer = precomputed_coords['outer'][coord_cursor].to(device=device, dtype=dtype)
            except (IndexError, KeyError, TypeError):
                raise ValueError('Invalid precomputed_coords passed to extract_dual_region_keypoints')
        else:
            # =======================================================
            # 1. Inner Region (Grid or Random)
            # =======================================================
            if x_sample_num is not None and y_sample_num is not None:
                xs = torch.linspace(-0.5, 0.5, steps=x_sample_num, device=device, dtype=dtype)
                ys = torch.linspace(-0.5, 0.5, steps=y_sample_num, device=device, dtype=dtype)
                yy, xx = torch.meshgrid(ys, xs, indexing='ij')
                base_grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
                local_inner = base_grid.unsqueeze(0) * dims_inner.unsqueeze(1)
            else:
                rand_pts = torch.rand(num_valid, num_keypoints_inner, 2, device=device, dtype=dtype) - 0.5
                local_inner = rand_pts * dims_inner.unsqueeze(1)

            pts_rotated_inner = torch.matmul(local_inner, rot_mat.transpose(1, 2))
            world_inner = centers.unsqueeze(1) + pts_rotated_inner

            # =======================================================
            # 2. Outer Region (Grid or Random) - Optimized
            # =======================================================
            if outer_x_sample_num is not None and outer_y_sample_num is not None:
                xs_o = torch.linspace(-0.5, 0.5, steps=outer_x_sample_num, device=device, dtype=dtype)
                ys_o = torch.linspace(-0.5, 0.5, steps=outer_y_sample_num, device=device, dtype=dtype)
                yy_o, xx_o = torch.meshgrid(ys_o, xs_o, indexing='ij')
                base_grid_o = torch.stack([xx_o.reshape(-1), yy_o.reshape(-1)], dim=-1)
                local_outer_cand = base_grid_o.unsqueeze(0) * dims_outer.unsqueeze(1)
                num_cand = local_outer_cand.shape[1]
                use_random_score = False
            else:
                num_cand = int(target_outer * oversample_rate)
                rand_pts_o = torch.rand(num_valid, num_cand, 2, device=device, dtype=dtype) - 0.5
                local_outer_cand = rand_pts_o * dims_outer.unsqueeze(1)
                use_random_score = not deterministic

            half_inner = dims_inner / 2.0
            inside_self = (torch.abs(local_outer_cand[..., 0]) <= half_inner[:, 0:1]) & \
                          (torch.abs(local_outer_cand[..., 1]) <= half_inner[:, 1:2])

            pts_rotated_outer = torch.matmul(local_outer_cand, rot_mat.transpose(1, 2))
            world_outer_cand = centers.unsqueeze(1) + pts_rotated_outer

            flat_candidates = world_outer_cand.view(-1, 2)
            is_collision_flat = torch.zeros(flat_candidates.shape[0], dtype=torch.bool, device=device)
            half_dims_inner_all = dims_inner / 2.0

            for i in range(0, flat_candidates.shape[0], chunk_size):
                chunk_pts = flat_candidates[i : i + chunk_size]
                rel_pos = chunk_pts.unsqueeze(0) - centers.unsqueeze(1)
                local_pos = torch.einsum('ncj, nkj -> nck', rel_pos, rot_mat)
                in_box = (torch.abs(local_pos[..., 0]) <= half_dims_inner_all[:, 0:1]) & \
                         (torch.abs(local_pos[..., 1]) <= half_dims_inner_all[:, 1:2])
                is_collision_flat[i : i + chunk_size] = in_box.any(dim=0)

            is_collision = is_collision_flat.view(num_valid, num_cand)
            invalid_mask = inside_self | is_collision

            valid_score = (~invalid_mask).float()
            if use_random_score:
                valid_score += torch.rand_like(valid_score) * 0.1

            _, sorted_indices = torch.topk(valid_score, k=num_cand, dim=1)
            num_valid_per_box = (~invalid_mask).sum(dim=1)
            idx_grid = torch.arange(target_outer, device=device).unsqueeze(0).expand(num_valid, -1)
            safe_num_valid = num_valid_per_box.clamp(min=1).unsqueeze(1)
            refined_indices_pos = idx_grid % safe_num_valid
            final_indices = torch.gather(sorted_indices, 1, refined_indices_pos)

            final_world_outer = torch.gather(
                world_outer_cand, 1,
                final_indices.unsqueeze(-1).expand(-1, -1, 2)
            )

            zero_valid_mask = (num_valid_per_box == 0)
            if zero_valid_mask.any():
                corner_signs = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]], device=device, dtype=dtype) * 0.5
                corners_local = corner_signs.unsqueeze(0).repeat(1, (target_outer + 3) // 4, 1)[:, :target_outer, :]
                bad_indices = torch.nonzero(zero_valid_mask).squeeze(1)

                c_local = corners_local.repeat(bad_indices.shape[0], 1, 1) * dims_outer[bad_indices].unsqueeze(1)
                c_rot = rot_mat[bad_indices]
                c_center = centers[bad_indices]
                c_world = c_center.unsqueeze(1) + torch.matmul(c_local, c_rot.transpose(1, 2))
                final_world_outer[bad_indices] = c_world
        
        features_inner = F.grid_sample(
            batch_bev, get_normalized_grid(world_inner).unsqueeze(0), 
            mode='bilinear', align_corners=True
        ).squeeze(0).permute(1, 2, 0)
        inner_keypoints_list.append(features_inner)

        features_outer = F.grid_sample(
            batch_bev, get_normalized_grid(final_world_outer).unsqueeze(0), 
            mode='bilinear', align_corners=True
        ).squeeze(0).permute(1, 2, 0)

        outer_keypoints_list.append(features_outer)

        if return_coords:
            stored_inner_coords.append(world_inner)
            stored_outer_coords.append(final_world_outer)

        coord_cursor += 1

    if len(inner_keypoints_list) > 0:
        if return_coords:
            return inner_keypoints_list, outer_keypoints_list, {'inner': stored_inner_coords, 'outer': stored_outer_coords}
        return inner_keypoints_list, outer_keypoints_list
    else:
        if return_coords:
            return [], [], {'inner': [], 'outer': []}
        return [], []
    

def clip_sigmoid(x, eps=1e-4):
    
    # FIXME change back!
    # y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    y = torch.clamp(x.sigmoid(), min=eps, max=1 - eps)
    return y


def compute_kl_loss(student_logits, teacher_logits, temperature=1.0):
    
    # 1. Student는 LogSoftmax, Teacher는 Softmax를 취해야 함 (KL 공식 특성상)
    # dim=-1: 마지막 차원(보통 관계 대상)에 대해 확률 분포를 만듦
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
    
    # 2. KLDivLoss 계산 (reduction='batchmean' 권장)
    # KL(P || Q) = sum(P * (log P - log Q)) 형태이지만 
    # PyTorch KLDivLoss는 Input이 Log P, Target이 P를 받음.
    kl_loss = F.kl_div(student_log_prob, teacher_prob, reduction='batchmean')
    
    # 3. Temperature의 제곱만큼 Gradient가 작아지므로 보정
    return kl_loss * (temperature ** 2)
            

class Radar_Distill(BaseBEVBackboneV2):
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg, **kwargs)
        self.model_cfg = model_cfg
        
        # Class-adaptive feature selection strategy
        # nuScenes class indices: ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 
        #                          'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
        self.HIGH_LEVEL_CLASSES = [1, 2, 3]  # truck, bus, trailer - benefit from high-level semantic features
        self.LOW_LEVEL_ONLY_CLASSES = [8, 9]  # traffic_cone, barrier - low-level spatial details only
        self.BALANCED_CLASSES = [0, 4, 5, 6, 7]  # car, construction_vehicle, pedestrian, motorcycle, bicycle
        
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
    
    @torch.no_grad()
    def _get_class_weight_mask(self, batch_dict, H, W, device, dtype):
        """
        Generate class-adaptive weight mask for high-level features.
        
        Different object classes benefit from different feature levels:
        - Large moving vehicles (truck, bus, trailer) → high-level semantic features
        - Static objects (barrier, traffic_cone) → low-level spatial details only
        - Balanced classes → moderate high-level contribution
        
        Args:
            batch_dict: dictionary containing gt_boxes
            H, W: height and width of BEV feature map
            device, dtype: torch device and data type
            
        Returns:
            weight_mask: (B, 1, H, W) - per-pixel weight for high-level loss
                         1.0 for high-level classes (truck, bus, trailer)
                         0.0 for low-level only classes (barrier, traffic_cone)
                         0.5 for balanced classes (car, pedestrian, etc.)
        """
        B = batch_dict['batch_size']
        weight_mask = torch.ones((B, 1, H, W), device=device, dtype=dtype)
        
        gt_boxes = batch_dict['gt_boxes']
        if not torch.is_tensor(gt_boxes):
            gt_boxes = torch.from_numpy(gt_boxes).to(device=device, dtype=dtype)
        else:
            gt_boxes = gt_boxes.to(device=device, dtype=dtype)
        
        pixel_size = float(self.voxel_size[0])
        pc_range = self.point_cloud_range
        x_min, y_min = pc_range[0], pc_range[1]
        x_max, y_max = pc_range[3], pc_range[4]
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        for b in range(B):
            valid_mask = gt_boxes[b].sum(dim=1) != 0
            boxes = gt_boxes[b][valid_mask]
            
            if boxes.shape[0] == 0:
                continue
            
            for box in boxes:
                class_id = int(box[7])  # Last element is class ID
                
                # Determine weight based on class category
                if class_id in self.LOW_LEVEL_ONLY_CLASSES:
                    weight = 0.0  # No high-level features for barrier, traffic_cone
                elif class_id in self.HIGH_LEVEL_CLASSES:
                    weight = 1.0  # Full high-level features for truck, bus, trailer
                elif class_id in self.BALANCED_CLASSES:
                    weight = 0.5  # Moderate high-level features for car, pedestrian, etc.
                else:
                    weight = 0.5  # Default: balanced
                
                # Create box mask in BEV grid
                center = box[0:2]
                size = box[3:5]
                angle = box[6]
                
                # Create a local grid around the box (similar to radar distribution projection)
                # Use box size * 1.2 to cover the object region
                coverage = max(size).item() * 1.2
                grid_size = int(coverage / pixel_size)
                grid_size = max(16, min(grid_size, 128))
                if grid_size % 2 == 0:
                    grid_size += 1
                
                # Generate local coordinates
                half_grid_world = (grid_size * pixel_size) / 2.0
                local_x = torch.linspace(-half_grid_world, half_grid_world, grid_size, device=device, dtype=dtype)
                local_y = torch.linspace(-half_grid_world, half_grid_world, grid_size, device=device, dtype=dtype)
                yy, xx = torch.meshgrid(local_y, local_x, indexing='ij')
                local_coords = torch.stack([xx, yy], dim=-1)
                
                # Rotate to world frame
                cos_a, sin_a = torch.cos(angle), torch.sin(angle)
                rot_mat = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=device, dtype=dtype)
                
                local_flat = local_coords.reshape(-1, 2)
                world_flat = torch.matmul(local_flat, rot_mat.T) + center.unsqueeze(0)
                
                # World → Global BEV grid index
                bev_x_idx = ((world_flat[:, 0] - x_min) / x_range * W).long().clamp(0, W - 1)
                bev_y_idx = ((world_flat[:, 1] - y_min) / y_range * H).long().clamp(0, H - 1)
                
                # Apply class-specific weight to this box region (VECTORIZED - no for loop!)
                weight_mask[b, 0, bev_y_idx, bev_x_idx] = weight
        
        return weight_mask
    
    
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

        # ================== TiGDistill-BEV Inter-channel Distillation Loss with Dual Region ==================
        # Temporarily disabled while focusing on low-level losses only.
        enable_tig_distill = False
        distill_cfg = self.model_cfg.get('TIG_DISTILL', {}) if enable_tig_distill else {}
        tig_distill_dict = {}
        if enable_tig_distill and distill_cfg:
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
            # Use high-level BEV features (DenseEnc outputs) for TiG distillation
            # to capture semantically richer context along object boundaries.
            teacher_bev_features = high_lidar_bev
            student_bev_features = high_radar_bev
            
            # Extract dual region keypoint features from both teacher and student
            teacher_inner_list, teacher_outer_list, keypoint_coords = extract_dual_region_keypoints(
                teacher_bev_features, gt_boxes, 
                self.point_cloud_range, self.voxel_size, 
                num_keypoints_inner, num_keypoints_outer,
                inner_factor, outer_factor,
                x_sample_num, y_sample_num,
                outer_x_sample_num, outer_y_sample_num,
                return_coords=True
            )

            if len(keypoint_coords['inner']) > 0:
                student_inner_list, student_outer_list = extract_dual_region_keypoints(
                    student_bev_features, gt_boxes, 
                    self.point_cloud_range, self.voxel_size, 
                    num_keypoints_inner, num_keypoints_outer,
                    inner_factor, outer_factor,
                    x_sample_num, y_sample_num,
                    outer_x_sample_num, outer_y_sample_num,
                    precomputed_coords=keypoint_coords
                )
            else:
                student_inner_list, student_outer_list = [], []

            # Compute inter-channel and inter-keypoint correlation loss
            loss_bev_ic = 0.0  # inter-channel loss (inner region only)
            loss_bev_ik = 0.0  # inter-keypoint loss (inner region only)
            loss_bev_contrastive = 0.0  # contrastive loss (boundary learning)
            # Detailed boundary loss components
            # loss_inner_align = 0.0
            # loss_outer_align = 0.0
            loss_boundary_consist = 0.0
            loss_cross_boundary = 0.0
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
                        
                        # ============= Outer Region: Teacher-Student Boundary Learning =============
                        # 목표: Teacher의 box 내부/외부 경계 관계를 Student에게 전달
                        # 
                        # Contrastive Learning 구성:
                        # - Anchor: Student inner features (GT box 내부)
                        # - Positive: Teacher inner features (같은 객체, 정렬 대상)
                        # - Negative: Outer region features (배경, 구분 대상)
                        #
                        # 4가지 Loss Components:
                        # 1. Inner Alignment: Student inner ↔ Teacher inner (객체 영역 일치)
                        # 2. Outer Alignment: Student outer ↔ Teacher outer (배경 영역 일치)
                        # 3. Boundary Consistency: Teacher의 inner-outer 거리 → Student 학습
                        # 4. Cross-Boundary: Student inner ↔ Teacher outer (경계 명확화)
                        f_teacher_outer = teacher_outer[obj_idx]  # (num_keypoints_outer, C)
                        f_student_outer = student_outer[obj_idx]  # (num_keypoints_outer, C)
                        
                        t_inner_norm = f_teacher_inner
                        t_outer_norm = f_teacher_outer
                        s_inner_norm = f_student_inner
                        s_outer_norm = f_student_outer
                        
                        # 3. Boundary Consistency: Teacher의 inner-outer 경계 거리를 Student도 학습
                        # KL Divergence를 사용하여 관계 분포를 정렬
                        
                        # Similarity Matrix 계산 (scaling factor 추가)
                        # (N_in, N_out) Teacher의 관계 맵
                        scale_factor = t_inner_norm.shape[1] ** -0.5  # sqrt(dim)로 나눔
                        teacher_rel_logits = torch.mm(t_inner_norm, t_outer_norm.T) * scale_factor
                        
                        # (N_in, N_out) Student의 관계 맵
                        student_rel_logits = torch.mm(s_inner_norm, s_outer_norm.T) * scale_factor
                        
                        # KL Divergence로 경계 일관성 손실 계산
                        # "Inner 포인트 하나가 Outer 포인트들에 대해 가지는 분포"를 맞춤 (dim=-1 기준)
                        boundary_consistency = compute_kl_loss(student_rel_logits, teacher_rel_logits, temperature=4.0)
                        
                        # 4. Cross-Boundary Contrast: Student inner와 Teacher outer는 멀어야 함
                        cross_boundary_sim = torch.mm(s_inner_norm, t_outer_norm.T)
                        cross_boundary_loss = torch.clamp(cross_boundary_sim + contrastive_margin, min=0.0).mean()
                        
                        # Total boundary loss
                        contrastive_loss = boundary_consistency + cross_boundary_loss
                        loss_bev_contrastive += contrastive_loss
                        
                        # Accumulate individual components for logging
                        # loss_inner_align += inner_alignment
                        # loss_outer_align += outer_alignment
                        loss_boundary_consist += boundary_consistency
                        loss_cross_boundary += cross_boundary_loss
                        
                        num_objects += 1

            if num_objects > 0:
                loss_bev_ic = loss_bev_ic / num_objects
                loss_bev_ik = loss_bev_ik / num_objects
                loss_bev_contrastive = loss_bev_contrastive / num_objects
                # loss_inner_align = loss_inner_align / num_objects
                # loss_outer_align = loss_outer_align / num_objects
                loss_boundary_consist = loss_boundary_consist / num_objects
                loss_cross_boundary = loss_cross_boundary / num_objects
                
                # Add all losses to total distill_loss
                distill_loss = distill_loss + \
                               loss_bev_ic * bev_ic_weight + \
                               loss_bev_ik * bev_ik_weight + \
                               loss_bev_contrastive * bev_contrastive_weight
                
                # Store in dict for logging
                tig_distill_dict['loss_bev_ic'] = loss_bev_ic.item()
                tig_distill_dict['loss_bev_ik'] = loss_bev_ik.item()
                tig_distill_dict['loss_bev_contrastive'] = loss_bev_contrastive.item()
                # Detailed boundary loss components
                # tig_distill_dict['loss_inner_align'] = loss_inner_align.item()
                # tig_distill_dict['loss_outer_align'] = loss_outer_align.item()
                tig_distill_dict['loss_boundary_consist'] = loss_boundary_consist.item()
                tig_distill_dict['loss_cross_boundary'] = loss_cross_boundary.item()
                
                # Store in batch_dict for pillarnet to use (for backward compatibility)
                batch_dict['loss_bev_ic'] = loss_bev_ic
                batch_dict['loss_bev_ik'] = loss_bev_ik
                batch_dict['loss_bev_contrastive'] = loss_bev_contrastive
                batch_dict['loss_bev_combined'] = loss_bev_ic + loss_bev_ik + loss_bev_contrastive
            else:
                tig_distill_dict['loss_bev_ic'] = 0.0
                tig_distill_dict['loss_bev_ik'] = 0.0
                tig_distill_dict['loss_bev_contrastive'] = 0.0
                # tig_distill_dict['loss_inner_align'] = 0.0
                # tig_distill_dict['loss_outer_align'] = 0.0
                tig_distill_dict['loss_boundary_consist'] = 0.0
                tig_distill_dict['loss_cross_boundary'] = 0.0
                
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
    
