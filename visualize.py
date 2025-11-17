#!/usr/bin/env python3
"""
Simple BEV Visualization Script
Visualizes radar points in BEV with GT bounding boxes
Shows empty bboxes (boxes without points) in green
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

# Class names for NuScenes dataset
CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

def to_np(x):
    """Convert tensor to numpy array"""
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

def obb_corners_2d(cx, cy, dx, dy, yaw):
    """Calculate 2D oriented bounding box corners"""
    hx, hy = dx / 2.0, dy / 2.0
    corners = np.array([
        [ hx,  hy],
        [ hx, -hy],
        [-hx, -hy],
        [-hx,  hy],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    rot = corners @ R.T
    rot[:, 0] += cx
    rot[:, 1] += cy
    return rot  # (4,2) in (X,Y)

def points_in_poly(pts, poly):
    """Check if points are inside polygon"""
    path = MplPath(poly)
    return path.contains_points(pts)

def in_view_box(corners, lim=54.0):
    """Check if box corners are within view range"""
    xs, ys = corners[:, 0], corners[:, 1]
    return np.any((xs >= -lim) & (xs <= lim) & (ys >= -lim) & (ys <= lim))

def setup_environment():
    """Setup configuration and dataloader"""
    # Change to tools directory so relative paths in config work
    os.chdir(str(project_root / 'tools'))
    
    # Load config
    cfg_file = 'cfgs/radar_distill/radar_distill_train.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    
    # Override INFO_PATH with available data files
    cfg.DATA_CONFIG.INFO_PATH = {
        'train': ['/home/byounggun/RadarDistill/data/nuscenes/v1.0-trainval/nuscenes_infos_6radar_10sweeps_train.pkl'],
        'test': ['/home/byounggun/RadarDistill/data/nuscenes/v1.0-trainval/nuscenes_infos_6radar_10sweeps_val.pkl'],
    }
    
    # Create logger
    logger = common_utils.create_logger()
    
    # Build dataloader - use test split (validation data)
    dataset, dataloader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,  # Single sample for visualization
        dist=False, workers=0, logger=logger, training=False
    )
    
    return dataloader, logger, cfg

def bev_corners_xy(box):
    """Calculate 2D BEV box corners from [x,y,z,dx,dy,dz,yaw]"""
    x, y = float(box[0]), float(box[1])
    dx, dy = float(box[3]), float(box[4])
    yaw = float(box[6])
    
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([
        [ dx/2,  dy/2],
        [ dx/2, -dy/2],
        [-dx/2, -dy/2],
        [-dx/2,  dy/2]
    ], dtype=np.float32)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (local @ R.T) + np.array([x, y], dtype=np.float32)

def world_to_pixels(xy, x_lim, y_lim, W, H):
    """Convert world coordinates to pixel coordinates
    
    Args:
        xy: (N, 2) array of [x, y] in world coordinates
        x_lim: (xmin, xmax) world range
        y_lim: (ymin, ymax) world range
        W, H: image width and height in pixels
    
    Returns:
        (N, 2) array of [u, v] pixel coordinates
    """
    xmin, xmax = x_lim
    ymin, ymax = y_lim
    
    # Normalize to [0, 1]
    u = (xy[... , 0] - xmin) / max(xmax - xmin, 1e-8)
    v = (xy[... , 1] - ymin) / max(ymax - ymin, 1e-8)
    
    # Convert to pixel coordinates
    U = u * (W - 1)
    V = v * (H - 1)
    
    return np.stack([U, V], axis=-1)

def visualize_sample(batch_dict, sample_idx, output_dir='visualize/results'):
    """Visualize BEV prediction heatmap with radar points and GT bounding boxes"""
    os.makedirs(output_dir, exist_ok=True)
    
    # ── 1) Get BEV feature map (Low-Level Features) ──
    if 'radar_multi_scale_2d_features' in batch_dict and \
       'radar_spatial_features_8x_2' in batch_dict['radar_multi_scale_2d_features']:
        
        # (B, C, H, W) 텐서 로드
        bev_features_tensor = batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_2']
        
        # (B, C, H, W) -> (C, H, W) (첫 번째 배치 아이템 선택)
        bev_features = bev_features_tensor[0] 
        
        # =================================================================================
        # ▼▼▼ [핵심 수정] Mean 대신 Max(최댓값)를 사용하여 맵 생성 ▼▼▼
        # =================================================================================
        # (C, H, W) -> (H, W) by taking max across channels
        bev_map_tensor = bev_features.max(dim=0)[0] # max는 (values, indices) 반환
        bev_map = to_np(bev_map_tensor)
        # =================================================================================
        
        feature_source = "Low-Level Features (radar_spatial_features_8x_2 - Max)"
        
        # =================================================================================
        # ▼▼▼ [핵심 수정] 0 미만의 값은 무시하고, 양수 값만 정규화 ▼▼▼
        # =================================================================================
        # 1. 0 미만의 값을 0으로 클리핑 (ReLU와 유사)
        bev_map[bev_map < 0] = 0
        
        # 2. 이제 (0, max) 범위를 (0, 1) 범위로 정규화
        bev_map_min = bev_map.min() # 0이 됨
        bev_map_max = bev_map.max()
        bev_map = (bev_map - bev_map_min) / (bev_map_max - bev_map_min + 1e-8)
        # =================================================================================
        
        H, W = bev_map.shape
        
    else:
        print("Error: Could not find 'radar_spatial_features_8x_2'.")
        print(f"DEBUG: Available keys in radar_multi_scale_2d_features: {batch_dict.get('radar_multi_scale_2d_features', {}).keys()}")
        return
    # ────────────────────────────────────────────────────────
    
    # ── 2) Load points ──
    pts = to_np(batch_dict["points"])  # (N,7): [b, x, y, z, rcs, vx, vy]
    b = 0
    mask_b = (pts[:, 0] == b) if pts.shape[1] >= 7 else np.ones(len(pts), dtype=bool)
    x_world = pts[mask_b, 1]   # forward
    y_world = pts[mask_b, 2]   # left
    
    # ── 3) Load GT boxes ──
    gt = to_np(batch_dict["gt_boxes"])  # (B, M, D)
    boxes = gt[b]
    M, D = boxes.shape
    valid = ~(np.all(np.isclose(boxes, 0.0), axis=1))
    boxes = boxes[valid]
    
    # ── 4) Coordinate range ──
    x_min, x_max = -54.0, 54.0
    y_min, y_max = -54.0, 54.0
    
    # ── 5) Plot ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Display BEV feature map
    im = ax.imshow(bev_map, cmap='viridis', origin='lower', extent=[0, W, 0, H])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Feature Intensity')
    
    # Convert world points to pixel coordinates
    points_xy = np.stack([x_world, y_world], axis=1)  # (N, 2)
    points_uv = world_to_pixels(points_xy, (x_min, x_max), (y_min, y_max), W, H)
    
    # Plot radar points (red, small)
    ax.scatter(points_uv[:, 0], points_uv[:, 1], s=1, c='red', alpha=0.6, zorder=2)
    
    # Draw GT boxes with rotation
    for box in boxes:
        # Get box corners in world coordinates
        corners_world = bev_corners_xy(box)  # (4, 2) [x, y]
        
        # Convert to pixel coordinates
        corners_pixels = world_to_pixels(corners_world, (x_min, x_max), (y_min, y_max), W, H)
        
        # Draw box (close the polygon)
        ax.plot(
            corners_pixels[[0, 1, 2, 3, 0], 0], 
            corners_pixels[[0, 1, 2, 3, 0], 1], 
            color='lime', linewidth=2, alpha=0.9, zorder=3
        )
    
    # Axis settings
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('BEV X (pixels)', fontsize=12)
    ax.set_ylabel('BEV Y (pixels)', fontsize=12) # 오타 수정됨
    ax.set_title(f'BEV Map with Radar Points & GT Boxes - {feature_source}', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.3)
    
    # Save figure
    # output_path = os.stat(output_dir, f'bev_low_level_max_{sample_idx:03d}.png') # 파일 이름 변경
    output_path = os.path.join(output_dir, f'bev_low_level_max_{sample_idx:03d}.png') # 파일 이름 변경
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved: {output_path}")
    return output_path

def main():
    """Main function to run visualization"""
    print("Setting up environment...")
    dataloader, logger, cfg = setup_environment()
    
    print(f"Dataset contains {len(dataloader)} samples")
    print("Starting visualization...")
    
    # Visualize first few samples
    num_samples_to_visualize = 5
    
    # Build model for forward pass to get BEV features
    ckpt_path = '/home/byounggun/RadarDistill/output2/radar_distill/radar_distill_train/default/ckpt/checkpoint_epoch_40.pth'
    # ckpt_path = '/home/byounggun/RadarDistill/baseline.pth'
    dataset = dataloader.dataset
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=False)
    model.cuda()
    
    # Set model to train() mode to get all intermediate feature outputs
    model.train() 
    
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(dataloader):
            if batch_idx >= num_samples_to_visualize:
                break
                
            print(f"\nProcessing sample {batch_idx + 1}/{num_samples_to_visualize}")
            
            # Move to GPU
            for key, val in batch_dict.items():
                if isinstance(val, torch.Tensor):
                    batch_dict[key] = val.cuda()
                elif isinstance(val, np.ndarray):
                    if np.issubdtype(val.dtype, np.number):
                        batch_dict[key] = torch.from_numpy(val).cuda()
            
            # Call forward. This populates batch_dict with all intermediate features
            _ = model.forward(batch_dict)
            
            # Visualize (batch_dict만 전달)
            output_path = visualize_sample(batch_dict, batch_idx)
            
            # Print some info about the sample
            if 'gt_boxes' in batch_dict:
                gt_boxes = to_np(batch_dict['gt_boxes'])[0]
                valid_gt = ~(np.all(np.isclose(gt_boxes, 0.0), axis=1))
                num_gt = valid_gt.sum()
                print(f"  - Ground truth boxes: {num_gt}")
            
            if 'points' in batch_dict:
                pts = to_np(batch_dict['points'])
                num_pts = len(pts[pts[:, 0] == 0]) if pts.shape[1] >= 1 else len(pts)
                print(f"  - Radar points: {num_pts}")
    
    print(f"\nVisualization completed! Check the 'visualize/results' directory for output images.")

if __name__ == '__main__':
    main()