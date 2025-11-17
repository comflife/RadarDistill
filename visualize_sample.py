#!/usr/bin/env python3
"""
Simple BEV Visualization Script
Visualizes BEV Spatial Features (Low level, Channel Avg, RAW)
"""

import copy
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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

DEFAULT_VIS_CFG = {
    'aggregation_mode': 'mean',
    'center_channels': False,
    'apply_relu': True,
    'post_sigmoid': False,
    'log_scale_gain': 0.0,
    'smoothing_kernel': 1,
    'normalization': 'percentile',
    'low_percentile': 0.5,
    'high_percentile': 99.5,
    'use_world_coords': False,
    'show_colorbar': False,
    'show_box_labels': False,
    'cmap': 'viridis',
    'interpolation': 'bilinear',
    'active_percentile': None,
}

FEATURE_VIS_OVERRIDES = {
    # Student radar BEV (low-level)
    'radar_spatial_features_2d_8x': {
        'aggregation_mode': 'l2',
        'center_channels': True,
        'apply_relu': False,
        'post_sigmoid': False,
        'log_scale_gain': 0.0,
        'smoothing_kernel': 3,
        'low_percentile': 5.0,
        'high_percentile': 99.7,
        'show_box_labels': False,
        'active_percentile': 98.5,
    },
    'radar_prediction_heatmap': {
        'aggregation_mode': 'mean',
        'center_channels': False,
        'apply_relu': False,
        'post_sigmoid': False,
        'log_scale_gain': 5.0,
        'smoothing_kernel': 3,
        'normalization': 'minmax',
        'low_percentile': 0.0,
        'high_percentile': 100.0,
        'show_box_labels': False,
        'active_percentile': 99.5,
    },
    # Teacher lidar BEV (high-level)
    'spatial_features_2d_8x': {
        'aggregation_mode': 'l2',
        'center_channels': True,
        'apply_relu': False,
        'post_sigmoid': True,
        'log_scale_gain': 8.0,
        'smoothing_kernel': 3,
        'low_percentile': 1.0,
        'high_percentile': 99.5,
        'show_box_labels': False,
        'active_percentile': 99.0,
    },
    # Teacher lidar BEV after aggregation (final)
    'spatial_features_2d': {
        'aggregation_mode': 'l2',
        'center_channels': True,
        'apply_relu': False,
        'post_sigmoid': True,
        'log_scale_gain': 8.0,
        'smoothing_kernel': 5,
        'low_percentile': 1.0,
        'high_percentile': 99.8,
        'show_box_labels': False,
        'active_percentile': 99.0,
    },
    'lidar_prediction_heatmap': {
        'aggregation_mode': 'mean',
        'center_channels': False,
        'apply_relu': False,
        'post_sigmoid': False,
        'log_scale_gain': 5.0,
        'smoothing_kernel': 3,
        'normalization': 'minmax',
        'low_percentile': 0.0,
        'high_percentile': 100.0,
        'show_box_labels': False,
        'active_percentile': None,
    },
    'lidar_prediction_heatmap_distill': {
        'aggregation_mode': 'mean',
        'center_channels': False,
        'apply_relu': False,
        'post_sigmoid': False,
        'log_scale_gain': 5.0,
        'smoothing_kernel': 3,
        'normalization': 'minmax',
        'low_percentile': 0.0,
        'high_percentile': 100.0,
        'show_box_labels': False,
        'active_percentile': None,
    },
    'lidar_prediction_heatmap_baseline': {
        'aggregation_mode': 'mean',
        'center_channels': False,
        'apply_relu': False,
        'post_sigmoid': False,
        'log_scale_gain': 5.0,
        'smoothing_kernel': 3,
        'normalization': 'minmax',
        'low_percentile': 0.0,
        'high_percentile': 100.0,
        'show_box_labels': False,
        'active_percentile': None,
    },
    'lidar_prediction_heatmap_diff': {
        'aggregation_mode': 'mean',
        'center_channels': False,
        'apply_relu': False,
        'post_sigmoid': False,
        'log_scale_gain': 0.0,
        'smoothing_kernel': 3,
        'normalization': 'signed',
        'low_percentile': 0.0,
        'high_percentile': 100.0,
        'show_box_labels': False,
        'active_percentile': None,
        'cmap': 'seismic',
        'show_colorbar': True,
    },
}


def normalize_bev_map(bev_map, method='percentile', low_percentile=1.0, high_percentile=99.5):
    """Normalize BEV map to [0, 1] for visualization."""
    bev_np = np.nan_to_num(bev_map.astype(np.float32), copy=False)
    finite_mask = np.isfinite(bev_np)
    if not finite_mask.any():
        return np.zeros_like(bev_np, dtype=np.float32), (0.0, 1.0)
    values = bev_np[finite_mask]
    if method == 'minmax':
        vmin = float(values.min())
        vmax = float(values.max())
    elif method == 'signed':
        max_abs = float(np.max(np.abs(values)))
        if max_abs <= 1e-12:
            return np.full_like(bev_np, 0.5, dtype=np.float32), (-0.0, 0.0)
        normalized = (bev_np / max_abs) * 0.5 + 0.5
        return np.clip(normalized, 0.0, 1.0), (-max_abs, max_abs)
    else:
        vmin = float(np.percentile(values, low_percentile))
        vmax = float(np.percentile(values, high_percentile))
    if vmax <= vmin + 1e-12:
        return np.zeros_like(bev_np, dtype=np.float32), (vmin, vmax)
    normalized = (bev_np - vmin) / (vmax - vmin)
    return np.clip(normalized, 0.0, 1.0), (vmin, vmax)

def format_class_label(raw_class_id):
    """Map raw class ids to display labels, handling both 0/1-based ids."""
    if raw_class_id is None or not np.isfinite(raw_class_id):
        return None
    raw_idx = int(round(float(raw_class_id)))
    if 0 <= raw_idx < len(CLASS_NAMES):
        return CLASS_NAMES[raw_idx]
    shifted_idx = raw_idx - 1
    if 0 <= shifted_idx < len(CLASS_NAMES):
        return CLASS_NAMES[shifted_idx]
    return f'class {raw_idx}'

def world_to_pixels(xy, x_lim, y_lim, W, H):
    """Convert world coordinates (meters) to BEV pixel coordinates."""
    xmin, xmax = x_lim
    ymin, ymax = y_lim
    u = (xy[..., 0] - xmin) / max(xmax - xmin, 1e-8)
    v = (xy[..., 1] - ymin) / max(ymax - ymin, 1e-8)
    U = u * (W - 1)
    V = v * (H - 1)
    return np.stack([U, V], axis=-1)

def to_np(x):
    """Convert tensor to numpy array"""
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

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

# =================================================================================
# ▼▼▼ [핵심] "Channel Avg" 피처 시각화 (후처리 없음) ▼▼▼
# =================================================================================
def aggregate_bev_channels(bev_tensor, mode='mean', center_channels=False, apply_relu=False):
    """Collapse channel dimension with optional centering."""
    if center_channels:
        bev_tensor = bev_tensor - bev_tensor.mean(dim=(1, 2), keepdim=True)
    if apply_relu:
        bev_tensor = torch.relu(bev_tensor)
    if mode == 'mean':
        return bev_tensor.mean(dim=0)
    if mode == 'max':
        return bev_tensor.max(dim=0).values
    if mode == 'sum':
        return bev_tensor.sum(dim=0)
    if mode == 'l2':
        return torch.sqrt((bev_tensor ** 2).sum(dim=0).clamp_min(1e-12))
    # Default to L2 / RMS aggregation for sharper activations.
    return torch.sqrt(torch.mean(bev_tensor ** 2, dim=0).clamp_min(1e-12))


def postprocess_bev_map(bev_map_tensor, post_sigmoid=False, log_scale_gain=0.0, smoothing_kernel=1):
    """Apply optional non-linearities and smoothing to the BEV map."""
    if post_sigmoid:
        bev_map_tensor = torch.sigmoid(bev_map_tensor)
    if log_scale_gain and log_scale_gain > 0:
        bev_map_tensor = torch.log1p(log_scale_gain * bev_map_tensor.clamp_min(0.0))
    if smoothing_kernel and smoothing_kernel > 1:
        padding = smoothing_kernel // 2
        bev_map_tensor = F.avg_pool2d(
            bev_map_tensor.unsqueeze(0).unsqueeze(0),
            kernel_size=smoothing_kernel,
            stride=1,
            padding=padding
        ).squeeze(0).squeeze(0)
    return bev_map_tensor


def split_active_inactive(bev_map_tensor, percentile):
    """Return boolean mask for active regions above the given percentile."""
    if percentile is None or percentile <= 0.0 or percentile >= 100.0:
        return None, None
    flat = bev_map_tensor.reshape(-1)
    if flat.numel() == 0:
        return None, None
    threshold = torch.quantile(flat, percentile / 100.0)
    if torch.isnan(threshold):
        return None, None
    active_mask = bev_map_tensor >= threshold
    return active_mask, float(threshold)


def build_prediction_heatmap(pred_dicts):
    """Aggregate multi-head CenterHead predictions into a single heatmap."""
    if not isinstance(pred_dicts, (list, tuple)) or len(pred_dicts) == 0:
        return None

    per_head_maps = []
    for pred in pred_dicts:
        if not isinstance(pred, dict) or 'hm' not in pred:
            continue
        hm = pred['hm']  # (B, C, H, W)
        hm_sig = torch.sigmoid(hm)
        # Take the max score across classes for this head
        hm_max = hm_sig.max(dim=1, keepdim=True).values  # (B, 1, H, W)
        per_head_maps.append(hm_max)

    if not per_head_maps:
        return None

    stacked = torch.stack(per_head_maps, dim=0)  # (num_heads, B, 1, H, W)
    aggregated = stacked.max(dim=0).values       # (B, 1, H, W)
    return aggregated


def clone_batch_cpu(batch_dict):
    cloned = {}
    for key, val in batch_dict.items():
        if isinstance(val, torch.Tensor):
            cloned[key] = val.clone()
        elif isinstance(val, np.ndarray):
            cloned[key] = val.copy()
        else:
            cloned[key] = copy.deepcopy(val)
    return cloned


def clone_batch_to_device(batch_dict, device):
    cloned = {}
    for key, val in batch_dict.items():
        if isinstance(val, torch.Tensor):
            cloned[key] = val.clone().to(device)
        elif isinstance(val, np.ndarray):
            if np.issubdtype(val.dtype, np.number):
                tensor = torch.from_numpy(val.copy())
                cloned[key] = tensor.float().to(device)
            else:
                cloned[key] = val.copy()
        else:
            cloned[key] = copy.deepcopy(val)
    return cloned


def plot_bev_map(
    bev_map_vis,
    boxes,
    output_path,
    title,
    H,
    W,
    *,
    use_world_coords=False,
    show_colorbar=False,
    show_box_labels=False,
    normalization='percentile',
    p_low=0.0,
    p_high=1.0,
    cmap='viridis',
    interpolation='bilinear',
    x_min=-54.0,
    x_max=54.0,
    y_min=-54.0,
    y_max=54.0,
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    extent = [x_min, x_max, y_min, y_max] if use_world_coords else [0, W, 0, H]
    im = ax.imshow(
        bev_map_vis,
        cmap=cmap,
        origin='lower',
        extent=extent,
        vmin=0.0,
        vmax=1.0,
        interpolation=interpolation
    )
    if show_colorbar:
        if normalization == 'percentile':
            norm_desc = 'percentile'
        elif normalization == 'minmax' or normalization == 'min-max':
            norm_desc = 'min-max'
        elif normalization == 'signed':
            norm_desc = 'signed'
        else:
            norm_desc = normalization
        plt.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label=f'Normalized activation ({norm_desc}: {p_low:.2f} → {p_high:.2f})'
        )

    for box in boxes:
        corners_world = bev_corners_xy(box)
        if use_world_coords:
            corners_plot = corners_world
        else:
            corners_plot = world_to_pixels(
                corners_world,
                (x_min, x_max),
                (y_min, y_max),
                W,
                H
            )
        poly = Polygon(
            corners_plot,
            closed=True,
            edgecolor='red',
            facecolor='none',
            linewidth=1.5,
            alpha=0.9,
            zorder=3
        )
        ax.add_patch(poly)
        label = format_class_label(box[7]) if len(box) > 7 else None
        if show_box_labels and label:
            cx, cy = corners_plot.mean(axis=0)
            ax.text(
                cx,
                cy,
                label,
                color='red',
                fontsize=7,
                ha='center',
                va='center',
                zorder=4,
                bbox=dict(facecolor='black', alpha=0.45, lw=0, pad=0.2)
            )

    if use_world_coords:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    else:
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def visualize_raw_channel_avg_features(
    batch_dict,
    sample_idx,
    feature_key,
    output_dir='visualize/results',
    aggregation_mode='mean',
    center_channels=False,
    apply_relu=False,
    normalization='percentile',
    low_percentile=1.0,
    high_percentile=99.5,
    use_world_coords=False,
    show_colorbar=False,
    show_box_labels=False,
    cmap='viridis',
    interpolation='bilinear',
    post_sigmoid=False,
    log_scale_gain=0.0,
    smoothing_kernel=1,
    active_percentile=None,
):
    """BEV 공간 피처를 'Channel Average'하여 원본(Raw) 그대로 시각화합니다."""
    os.makedirs(output_dir, exist_ok=True)
    
    # ── 1) Get BEV Spatial Feature Map ──
    if feature_key not in batch_dict:
        print(f"Error: Feature key '{feature_key}' not found.")
        print(f"DEBUG: Available keys: {batch_dict.keys()}")
        return None
        
    # (B, C, H, W) 텐서 로드
    bev_features_tensor = batch_dict[feature_key]
    
    # (B, C, H, W) -> (C, H, W) (첫 번째 배치 아이템 선택)
    bev_features = bev_features_tensor[0] 
    
    # =================================================================================
    # ▼▼▼ [핵심] "channel avg" (채널 평균)을 사용하여 맵 생성 ▼▼▼
    # =================================================================================
    # (C, H, W) -> (H, W) by aggregating across channels
    bev_map_tensor = aggregate_bev_channels(
        bev_features,
        mode=aggregation_mode,
        center_channels=center_channels,
        apply_relu=apply_relu
    )
    bev_map_tensor = postprocess_bev_map(
        bev_map_tensor,
        post_sigmoid=post_sigmoid,
        log_scale_gain=log_scale_gain,
        smoothing_kernel=smoothing_kernel
    )
    bev_map = to_np(bev_map_tensor)
    # Normalize only for visualization; keep raw statistics for reference.
    bev_map_vis, (p_low, p_high) = normalize_bev_map(
        bev_map,
        method=normalization,
        low_percentile=low_percentile,
        high_percentile=high_percentile
    )
    # =================================================================================
    
    feature_source = f"BEV Features ({feature_key} - Raw Channel Avg)"
    
    # =================================================================================
    # ▼▼▼ [핵심 수정] ReLU 및 정규화 로직 제거 ▼▼▼
    # =================================================================================
    # bev_map[bev_map < 0] = 0  <-- (제거)
    # bev_map = (bev_map - bev_map.min()) / ... <-- (제거)
    # =================================================================================
    
    H, W = bev_map.shape
    print(
        "    BEV tensor shape:",
        tuple(bev_features_tensor.shape),
        "-> map",
        (H, W),
    f"| post_sigmoid={post_sigmoid}, log_gain={log_scale_gain}, smooth={smoothing_kernel}"
    f" | normalization={normalization} ({p_low:.3f}, {p_high:.3f})"
    )
    
    # ────────────────────────────────────────────────────────

    # ── 2) Load GT boxes ──
    b = 0
    gt = to_np(batch_dict["gt_boxes"])
    boxes = gt[b]
    valid = ~(np.all(np.isclose(boxes, 0.0), axis=1))
    boxes = boxes[valid]
    
    # ── 4) Coordinate range ──
    x_min, x_max = -54.0, 54.0
    y_min, y_max = -54.0, 54.0

    base_filename = f'bev_features_{feature_key}_{sample_idx:03d}'
    output_path = os.path.join(output_dir, f'{base_filename}.png')
    title = f'{feature_source} | sample {sample_idx}'
    plot_bev_map(
        bev_map_vis,
        boxes,
        output_path,
        title,
        H,
        W,
        use_world_coords=use_world_coords,
        show_colorbar=show_colorbar,
        show_box_labels=show_box_labels,
        normalization=normalization,
        p_low=p_low,
        p_high=p_high,
        cmap=cmap,
        interpolation=interpolation,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    print(f"[OK] Saved feature map: {output_path}")

    # Visualize active/inactive regions if requested.
    if active_percentile is not None:
        active_mask, threshold = split_active_inactive(bev_map_tensor, active_percentile)
        if active_mask is not None and threshold is not None:
            print(
                f"    Active-region threshold (top {active_percentile:.2f}%): {threshold:.4f}"
            )
            inactive_mask = ~active_mask

            region_specs = [
                (bev_map_tensor * active_mask.float(), 'active', f'Active (top {active_percentile:.1f}%)'),
                (bev_map_tensor * inactive_mask.float(), 'inactive', 'Inactive'),
            ]

            for region_tensor, suffix, label in region_specs:
                if torch.count_nonzero(region_tensor).item() == 0:
                    continue
                region_np = to_np(region_tensor)
                region_vis, (r_low, r_high) = normalize_bev_map(
                    region_np,
                    method=normalization,
                    low_percentile=low_percentile,
                    high_percentile=high_percentile
                )
                region_path = os.path.join(output_dir, f'{base_filename}_{suffix}.png')
                region_title = f'{feature_source} | {label} | sample {sample_idx}'
                plot_bev_map(
                    region_vis,
                    boxes,
                    region_path,
                    region_title,
                    H,
                    W,
                    use_world_coords=use_world_coords,
                    show_colorbar=show_colorbar,
                    show_box_labels=show_box_labels,
                    normalization=normalization,
                    p_low=r_low,
                    p_high=r_high,
                    cmap=cmap,
                    interpolation=interpolation,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                )
                print(f"    [OK] Saved {label.lower()} map: {region_path}")

    return output_path

# =================================================================================

def main():
    """Main function to run visualization"""
    print("Setting up environment...")
    dataloader, logger, cfg = setup_environment()
    
    print(f"Dataset contains {len(dataloader)} samples")
    print("Starting visualization...")
    
    # Visualize first few samples
    num_samples_to_visualize = 5

    ckpt_paths = {
        'distill': '/home/byounggun/RadarDistill/output2/radar_distill/radar_distill_train/default/ckpt/checkpoint_epoch_40.pth',
        'baseline': '/home/byounggun/RadarDistill/output2/radar_distill/radar_distill_train/default/ckpt/checkpoint_epoch_11.pth',
    }

    dataset = dataloader.dataset
    models = {}
    for tag, ckpt_path in ckpt_paths.items():
        print(f"Loading model '{tag}' from {ckpt_path}")
        model_cfg_copy = copy.deepcopy(cfg.MODEL)
        model = build_network(model_cfg=model_cfg_copy, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
        model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=False)
        model.cuda()
        model.train()
        models[tag] = model
    
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(dataloader):
            if batch_idx >= num_samples_to_visualize:
                break
                
            print(f"\nProcessing sample {batch_idx + 1}/{num_samples_to_visualize}")
            cpu_batch = clone_batch_cpu(batch_dict)

            heatmap_results = {}
            for tag, model in models.items():
                local_batch = clone_batch_to_device(cpu_batch, device='cuda')
                model.forward(local_batch)
                lidar_pred_heatmap = None
                if 'lidar_pred_dicts' in local_batch:
                    lidar_pred_heatmap = build_prediction_heatmap(local_batch['lidar_pred_dicts'])
                if lidar_pred_heatmap is None:
                    print(f"  [WARN] Missing lidar prediction heatmap for model '{tag}'. Skipping sample.")
                    heatmap_results = {}
                    break
                heatmap_results[tag] = lidar_pred_heatmap.detach().cpu()

            if len(heatmap_results) != len(models):
                print("  [WARN] Could not compute heatmaps for all models; skipping visualization for this sample.")
                continue

            distill_heatmap = heatmap_results['distill']
            baseline_heatmap = heatmap_results['baseline']

            if distill_heatmap.shape != baseline_heatmap.shape:
                print("  [WARN] Heatmap shape mismatch between models; skipping sample.")
                continue

            cpu_batch['lidar_prediction_heatmap_distill'] = distill_heatmap
            cpu_batch['lidar_prediction_heatmap_baseline'] = baseline_heatmap
            cpu_batch['lidar_prediction_heatmap_diff'] = (distill_heatmap - baseline_heatmap)

            output_dir = 'visualize/results'

            features_to_plot = [
                'lidar_prediction_heatmap_distill',
                'lidar_prediction_heatmap_baseline',
                'lidar_prediction_heatmap_diff',
            ]

            for feat_key in features_to_plot:
                if feat_key not in cpu_batch:
                    continue
                print(f"--- Visualizing feature: {feat_key} ---")
                vis_cfg = {
                    **DEFAULT_VIS_CFG,
                    **FEATURE_VIS_OVERRIDES.get(feat_key, {}),
                }
                visualize_raw_channel_avg_features(
                    cpu_batch,
                    batch_idx,
                    feature_key=feat_key,
                    output_dir=output_dir,
                    **vis_cfg,
                )

            # Print some info about the sample
            if 'gt_boxes' in cpu_batch:
                gt_boxes = to_np(cpu_batch['gt_boxes'])[0]
                valid_gt = ~(np.all(np.isclose(gt_boxes, 0.0), axis=1))
                num_gt = valid_gt.sum()
                print(f"  - Ground truth boxes: {num_gt}")
            
            if 'points' in cpu_batch:
                pts = to_np(cpu_batch['points'])
                num_pts = len(pts[pts[:, 0] == 0]) if pts.shape[1] >= 1 else len(pts)
                print(f"  - Radar points: {num_pts}")
    
    print(f"\nVisualization completed! Check the 'visualize/results' directory for output images.")

if __name__ == '__main__':
    main()