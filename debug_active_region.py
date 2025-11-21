"""
Active Region 디버깅 스크립트
radar distribution 적용 전후의 active region을 시각화
원본 모델 코드는 수정하지 않고 monkey patch를 사용하여 중간 결과를 캡처
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# OpenPCDet 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__)))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# 중간 결과를 저장할 전역 변수
captured_data = {}


def visualize_active_regions(lidar_bev, radar_bev, batch_dict, radar_distribution, 
                            lidar_mask_before, guided_lidar_mask_after, 
                            sample_idx, save_dir='/home/byounggun/RadarDistill/debug_active_regions'):
    """
    Active region 적용 전후를 시각화하여 저장
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    B, _, H, W = radar_bev.shape
    device = lidar_bev.device
    dtype = lidar_bev.dtype

    # 기본 마스크 계산
    radar_mask = radar_bev.sum(1, keepdim=True)
    
    # Active region 계산 (원본 코드와 동일)
    gt_box_mask = (radar_distribution > 0).float()
    activate_map = (radar_mask > 0).float() + guided_lidar_mask_after * 0.5

    mask_radar_lidar = torch.zeros_like(activate_map, dtype=torch.float)
    mask_radar_de_lidar = torch.zeros_like(activate_map, dtype=torch.float)
    mask_radar_lidar[activate_map == 1.5] = 1
    mask_radar_de_lidar[activate_map == 1.0] = 1

    if mask_radar_de_lidar.sum() > 0:
        mask_radar_de_lidar *= (mask_radar_lidar.sum() / (mask_radar_de_lidar.sum() + 1e-6))

    # Radar points 추출 (BEV 좌표계로 변환)
    radar_points_bev = None
    if 'radar_points' in batch_dict:
        radar_pts = batch_dict['radar_points']
        if isinstance(radar_pts, torch.Tensor):
            radar_pts = radar_pts.cpu().numpy()
        
        # Point cloud range (from model config, typical for NuScenes)
        pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]  # [x_min, y_min, z_min, x_max, y_max, z_max]
        x_min, y_min = pc_range[0], pc_range[1]
        x_max, y_max = pc_range[3], pc_range[4]
        
        # Filter points for current batch (batch index 0)
        batch_mask = radar_pts[:, 0] == 0
        pts = radar_pts[batch_mask]
        
        if len(pts) > 0:
            # Convert world coordinates to pixel coordinates
            x_coords = pts[:, 1]  # x in world frame
            y_coords = pts[:, 2]  # y in world frame
            
            # Map to pixel coordinates
            u = (x_coords - x_min) / (x_max - x_min) * W
            v = (y_coords - y_min) / (y_max - y_min) * H
            
            # Filter points within image bounds
            valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            radar_points_bev = np.stack([u[valid], v[valid]], axis=1)

    # 각 배치에 대해 시각화
    for b in range(B):
        print(f"\n{'='*50}")
        print(f"Visualizing Sample {sample_idx}, Batch {b}")
        print(f"{'='*50}")
        
        # === 메인 비교 그림 (8 패널) ===
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        
        # Helper function to add radar points to each subplot
        def add_radar_points(ax, alpha=0.3, color='cyan', size=1):
            if radar_points_bev is not None and len(radar_points_bev) > 0:
                ax.scatter(radar_points_bev[:, 0], radar_points_bev[:, 1], 
                          c=color, s=size, alpha=alpha, marker='.', zorder=10)
        
        # Row 1: 입력 및 중간 결과
        # 1. Original Lidar Mask (BEFORE)
        im1 = axes[0, 0].imshow(lidar_mask_before[b, 0].cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
        axes[0, 0].set_title('1. Lidar Mask\n(BEFORE - Original)', fontsize=13, fontweight='bold', color='blue')
        axes[0, 0].axis('off')
        add_radar_points(axes[0, 0], alpha=0.4, color='red', size=2)
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # 2. Radar Distribution (Multiplier)
        im2 = axes[0, 1].imshow(radar_distribution[b, 0].cpu().numpy(), cmap='hot')
        axes[0, 1].set_title('2. Radar Distribution\n(Object-aligned, Multiplier)', 
                            fontsize=13, fontweight='bold')
        axes[0, 1].axis('off')
        add_radar_points(axes[0, 1], alpha=0.5, color='cyan', size=2)
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # 3. GT Box Mask
        im3 = axes[0, 2].imshow(gt_box_mask[b, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[0, 2].set_title('3. GT Box Mask\n(radar_distribution > 0)', 
                            fontsize=13, fontweight='bold')
        axes[0, 2].axis('off')
        add_radar_points(axes[0, 2], alpha=0.4, color='red', size=2)
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # 4. Guided Lidar Mask (AFTER)
        im4 = axes[0, 3].imshow(guided_lidar_mask_after[b, 0].cpu().numpy(), cmap='viridis')
        axes[0, 3].set_title('4. Guided Lidar Mask\n(AFTER - Radar Applied)', 
                            fontsize=13, fontweight='bold', color='red')
        axes[0, 3].axis('off')
        add_radar_points(axes[0, 3], alpha=0.4, color='red', size=2)
        plt.colorbar(im4, ax=axes[0, 3], fraction=0.046, pad=0.04)
        
        # Row 2: 추가 분석
        # 5. Radar Mask
        im5 = axes[1, 0].imshow(radar_mask[b, 0].cpu().numpy(), cmap='plasma')
        axes[1, 0].set_title('5. Radar Mask\n(radar_bev sum)', fontsize=13, fontweight='bold')
        axes[1, 0].axis('off')
        add_radar_points(axes[1, 0], alpha=0.6, color='cyan', size=2)
        plt.colorbar(im5, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # 6. Activate Map
        im6 = axes[1, 1].imshow(activate_map[b, 0].cpu().numpy(), cmap='jet')
        axes[1, 1].set_title('6. Activate Map\n(radar + guided_lidar*0.5)', 
                            fontsize=13, fontweight='bold')
        axes[1, 1].axis('off')
        add_radar_points(axes[1, 1], alpha=0.4, color='white', size=2)
        plt.colorbar(im6, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # 7. Mask Radar-Lidar (overlap, value=1.5)
        im7 = axes[1, 2].imshow(mask_radar_lidar[b, 0].cpu().numpy(), cmap='Greens', vmin=0, vmax=1)
        axes[1, 2].set_title('7. Radar-Lidar Overlap\n(activate_map == 1.5)', 
                            fontsize=13, fontweight='bold')
        axes[1, 2].axis('off')
        add_radar_points(axes[1, 2], alpha=0.5, color='red', size=2)
        plt.colorbar(im7, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        # 8. Mask Radar-Delidar (radar only, value=1.0)
        im8 = axes[1, 3].imshow(mask_radar_de_lidar[b, 0].cpu().numpy(), cmap='Reds')
        axes[1, 3].set_title('8. Radar Only (weighted)\n(activate_map == 1.0)', 
                            fontsize=13, fontweight='bold')
        axes[1, 3].axis('off')
        add_radar_points(axes[1, 3], alpha=0.5, color='yellow', size=2)
        plt.colorbar(im8, ax=axes[1, 3], fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Active Region Visualization - Sample {sample_idx}, Batch {b}\n'
                     f'BEFORE vs AFTER Radar Distribution Application (Cyan/Red dots = Radar Points)', 
                     fontsize=17, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        save_path = save_dir / f'sample_{sample_idx}_batch_{b}_full_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✓ Saved: {save_path}')
        plt.close()
        
        # === BEFORE/AFTER 직접 비교 (3 패널) ===
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        
        # Before (Original Lidar Mask)
        im1 = axes[0].imshow(lidar_mask_before[b, 0].cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
        axes[0].set_title('BEFORE\nOriginal Lidar Mask', fontsize=15, fontweight='bold', color='blue')
        axes[0].axis('off')
        if radar_points_bev is not None and len(radar_points_bev) > 0:
            axes[0].scatter(radar_points_bev[:, 0], radar_points_bev[:, 1], 
                          c='red', s=3, alpha=0.5, marker='.', label='Radar Points', zorder=10)
            axes[0].legend(loc='upper right', fontsize=10)
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Radar Distribution (multiplier)
        im2 = axes[1].imshow(radar_distribution[b, 0].cpu().numpy(), cmap='hot')
        axes[1].set_title('APPLIED\nRadar Distribution\n(Multiplier)', 
                         fontsize=15, fontweight='bold', color='orange')
        axes[1].axis('off')
        if radar_points_bev is not None and len(radar_points_bev) > 0:
            axes[1].scatter(radar_points_bev[:, 0], radar_points_bev[:, 1], 
                          c='cyan', s=3, alpha=0.6, marker='.', label='Radar Points', zorder=10)
            axes[1].legend(loc='upper right', fontsize=10)
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # After (Guided Lidar Mask)
        im3 = axes[2].imshow(guided_lidar_mask_after[b, 0].cpu().numpy(), cmap='viridis')
        axes[2].set_title('AFTER\nGuided Lidar Mask\n(Radar-weighted)', 
                         fontsize=15, fontweight='bold', color='red')
        axes[2].axis('off')
        if radar_points_bev is not None and len(radar_points_bev) > 0:
            axes[2].scatter(radar_points_bev[:, 0], radar_points_bev[:, 1], 
                          c='red', s=3, alpha=0.5, marker='.', label='Radar Points', zorder=10)
            axes[2].legend(loc='upper right', fontsize=10)
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.suptitle(f'BEFORE → AFTER Comparison - Sample {sample_idx}, Batch {b}\n'
                     f'Formula: guided = lidar * (1 - gt_box_mask) + lidar * radar_distribution', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = save_dir / f'sample_{sample_idx}_batch_{b}_before_after.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✓ Saved: {save_path}')
        plt.close()
        
        # === 차이 맵 (Difference Map) ===
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        diff_map = guided_lidar_mask_after[b, 0] - lidar_mask_before[b, 0]
        
        im1 = axes[0].imshow(diff_map.cpu().numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0].set_title('Difference Map\n(AFTER - BEFORE)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='Change')
        
        # 차이가 있는 영역만 표시
        diff_mask = (torch.abs(diff_map) > 1e-5).cpu().numpy()
        im2 = axes[1].imshow(diff_mask, cmap='binary')
        axes[1].set_title('Changed Region Mask\n(|diff| > 1e-5)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.suptitle(f'Change Analysis - Sample {sample_idx}, Batch {b}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = save_dir / f'sample_{sample_idx}_batch_{b}_difference.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✓ Saved: {save_path}')
        plt.close()
        
        # === 통계 정보 출력 ===
        print(f"\n📊 Statistics:")
        print(f"  Lidar Mask (BEFORE):       non-zero = {(lidar_mask_before[b] > 0).sum().item():.0f} pixels, "
              f"sum = {lidar_mask_before[b].sum().item():.2f}")
        print(f"  Radar Distribution:        non-zero = {(radar_distribution[b] > 0).sum().item():.0f} pixels, "
              f"max = {radar_distribution[b].max().item():.4f}, mean = {radar_distribution[b].mean().item():.4f}")
        print(f"  Guided Lidar (AFTER):      non-zero = {(guided_lidar_mask_after[b] > 0).sum().item():.0f} pixels, "
              f"sum = {guided_lidar_mask_after[b].sum().item():.2f}")
        print(f"  Difference (AFTER-BEFORE): {(guided_lidar_mask_after[b].sum() - lidar_mask_before[b].sum()).item():.2f}")
        print(f"  Changed pixels:            {diff_mask.sum():.0f}")
        print()


def wrap_low_loss_method(original_method):
    """
    low_loss 메서드를 래핑하여 중간 계산 결과를 캡처
    """
    def wrapped_low_loss(self, lidar_bev, radar_bev, batch_dict):
        B, _, H, W = radar_bev.shape
        device = lidar_bev.device
        dtype = lidar_bev.dtype

        # BEFORE: Original lidar mask 계산
        lidar_mask_before = (lidar_bev.sum(1).unsqueeze(1) > 0).float()
        
        # Radar distribution 생성 (원본 코드)
        density_maps = self._build_gt_radar_distribution_object_aligned(
            batch_dict, device, dtype
        )
        
        # Object-aligned grids → Global BEV grid로 projection
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
                
                half_grid_world = (grid_size * pixel_size) / 2.0
                local_x = torch.linspace(-half_grid_world, half_grid_world, grid_size, device=device, dtype=dtype)
                local_y = torch.linspace(-half_grid_world, half_grid_world, grid_size, device=device, dtype=dtype)
                yy, xx = torch.meshgrid(local_y, local_x, indexing='ij')
                local_coords = torch.stack([xx, yy], dim=-1)
                
                cos_a, sin_a = torch.cos(angle), torch.sin(angle)
                rot_mat = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=device, dtype=dtype)
                
                local_flat = local_coords.reshape(-1, 2)
                world_flat = torch.matmul(local_flat, rot_mat.T) + center.unsqueeze(0)
                
                bev_x_idx = ((world_flat[:, 0] - x_min) / x_range * W).long().clamp(0, W - 1)
                bev_y_idx = ((world_flat[:, 1] - y_min) / y_range * H).long().clamp(0, H - 1)
                
                density_values = obj_grid.reshape(-1)
                flat_indices = bev_y_idx * W + bev_x_idx
                radar_distribution[b, 0].view(-1).scatter_add_(0, flat_indices, density_values)
        
        for b in range(B):
            max_val = radar_distribution[b].max()
            if max_val > 0:
                radar_distribution[b] = radar_distribution[b] / max_val

        # AFTER: Guided lidar mask 계산
        gt_box_mask = (radar_distribution > 0).float()
        guided_lidar_mask_after = lidar_mask_before * (1 - gt_box_mask) + lidar_mask_before * radar_distribution

        # 중간 결과 저장
        captured_data['lidar_bev'] = lidar_bev.detach()
        captured_data['radar_bev'] = radar_bev.detach()
        captured_data['batch_dict'] = batch_dict
        captured_data['radar_distribution'] = radar_distribution.detach()
        captured_data['lidar_mask_before'] = lidar_mask_before.detach()
        captured_data['guided_lidar_mask_after'] = guided_lidar_mask_after.detach()
        
        # 원본 메서드 호출하여 loss 계산
        return original_method(lidar_bev, radar_bev, batch_dict)
    
    return wrapped_low_loss


def main():
    # 설정
    # Change to tools directory so relative paths in config work
    tools_dir = Path('/home/byounggun/RadarDistill/tools')
    os.chdir(str(tools_dir))
    
    cfg_file = 'cfgs/radar_distill/radar_distill_train.yaml'
    ckpt_path = '/home/byounggun/RadarDistill/output_best0/radar_distill/radar_distill_train/default/ckpt/checkpoint_epoch_40.pth'
    save_dir = '/home/byounggun/RadarDistill/debug_active_regions'
    num_samples = 5
    
    print("="*70)
    print("Active Region Debugging Script")
    print("="*70)
    print(f"Config: {cfg_file}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Output directory: {save_dir}")
    print(f"Number of samples: {num_samples}")
    print("="*70)
    
    # Config 로드
    cfg_from_yaml_file(cfg_file, cfg)
    
    # Override INFO_PATH with available data files (from visualize_sample.py)
    cfg.DATA_CONFIG.INFO_PATH = {
        'train': ['/home/byounggun/RadarDistill/data/nuscenes/v1.0-trainval/nuscenes_infos_6radar_10sweeps_train.pkl'],
        'test': ['/home/byounggun/RadarDistill/data/nuscenes/v1.0-trainval/nuscenes_infos_6radar_10sweeps_val.pkl'],
    }
    
    logger = common_utils.create_logger()
    
    # 데이터셋 로드
    print("\n📦 Loading dataset...")
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=4,
        logger=logger,
        training=False  # Use test split (validation data)
    )
    print(f"✓ Dataset loaded: {len(test_set)} samples")
    
    # 모델 로드
    print("\n🤖 Loading model...")
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=False)
    model.cuda()
    model.train()  # Use train mode to enable loss calculation
    print("✓ Model loaded and set to train mode")
    
    # Monkey patch: low_loss 메서드 래핑
    print("\n🔧 Finding module with 'low_loss' method...")
    target_module = None
    for i, module in enumerate(model.module_list):
        if hasattr(module, 'low_loss'):
            target_module = module
            print(f"✓ Found 'low_loss' in module_list[{i}]: {type(module).__name__}")
            break
    
    if target_module is None:
        print("❌ Error: Could not find any module with 'low_loss' method")
        print("\nAvailable modules:")
        for i, module in enumerate(model.module_list):
            print(f"  module_list[{i}]: {type(module).__name__}")
        return
    
    print("🔧 Applying monkey patch to capture intermediate results...")
    original_low_loss = target_module.low_loss
    target_module.low_loss = lambda *args, **kwargs: wrap_low_loss_method(original_low_loss)(target_module, *args, **kwargs)
    print("✓ Monkey patch applied")
    
    # 추론 및 시각화
    print(f"\n🚀 Starting inference on {num_samples} samples...\n")
    
    with torch.no_grad():
        for i, batch_dict in enumerate(test_loader):
            if i >= num_samples:
                break
            
            print(f"\n{'='*70}")
            print(f"Processing sample {i+1}/{num_samples}")
            print(f"{'='*70}")
            
            # GPU로 데이터 로드
            load_data_to_gpu(batch_dict)
            
            # Forward pass (중간 결과는 captured_data에 저장됨)
            try:
                captured_data.clear()
                model.forward(batch_dict)  # Just update batch_dict, don't need return values
                
                # 중간 결과가 캡처되었는지 확인
                if 'lidar_mask_before' in captured_data and 'guided_lidar_mask_after' in captured_data:
                    visualize_active_regions(
                        captured_data['lidar_bev'],
                        captured_data['radar_bev'],
                        captured_data['batch_dict'],
                        captured_data['radar_distribution'],
                        captured_data['lidar_mask_before'],
                        captured_data['guided_lidar_mask_after'],
                        sample_idx=i,
                        save_dir=save_dir
                    )
                else:
                    print("⚠️  Warning: Could not capture intermediate results")
                    
            except Exception as e:
                print(f"❌ Error during forward pass: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("✅ Visualization complete!")
    print(f"📁 Check the '{save_dir}' directory for results.")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
