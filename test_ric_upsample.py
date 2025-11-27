
import torch
import torch.nn as nn
import sys
import os

# Add path to find pcdet
sys.path.append('/home/byounggun/RadarDistill')

from pcdet.models.backbones_2d.radar_distill_final import RICUpsample, Radar_Distill
from easydict import EasyDict

def test_ric_upsample():
    print("Testing RICUpsample...")
    model_cfg = EasyDict({
        'VOXEL_SIZE': [0.1, 0.1, 0.2],
        'POINT_CLOUD_RANGE': [0, -40, -3, 70.4, 40, 1]
    })
    
    # Initialize module
    try:
        model = RICUpsample(
            in_channels=64, 
            out_channels=64, 
            input_stride=1, 
            voxel_size=model_cfg.VOXEL_SIZE, 
            point_cloud_range=model_cfg.POINT_CLOUD_RANGE,
            model_cfg=model_cfg
        ).cuda()
    except Exception as e:
        print(f"Failed to init RICUpsample: {e}")
        return
    
    # Dummy input
    B, C, H, W = 2, 64, 100, 100
    x = torch.randn(B, C, H, W).cuda()
    
    # Dummy GT boxes: (B, N, 8) [x, y, z, dx, dy, dz, heading, label]
    gt_boxes = torch.tensor([
        [[10, 0, 0, 4, 2, 1, 0, 1], [20, 10, 0, 5, 2, 1, 1.5, 1]],
        [[30, -10, 0, 3, 2, 1, 0.5, 1], [0, 0, 0, 0, 0, 0, 0, 0]]
    ]).float().cuda()
    
    # Forward
    print("Running forward pass...")
    try:
        out, aux_loss = model(x, gt_boxes)
        print(f"Output shape: {out.shape}")
        print(f"Aux loss: {aux_loss}")
        print("RICUpsample Test Passed!")
    except Exception as e:
        print(f"RICUpsample Forward Failed: {e}")
        import traceback
        traceback.print_exc()

def test_radar_distill_vectorized():
    print("\nTesting Radar_Distill Vectorized Distribution...")
    model_cfg = EasyDict({
        'VOXEL_SIZE': [0.1, 0.1, 0.2],
        'POINT_CLOUD_RANGE': [0, -40, -3, 70.4, 40, 1]
    })
    
    # Instantiate with minimal config
    # We need to mock the parent class or ensure it can init
    # BaseBEVBackboneV2 might need more config
    # Let's try to just use the method if possible, but it's an instance method.
    # We can create a dummy class that inherits or just attach the method.
    
    class DummyDistill(Radar_Distill):
        def __init__(self, model_cfg):
            # Skip super init to avoid complex dependencies if possible
            # But we need self.voxel_size etc.
            self.model_cfg = model_cfg
            self.voxel_size = model_cfg.VOXEL_SIZE
            self.point_cloud_range = model_cfg.POINT_CLOUD_RANGE
            # We don't need encoders/decoders for this test
            
    try:
        model = DummyDistill(model_cfg)
        # We need to bind the method if we didn't inherit properly, but inheritance should work
        # if we override init.
        # Wait, get_radar_distribution_vectorized is in Radar_Distill.
        # If we override init, we still have the method.
    except Exception as e:
        print(f"Could not instantiate DummyDistill: {e}")
        return

    B, H, W = 2, 200, 200
    
    # Dummy batch dict
    batch_dict = {
        'batch_size': B,
        'gt_boxes': torch.tensor([
            [[10, 0, 0, 4, 2, 1, 0, 1]],
            [[20, 10, 0, 5, 2, 1, 1.5, 1]]
        ]).float().cuda(),
        'radar_points': torch.tensor([
            [0, 10.1, 0.1, 0],
            [0, 10.2, 0.2, 0],
            [1, 20.1, 10.1, 0]
        ]).float().cuda()
    }
    
    print("Running get_radar_distribution_vectorized...")
    try:
        dist = model.get_radar_distribution_vectorized(batch_dict, H, W)
        print(f"Distribution shape: {dist.shape}")
        print(f"Max value: {dist.max()}")
        print("Radar Distill Vectorized Test Passed!")
    except Exception as e:
        print(f"Vectorized Dist Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ric_upsample()
    test_radar_distill_vectorized()
