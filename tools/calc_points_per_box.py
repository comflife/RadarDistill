import argparse
import numpy as np
import torch
from tqdm import tqdm
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

def calculate_points_in_boxes_gpu(dataset, class_names):
    print(f"Total samples to process: {len(dataset)}")
    
    stats = {} 
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, pin_memory=True, num_workers=4, shuffle=False,
        collate_fn=dataset.collate_batch
    )

    for i, batch_dict in tqdm(enumerate(dataloader), total=len(dataloader)):
        # 1. GT Boxes 처리 (NumPy)
        gt_boxes = batch_dict.get('gt_boxes', None) # (B, M, 8)
        
        if gt_boxes is None:
            continue
            
        # Batch size=1 가정이므로 [0] 인덱싱
        # gt_boxes는 collate_batch에서 numpy array로 반환됨
        valid_mask = (gt_boxes[0, :, 3] > 0) 
        batch_gt_boxes = gt_boxes[0][valid_mask] # (M_valid, 8) numpy array

        if batch_gt_boxes.shape[0] == 0:
            continue

        # 2. 포인트 처리
        point_keys = [k for k in batch_dict.keys() if 'points' in k and batch_dict[k].ndim == 2]
        
        for key in point_keys:
            sensor_name = "LiDAR" if key == 'points' else "Radar"
            if sensor_name not in stats:
                stats[sensor_name] = {name: {'total_points': 0, 'total_boxes': 0, 'points_list': []} for name in class_names}
            
            points = batch_dict[key] # numpy array (N, 4+) [batch_idx, x, y, z, ...]
            
            # 배치 인덱스 마스크 (Batch=1)
            batch_mask = (points[:, 0] == 0)
            
            # === 수정된 부분: Numpy -> Tensor 변환 ===
            # NumPy 배열 슬라이싱
            points_xyz_np = points[batch_mask, 1:4] # (N, 3)
            
            if points_xyz_np.shape[0] == 0:
                # 포인트가 없는 경우 처리
                for box_idx in range(batch_gt_boxes.shape[0]):
                     label = int(batch_gt_boxes[box_idx, 7])
                     if 0 < label <= len(class_names):
                        stats[sensor_name][class_names[label-1]]['total_boxes'] += 1
                        stats[sensor_name][class_names[label-1]]['points_list'].append(0)
                continue

            # Tensor 변환 및 GPU 이동 (contiguous는 cuda tensor 생성 시 자동 처리되거나 필요 시 호출)
            points_tensor = torch.from_numpy(points_xyz_np).float().cuda().contiguous() # (N, 3)
            boxes_tensor = torch.from_numpy(batch_gt_boxes[:, :7]).float().cuda().contiguous() # (M, 7)

            # === GPU 연산 (Batch 차원 추가: 1, N, 3) ===
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_tensor.unsqueeze(0),
                boxes_tensor.unsqueeze(0)
            ).squeeze(0) # (N,)

            # === 결과 집계 ===
            valid_pts_mask = (box_idxs_of_pts >= 0)
            valid_box_indices = box_idxs_of_pts[valid_pts_mask]
            
            if len(valid_box_indices) > 0:
                points_counts = torch.bincount(valid_box_indices.long(), minlength=boxes_tensor.shape[0]).cpu().numpy()
            else:
                points_counts = np.zeros(boxes_tensor.shape[0], dtype=int)

            for box_idx in range(batch_gt_boxes.shape[0]):
                num_points = int(points_counts[box_idx])
                class_label_idx = int(batch_gt_boxes[box_idx, 7])
                
                if 0 < class_label_idx <= len(class_names):
                    class_name = class_names[class_label_idx - 1]
                    stats[sensor_name][class_name]['total_points'] += num_points
                    stats[sensor_name][class_name]['total_boxes'] += 1
                    stats[sensor_name][class_name]['points_list'].append(num_points)

    # 결과 출력
    for sensor_name, sensor_stats in stats.items():
        print(f"\n=== [{sensor_name}] Average Points per Box (GPU Calculated) ===")
        print(f"{'Class':<20} | {'Avg':<8} | {'Med':<6} | {'Min':<6} | {'Max':<6} | {'Total Boxes':<12}")
        print("-" * 80)
        
        for name in class_names:
            info = sensor_stats[name]
            if info['total_boxes'] > 0:
                avg_points = info['total_points'] / info['total_boxes']
                points_arr = np.array(info['points_list'])
                median_points = np.median(points_arr)
                min_points = np.min(points_arr)
                max_points = np.max(points_arr)
                
                print(f"{name:<20} | {avg_points:<8.1f} | {median_points:<6.1f} | {min_points:<6} | {max_points:<6} | {info['total_boxes']:<12}")
            else:
                print(f"{name:<20} | {'N/A':<8} | {'N/A':<6} | {'N/A':<6} | {'N/A':<6} | {0:<12}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/radar_distill/radar_distill_train.yaml', help='path to config file')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()

    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=4,
        logger=logger,
        training=False 
    )

    calculate_points_in_boxes_gpu(dataset, cfg.CLASS_NAMES)

if __name__ == '__main__':
    main()