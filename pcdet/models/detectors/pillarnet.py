import torch
import torch.nn as nn
import torch.nn.functional as F
from .detector3d_template import Detector3DTemplate


class PillarNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.model_cfg = model_cfg
        if self.model_cfg.get('FREEZE_PIPELINE', None) is not None:
            self.no_grad_module = model_cfg['FREEZE_PIPELINE']
            for i, cur_module in enumerate(self.module_list):
                cur_name = cur_module.__class__.__name__
                if cur_name in self.no_grad_module:
                    for param in cur_module.parameters():
                        param.requires_grad = False
        else:
            self.no_grad_module = []

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            cur_name = cur_module.__class__.__name__
            if cur_name in self.no_grad_module:
                cur_module.eval()
            batch_dict = cur_module(batch_dict)

        if self.training:
            if self.model_cfg.get('DISTILL', None) is None:
                loss, tb_dict, disp_dict = self.get_training_loss()
            elif self.model_cfg.get('DISTILL', None):
                loss, tb_dict, disp_dict = self.get_training_distll_loss(batch_dict)
            else:
                loss, tb_dict, disp_dict = self.get_training_wo_distll_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        
        return loss, tb_dict, disp_dict

    def get_training_distll_loss(self, batch_dict):
        disp_dict = {}
        
        loss_feature, tb_dict = self.radar_backbone_2d.get_loss(batch_dict)
        loss_rpn, _tb_dict = self.radar_dense_head.get_loss()
        tb_dict.update(_tb_dict)
        
        # TiGDistill-BEV Inter-channel and Inter-keypoint
        if 'loss_bev_combined' in batch_dict:
            loss_bev_ic = batch_dict['loss_bev_ic']
            loss_bev_ik = batch_dict['loss_bev_ik']
            loss_bev_combined = batch_dict['loss_bev_combined']
            loss_weights = self.model_cfg.RADAR_DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS
            bev_ic_weight = loss_weights.get('bev_ic_weight', 1.0)
            bev_ik_weight = loss_weights.get('bev_ik_weight', 1.0)
            
            # Use combined loss or weighted individual losses
            loss = loss_feature + loss_rpn + loss_bev_ic * bev_ic_weight + loss_bev_ik * bev_ik_weight
            
            tb_dict['loss_bev_ic'] = loss_bev_ic.item()
            tb_dict['loss_bev_ik'] = loss_bev_ik.item()
            tb_dict['loss_bev_combined'] = loss_bev_combined.item()
        else:
            loss = loss_feature + loss_rpn
        
        return loss, tb_dict, disp_dict

    def get_training_wo_distll_loss(self, batch_dict):
        disp_dict = {}
        loss_rpn, tb_dict = self.radar_dense_head.get_loss()
        loss = loss_rpn
        
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict