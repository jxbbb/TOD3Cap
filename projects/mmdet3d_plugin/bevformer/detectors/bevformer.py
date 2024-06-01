# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from copy import deepcopy

import llama.utils
from llama.llama_adapter import LLaMA_adapter
import util.misc as misc

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox

@DETECTORS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False
                 ):

        super(BEVFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }


        llama_ckpt_dir = "/data18/jinbu/nuscenes-caption/Attribute/LLaMA-Adapter//LLaMA-7B/7B"
        llama_tokenzier_path = "/data18/jinbu/nuscenes-caption/Attribute/LLaMA-Adapter//LLaMA-7B/tokenizer.model"
        self.llama_adapter = LLaMA_adapter(llama_ckpt_dir, llama_tokenzier_path) # TODO: fix phase


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats


    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_captions_3d,
                          gt_caplabels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev)

        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]

        losses, sampling_results = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)

        # in caption we only need the last layer output
        sampling_results = sampling_results[-1]

        # get caption loss
        # placehold
        caploss = self.caption_head(outs, pts_feats, sampling_results, gt_captions_3d, gt_caplabels_3d)

        losses.update({"loss_cap": caploss})

        return losses

    # def caption_head_with_gt(self, outs, pts_feats, sampling_results, gt_bboxes_3d, gt_captions_3d, gt_caplabels_3d):

    #     def _downsamplecap(array_size, sample_size):

    #         if sample_size <= array_size:
    #             # If the sample size is less than or equal to the array size, perform sampling without replacement
    #             sampled_indices = torch.randperm(array_size)[:sample_size]
    #         else:
    #             # If the sample size is greater than the array size, repeat certain positions' values until the desired length is reached
    #             num_repeats = sample_size // array_size
    #             remainder = sample_size % array_size

    #             # Repeat the entire array
    #             repeated_indices = torch.arange(array_size).repeat(num_repeats)

    #             # Randomly select indices for the remaining part
    #             remaining_indices = torch.randperm(array_size)[:remainder]

    #             # Concatenate the indices
    #             sampled_indices = torch.cat((repeated_indices, remaining_indices))

    #         return sampled_indices

    #     batch_size = len(gt_bboxes_3d)

    #     gt_bboxes = gt_bboxes_3d
    #     bev_embeds = pts_feats

    #     loss = []

    #     for bs in range(batch_size):
    #         gt_bbox = gt_bboxes[bs]
    #         gt_bbox = torch.cat((gt_bbox.gravity_center, gt_bbox.tensor[:, 3:]), dim=1)
    #         gt_bbox = normalize_bbox(gt_bbox, [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
    #         bev_embed = bev_embeds[bs]
    #         gt_captions = gt_captions_3d[bs]
    #         gt_caplabels = gt_caplabels_3d[bs]

    #         assert len(gt_bbox) == len(gt_captions)

    #         num_bboxes = gt_bbox.size(0)
    #         word_len = gt_captions.size(-1)

    #         sampled_indices = _downsamplecap(num_bboxes, 10)

    #         gt_bbox = gt_bbox[sampled_indices]
    #         bev_embed = bev_embed.unsqueeze(0).repeat(10, 1, 1, 1, 1, 1)

    #         caption_targets = gt_captions[sampled_indices]
    #         caplabel_targets = gt_caplabels[sampled_indices]
    #         caption_weights = caption_targets.new_ones((10,), dtype=torch.long)

    #         with torch.cuda.amp.autocast():
    #             caption_loss = self.llama_adapter((caption_targets, caplabel_targets, caption_weights), (bev_embed, gt_bbox[:, :8].to(bev_embed.device)))
                
    #             # # to test the generatation of gt box
    #             # self.eval()
    #             # format_instruction = "Describe the object in detail."
    #             # prompt = llama.format_prompt(format_instruction)
    #             # generated_caption = self.llama_adapter.generate((bev_embed[:1], gt_bbox[:1, :8].to(bev_embed.device)), ([prompt]))
    #             # print(generated_caption)
                
            
    #         loss.append(caption_loss)


    #     return torch.stack(loss, dim=0).mean()

    def caption_head(self, outs, pts_feats, sampling_results, gt_captions_3d, gt_caplabels_3d):

        def _downsamplecap(array_size, sample_size):

            if sample_size <= array_size:
                # If the sample size is less than or equal to the array size, perform sampling without replacement
                sampled_indices = torch.randperm(array_size)[:sample_size]
            else:
                # If the sample size is greater than the array size, repeat certain positions' values until the desired length is reached
                num_repeats = sample_size // array_size
                remainder = sample_size % array_size

                # Repeat the entire array
                repeated_indices = torch.arange(array_size).repeat(num_repeats)

                # Randomly select indices for the remaining part
                remaining_indices = torch.randperm(array_size)[:remainder]

                # Concatenate the indices
                sampled_indices = torch.cat((repeated_indices, remaining_indices))

            return sampled_indices

        batch_size = len(sampling_results)

        all_bbox_preds = outs['all_bbox_preds'][-1]
        bev_embeds = outs['bev_embed']

        loss = []

        for bs in range(batch_size):
            bbox_preds = all_bbox_preds[bs]
            bev_embed = bev_embeds[:, bs]
            gt_captions = gt_captions_3d[bs]
            gt_caplabels = gt_caplabels_3d[bs]

            num_bboxes = bbox_preds.size(0)
            word_len = gt_captions.size(-1)

            sampling_result = sampling_results[bs]
            pos_inds = sampling_result.pos_inds
            # neg_inds = sampling_result.neg_inds

            num_total_pos = pos_inds.numel()

            sampled_indices = _downsamplecap(num_total_pos, 2)

            bbox_preds = bbox_preds[pos_inds[sampled_indices]]
            bev_embed = bev_embed.unsqueeze(0).repeat(2, 1, 1)

            caption_targets = gt_captions[sampling_result.pos_assigned_gt_inds[sampled_indices]]
            caplabel_targets = gt_caplabels[sampling_result.pos_assigned_gt_inds[sampled_indices]]
            caption_weights = caption_targets.new_ones((2,), dtype=torch.long)

            with torch.cuda.amp.autocast():
                caption_loss = self.llama_adapter((caption_targets, caplabel_targets, caption_weights), (bev_embed, bbox_preds))

            loss.append(caption_loss)


        return torch.stack(loss, dim=0).mean()

    def generate_caption(self, outs, x):


        all_bbox_preds = outs['all_bbox_preds'][-1]
        all_cls_scores = outs['all_cls_scores'][-1]
        bev_embeds = outs['bev_embed']


        batch_size = len(all_bbox_preds)


        caption_outputs = []
        for bs in range(batch_size):
            bbox_preds = all_bbox_preds[bs]
            cls_scores = all_cls_scores[bs]

            bev_embed = bev_embeds[:, bs]
            bev_embed = bev_embed.unsqueeze(0)

            downsample_proposal = True
            if downsample_proposal:
                # TODO: make sure the num_bboxes is the same as bbox coder's
                # we only evaluate several objects because the huge memory cost of LLM
                num_bboxes = 20

                cls_scores = cls_scores.sigmoid()
                scores, indexs = cls_scores.view(-1).topk(num_bboxes)
                labels = indexs % 10
                bbox_index = indexs // 10
                bbox_preds = bbox_preds[bbox_index]

            else:
                num_bboxes = bbox_preds.size(0)

            
            ps_caption_output = []
            for object_id in range(num_bboxes):
                single_pred_bbox = bbox_preds[object_id].unsqueeze(0)

                with torch.cuda.amp.autocast():
                    format_instruction = "Describe the object in detail."
                    # prompt = llama.format_prompt(format_instruction)
                    # caption_output = self.llama_adapter.generate((bev_embed, single_pred_bbox), ([prompt]))
                
            
                ps_caption_output.append(format_instruction)

                print(caption_output)
                
                # break
            caption_outputs.append(ps_caption_output)
        print(f"*****************Iter Done: batch size: {batch_size} patch size:{num_bboxes} ******************")
        return caption_outputs


    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      gt_captions_3d=None,
                      gt_caplabels_3d=None,
                      gt_capmask_3d=None,
                      **kwargs
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        
        # do not freeze the bev encoder
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue-1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, 
                                            gt_bboxes_3d,
                                            gt_labels_3d, 
                                            gt_captions_3d,
                                            gt_caplabels_3d,
                                            img_metas,
                                            gt_bboxes_ignore, 
                                            prev_bev,)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)

        caption_results = self.generate_caption(outs, x)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels, caps=captions)
            for (bboxes, scores, labels), captions in zip(bbox_list, caption_results)
        ]

        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list
