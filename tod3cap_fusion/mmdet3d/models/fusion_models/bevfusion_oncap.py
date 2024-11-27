from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS

from copy import deepcopy

import llama.utils
from llama.llama_adapter import LLaMA_adapter
import util.misc as misc
# from scan2cap_utils.scan2cap_model import get_cap_model
# from scan2cap_utils.lib.loss_helper import compute_cap_loss

# from x_trans2cap_utils.x_trans2cap_model import get_x_trans_cap_decoder

from mmdet3d.core.bbox.util import normalize_bbox

import numpy as np

from .base import Base3DFusionModel

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if encoders.get("radar") is not None:
            if encoders["radar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["radar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["radar"]["voxelize"])
            self.encoders["radar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["radar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["radar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        # If the camera's vtransform is a BEVDepth version, then we're using depth loss. 
        self.use_depth_loss = ((encoders.get('camera', {}) or {}).get('vtransform', {}) or {}).get('type', '') in ['BEVDepth', 'AwareBEVDepth', 'DBEVDepth', 'AwareDBEVDepth']


        llama_ckpt_dir = "LLaMA-7B"
        llama_tokenzier_path = "LLaMA-7B/tokenizer.model"
        self.llama_adapter = LLaMA_adapter(llama_ckpt_dir, llama_tokenzier_path) # TODO: fix phase

        # self.scan_cap = get_cap_model()
        # self.x_trans_cap_decoder = get_x_trans_cap_decoder()
        
        # self.num_vocabs = 32000
        # self.emb_size = 300
        # self.lang_embedding = nn.Embedding(self.num_vocabs,self.emb_size)

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        radar_points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        gt_depths=None,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            radar_points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            depth_loss=self.use_depth_loss, 
            gt_depths=gt_depths,
        )
        return x
    
    def extract_features(self, x, sensor) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x, sensor)
        batch_size = coords[-1, 0] + 1
        x = self.encoders[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x
    
    # def extract_lidar_features(self, x) -> torch.Tensor:
    #     feats, coords, sizes = self.voxelize(x)
    #     batch_size = coords[-1, 0] + 1
    #     x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
    #     return x

    # def extract_radar_features(self, x) -> torch.Tensor:
    #     feats, coords, sizes = self.radar_voxelize(x)
    #     batch_size = coords[-1, 0] + 1
    #     x = self.encoders["radar"]["backbone"](feats, coords, batch_size, sizes=sizes)
    #     return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, sensor):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders[sensor]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    # @torch.no_grad()
    # @force_fp32()
    # def radar_voxelize(self, points):
    #     feats, coords, sizes = [], [], []
    #     for k, res in enumerate(points):
    #         ret = self.encoders["radar"]["voxelize"](res)
    #         if len(ret) == 3:
    #             # hard voxelize
    #             f, c, n = ret
    #         else:
    #             assert len(ret) == 2
    #             f, c = ret
    #             n = None
    #         feats.append(f)
    #         coords.append(F.pad(c, (1, 0), mode="constant", value=k))
    #         if n is not None:
    #             sizes.append(n)

    #     feats = torch.cat(feats, dim=0)
    #     coords = torch.cat(coords, dim=0)
    #     if len(sizes) > 0:
    #         sizes = torch.cat(sizes, dim=0)
    #         if self.voxelize_reduce:
    #             feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
    #                 -1, 1
    #             )
    #             feats = feats.contiguous()

    #     return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_captions_3d=None,
        gt_caplabels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                depths,
                radar,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                gt_captions_3d,
                gt_caplabels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_captions_3d=None,
        gt_caplabels_3d=None,
        **kwargs,
    ):
        features = []
        auxiliary_losses = {}
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    radar,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                    gt_depths=depths,
                )
                if self.use_depth_loss:
                    feature, auxiliary_losses['depth'] = feature[0], feature[-1]
            elif sensor == "lidar":
                feature = self.extract_features(points, sensor)
            elif sensor == "radar":
                feature = self.extract_features(radar, sensor)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        assert x.size(0) == 1
        bev_embed = x.clone()[0].permute(1,2,0).view(-1, 256)
        

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses, sampling_results = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                # elif type == "map":
                #     losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            if self.use_depth_loss:
                if 'depth' in auxiliary_losses:
                    outputs["loss/depth"] = auxiliary_losses['depth']
                else:
                    raise ValueError('Use depth loss is true, but depth loss not found')

            # get caption loss
            caploss = self.caption_head(pred_dict, bev_embed, sampling_results, gt_captions_3d, gt_caplabels_3d)

            outputs.update({"loss/object/caption": caploss})

            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    caption_results = self.generate_caption(bboxes, bev_embed)
                    for k, ((boxes, scores, labels), captions) in enumerate(zip(bboxes, caption_results)):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                                "captions_3d": captions,
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs

    def caption_head(self, outs, bev_embeds, sampling_results, gt_captions_3d, gt_caplabels_3d):

        batch_size = len(sampling_results)


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


        loss = []

        for bs in range(batch_size):
            bev_embed = bev_embeds
            gt_captions = gt_captions_3d[bs]
            gt_caplabels = gt_caplabels_3d[bs]

            sampling_result = sampling_results[bs]
            pos_inds = sampling_result.pos_inds
            # neg_inds = sampling_result.neg_inds

            num_total_pos = pos_inds.numel()

            sampled_indices = _downsamplecap(num_total_pos, 4)

            bbox_preds = sampling_result.pos_bboxes[sampled_indices] #torch.Size([20, 9])
            bev_embed = bev_embed.unsqueeze(0).repeat(4, 1, 1) #torch.Size([20, 32400, 256])
            
            caption_targets = gt_captions[sampling_result.pos_assigned_gt_inds[sampled_indices]]
            caplabel_targets = gt_caplabels[sampling_result.pos_assigned_gt_inds[sampled_indices]]
            caption_weights = caption_targets.new_ones((4,), dtype=torch.long)

            with torch.cuda.amp.autocast():
                caption_loss = self.llama_adapter((caption_targets, caplabel_targets, caption_weights), (bev_embed, bbox_preds))

            loss.append(caption_loss)


        return torch.stack(loss, dim=0).mean()

    def generate_caption(self, bboxes, bev_embeds):


        all_bbox_preds = [bbox[0] for bbox in bboxes]
        all_cls_scores = [bbox[1] for bbox in bboxes]


        batch_size = len(all_bbox_preds)


        caption_outputs = []
        for bs in range(batch_size):
            bbox_preds = all_bbox_preds[bs]
            cls_scores = all_cls_scores[bs]

            bev_embed = bev_embeds
            bev_embed = bev_embed.unsqueeze(0)

            downsample_proposal = True
            if downsample_proposal:
                # TODO: make sure the num_bboxes is the same as bbox coder's
                # we only evaluate several objects because the huge memory cost of LLM
                num_bboxes = min(64, len(bbox_preds))

                cls_scores = cls_scores.sigmoid()
                scores, indexs = cls_scores.view(-1).topk(num_bboxes)
                # the cls_scores is different from bevformer. it only has 1 dimension, 
                # which has already passed argmax (see mmdet3d/core/bbox/coders/transfusion_bbox_coder.py:57)
                # labels = indexs % 10
                # bbox_index = indexs // 10
                bbox_preds = bbox_preds[indexs].tensor

            else:
                num_bboxes = bbox_preds.size(0)

            
            # ps_caption_output = []
            # for object_id in range(num_bboxes):
            #     single_pred_bbox = bbox_preds[object_id].unsqueeze(0)

            #     with torch.cuda.amp.autocast():
            #         format_instruction = "Describe the object in detail."
            #         prompt = llama.format_prompt(format_instruction)
            #         caption_output = self.llama_adapter.generate((bev_embed, single_pred_bbox), ([prompt]))
                
            
            #     ps_caption_output.append(format_instruction)

            #     print(caption_output)
                
            #     # break
            # caption_outputs.append(ps_caption_output)

            # accelerate inference
            with torch.cuda.amp.autocast():
                format_instruction = "Describe the object in detail."
                prompt = llama.format_prompt(format_instruction)
                ps_caption_output = self.llama_adapter.generate((bev_embed.repeat(num_bboxes, 1, 1), bbox_preds), ([prompt]*num_bboxes))
                print(ps_caption_output)
            caption_outputs.append(ps_caption_output)
        print(f"*****************Iter Done: batch size: {batch_size} patch size:{num_bboxes} ******************")
        return caption_outputs