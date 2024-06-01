import pickle

import json
from mmdet3d.core import bbox3d2result


pickle_path = "cam+lidar.pkl"

with open(pickle_path, "rb") as f:
    data = pickle.load(f)

with open("/data/jinbu/nuscenes-caption/Baseline/dense_3D_captioning/BEVFormer/test/bevformer_tiny/finetune_bev_train_with_pred_e10/pts_bbox/results_nusc.json", "r") as f:
    test_data = json.load(f)

bbox_results = []
for result_dict in data:
    bboxes, scores, labels, captions = result_dict['boxes_3d'], result_dict['scores_3d'], result_dict['labels_3d'], result_dict['captions_3d']
    
    num_bboxes = 20
    cls_scores = scores.float().sigmoid()
    scores, indexs = cls_scores.view(-1).topk(num_bboxes)
    labels = indexs % 10
    bbox_index = indexs // 10
    bboxes = bboxes[bbox_index]
    scores = scores[bbox_index]
    labels = labels[bbox_index]
    
    bbox_result = bbox3d2result(bboxes, scores, labels, caps=captions)
    bbox_results.append(bbox_result)

