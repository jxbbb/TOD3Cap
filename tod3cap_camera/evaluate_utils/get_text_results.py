# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.

import argparse
import json
import os
import random
import time
from typing import Tuple, Dict, Any

import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from evaluate_utils.loaders import DenseCaptionBox, load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample

from nuscenes.eval.common.data_classes import MetricData, EvalBox

from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.rouge.rouge import Rouge
from tqdm import tqdm


class DetectionEval:
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """
    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DenseCaptionBox,
                                                     verbose=verbose)
        self.gt_boxes = load_gt(self.nusc, self.eval_set, DenseCaptionBox, verbose=verbose)

        # assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
        #     "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens

        with open("data/nuscenes/final_caption_bbox_token.json", "r") as f:
            self.all_gt_caption = json.load(f)

    def main(self,
             plot_examples: int = 0,
             render_curves: bool = True) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """

        sample_tokens = self.pred_boxes.boxes.keys()
        
        records = []
        for sample_token in tqdm(sample_tokens):
            pred_box_one_sample = self.pred_boxes.boxes[sample_token]
            gt_box_one_sample = self.gt_boxes.boxes[sample_token]
        
            for pred_object_box in pred_box_one_sample:
                all_ious = []
                all_captions = []
                if len(gt_box_one_sample) == 0:
                    continue
                for gt_object_box in gt_box_one_sample:
                    iou = scale_iou(gt_object_box, pred_object_box)
                    all_ious.append(iou)
                max_iou = max(all_ious)
                
                bbox_token = gt_box_one_sample[all_ious.index(max_iou)].box_token
                if bbox_token in self.all_gt_caption:
                    reference_all_data = self.all_gt_caption[bbox_token]
                    reference = reference_all_data['attribute_caption']['attribute_caption'] + \
                            " about " + str(round(reference_all_data['depth_caption']['depth']))+" meters away" + \
                            " " + reference_all_data['localization_caption']['localization_caption'] + \
                            " is " + reference_all_data['motion_caption']['motion_caption'] + \
                            " " + reference_all_data['map_caption']['map_caption']
                else:
                    reference = "The object is ignored."
                record = {
                    'sample_token': sample_token,
                    'gt_object_box': str(gt_object_box),
                    'pred_object_box': str(pred_object_box),
                    'iou': max_iou,
                    'candidate': pred_object_box.caption,
                    'references': [reference],
                }
                # print(record)
                records.append(record)
            # break
        
        # iou_threshold = 0.25
        
        scores_1 = self.score_captions_by_4_indicators(records, iou_threshold=0.25)

        # iou_threshold = 0.5

        scores_2 = self.score_captions_by_4_indicators(records, iou_threshold=0.5)

        # print(scores)
        
        scores = {
            "iou_0.25": scores_1,
            "iou_0.5": scores_2,
        }
        print(scores)
        return records, scores

    def score_captions_by_4_indicators(self, raw_records, iou_threshold=0.):
        
        filtered_records = []
        for record in raw_records:
            if record['iou'] > iou_threshold:
                filtered_records.append(record)
        
        records = filtered_records
        
        references = {}
        candidates = {}
        for i, record in enumerate(records):
            references[i] = [' '.join(token for token in ref.split() if token not in [",", "."])
                                for ref in record['references']]
            candidates[i] = [' '.join(token for token in record['candidate'].split()
                                if token not in [",", "."])]

        if len(references) == 0 or len(candidates) == 0:
            return 0., [0. for _ in range(len(records))]
        
        scores={}
        
        # mean_meteor
        meteor_scorer = Meteor()
        mean_meteor, _ = meteor_scorer.compute_score(references, candidates)
        scores['meteor']=mean_meteor
        meteor_scorer.close()

        # bleu
        bleu_scorer=Bleu()
        bleu_4_score,_=bleu_scorer.compute_score(references,candidates)
        for i in range(4):
            scores["bleu{}".format(i+1)]=bleu_4_score[i]

        # mean_cide
        cider_scorer=Cider()
        mean_cider,_=cider_scorer.compute_score(references,candidates)
        scores['cider']=mean_cider

        # mean_rouge
        rouge_scorer=Rouge()
        mean_rouge,_=rouge_scorer.compute_score(references,candidates)
        scores['rouge']=mean_rouge

        for key in scores.keys():
            scores[key]*=len(records)/len(raw_records)

        return scores



def scale_iou(sample_annotation: EvalBox, sample_result: EvalBox) -> float:
    """
    This method compares predictions to the ground truth in terms of scale.
    It is equivalent to intersection over union (IOU) between the two boxes in 3D,
    if we assume that the boxes are aligned, i.e. translation and rotation are considered identical.
    :param sample_annotation: GT annotation sample.
    :param sample_result: Predicted sample.
    :return: Scale IOU.
    """
    # Validate inputs.
    sa_size = np.array(sample_annotation.size)
    sr_size = np.array(sample_result.size)
    assert all(sa_size > 0), 'Error: sample_annotation sizes must be >0.'
    assert all(sr_size > 0), 'Error: sample_result sizes must be >0.'

    # Compute IOU.
    min_wlh = np.minimum(sa_size, sr_size)
    volume_annotation = np.prod(sa_size)
    volume_result = np.prod(sr_size)
    intersection = np.prod(min_wlh)  # type: float
    union = volume_annotation + volume_result - intersection  # type: float
    iou = intersection / union

    return iou


class NuScenesEval(DetectionEval):
    """
    Dummy class for backward-compatibility. Same as DetectionEval.
    """


if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes detection results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--result_path', type=str, default='test/bevformer_tiny_test/Fri_Nov__1_04_03_23_2024/pts_bbox/results_nusc_formatted.json', help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='data/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the CVPR 2019 configuration will be used.')
    parser.add_argument('--plot_examples', type=int, default=10,
                        help='How many example visualizations to write to disk.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render PR and TP curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    plot_examples_ = args.plot_examples
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)

    if config_path == '':
        cfg_ = config_factory('detection_cvpr_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))

    nusc_ = NuScenes(version=version_, verbose=verbose_, dataroot=dataroot_)
    nusc_eval = DetectionEval(nusc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
                              output_dir=output_dir_, verbose=verbose_)
    records, scores = nusc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_)
    
    with open(os.path.join(os.path.dirname(args.result_path), "caption_records.json"), "w") as f:
        json.dump(records, f)
    with open(os.path.join(os.path.dirname(args.result_path), "caption_scores.json"), "w") as f:
        json.dump(scores, f)
