import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

THRESHOLD = 0.3


def interval_intersection(interval1, interval2):
    start = max(interval1[0], interval2[0])
    end = min(interval1[1], interval2[1])
    return max(0, end - start + 1)


def ensure_format(pred, gt):
    pred = [pred] if isinstance(pred[0], int) else pred
    gt = [gt] if isinstance(gt[0], int) else gt
    assert all(isinstance(interval, list) and len(interval) == 2 for interval in pred)
    assert all(isinstance(interval, list) and len(interval) == 2 for interval in gt)
    return pred, gt


def IoU(pred, gt):
    """Compute intersection over union
    pred, gt: list of list of intervals ([[s1, e1], [s2, e2], ...])
    """
    pred, gt = ensure_format(pred, gt)
    intersection = sum(interval_intersection(p, g) for p in pred for g in gt)
    total_pred = sum(p[1] - p[0] + 1 for p in pred)
    total_gt = sum(g[1] - g[0] + 1 for g in gt)
    union = total_pred + total_gt - intersection

    return intersection / union if union != 0 else 0


def IoP(pred, gt):
    """Compute intersection over predicted intervals
    pred, gt: list of list of intervals ([[s1, e1], [s2, e2], ...])
    """
    pred, gt = ensure_format(pred, gt)
    intersection = sum(interval_intersection(p, g) for p in pred for g in gt)
    total_pred = sum(p[1] - p[0] + 1 for p in pred)

    return intersection / total_pred if total_pred != 0 else 0


def IoG(pred, gt):
    """Compute intersection over ground truth intervals
    pred, gt: list of list of intervals ([[s1, e1], [s2, e2], ...])
    """
    pred, gt = ensure_format(pred, gt)
    intersection = sum(interval_intersection(p, g) for p in pred for g in gt)
    total_gt = sum(g[1] - g[0] + 1 for g in gt)

    return intersection / total_gt if total_gt != 0 else 0


def format_dict_values(d):
    return {k: f"{v:.1%}" for k, v in d.items()}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--pred_file', type=str, default="")
    parser.add_argument('--gt_file', type=str, default="")
    parser.add_argument('--results_file', type=str, default="")
    args = parser.parse_args()

    predictions = json.load(open(args.pred_file))
    labels = json.load(open(args.gt_file))
    # assert len(predictions) == len(labels)

    num_iop = defaultdict(int)
    num_iog = defaultdict(int)
    num_iou = defaultdict(int)

    iou_dict = defaultdict(list)
    iop_dict = defaultdict(list)
    iog_dict = defaultdict(list)

    overall_iou_list = []
    overall_iop_list = []
    overall_iog_list = []

    #TODO: save results
    num_failed = 0
    eval_results = []
    for idx, sample in enumerate(predictions):

        label = labels[idx]
        for item in labels:
            if item['sample_id'] == sample['sample_id']:
                label = item

        assert sample['sample_id'] == label['sample_id']
        pred_intervals = sample['T']
        true_intervals = label['T']
        category = sample['category']
        question = sample['Q']

        if not len(pred_intervals):
            num_failed += 1
            continue

        iou = IoU(pred_intervals, true_intervals)
        iop = IoP(pred_intervals, true_intervals)
        iog = IoG(pred_intervals, true_intervals)

        iou_dict[category].append(iou)
        iop_dict[category].append(iop)
        iog_dict[category].append(iog)

        overall_iou_list.append(iou)
        overall_iop_list.append(iop)
        overall_iog_list.append(iog)

        if iou > THRESHOLD:
            num_iou[category] += 1
        if iop > THRESHOLD:
            num_iop[category] += 1
        if iog > THRESHOLD:
            num_iog[category] += 1

        eval_results.append({
            "sample_id": sample['sample_id'],
            "question": question,
            "true_evidence": true_intervals,
            "pred_evidence": pred_intervals,
            "iou": round(iou, 4),
            "iop": round(iop, 4),
            "iog": round(iog, 4),
        })

    mIoU = {category: np.mean(iou_list) for category, iou_list in iou_dict.items()}
    mIoP = {category: np.mean(iop_list) for category, iop_list in iop_dict.items()}
    mIoG = {category: np.mean(iog_list) for category, iog_list in iog_dict.items()}

    overall_mIoU = np.mean(overall_iou_list)
    overall_mIoP = np.mean(overall_iop_list)
    overall_mIoG = np.mean(overall_iog_list)

    category_counts = Counter(item['category'] for item in predictions)

    iou_percentage = {category: num_iou[category] / category_counts[category] for category in num_iou}
    iop_percentage = {category: num_iop[category] / category_counts[category] for category in num_iop}
    iog_percentage = {category: num_iog[category] / category_counts[category] for category in num_iog}

    overall_iop_percentage = sum(num for _, num in num_iop.items()) / len(predictions)
    overall_iog_percentage = sum(num for _, num in num_iog.items()) / len(predictions)
    overall_iou_percentage = sum(num for _, num in num_iou.items()) / len(predictions)

    print(f"#Failed: {num_failed} / {len(predictions)}")
    print(f"mIoU: {format_dict_values(mIoU)}")
    print(f"mIoP: {format_dict_values(mIoP)}")
    print(f"mIoG: {format_dict_values(mIoG)}")
    # print(f"Percentage@IoP>{THRESHOLD}: {format_dict_values(iop_percentage)}")
    # print(f"Percentage@IoG>{THRESHOLD}: {format_dict_values(iog_percentage)}")
    print('\n')
    print(f"Overall mIoP: {overall_mIoP:.1%}")
    print(f"Overall mIoG: {overall_mIoG:.1%}")
    print(f"Overall Percentage@IoU>{THRESHOLD}: {overall_iou_percentage:.1%}")
    print(f"Overall mIoU: {overall_mIoU:.1%}")
    # print(f"Overall Percentage@IoP>{THRESHOLD}: {overall_iop_percentage:.1%}")
    # print(f"Overall Percentage@IoG>{THRESHOLD}: {overall_iog_percentage:.1%}")

    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(eval_results, f, indent=2)
