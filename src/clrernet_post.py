# ===== CLRerNet post-processing + CULane metric (pure numpy / scipy / cv2) =====
# Faithful port of: libs/models/dense_heads/clrernet_head.py (get_lanes, predictions_to_lanes),
# libs/models/layers/nms (CUDA lane NMS), libs/utils/lane_utils.py and
# libs/datasets/metrics/culane_metric.py.  Works on outputs of PyTorch, ONNX Runtime or TensorRT.
import os
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, splev, splprep
from scipy.optimize import linear_sum_assignment

# ---- constants from configs/clrernet/base_clrernet.py + clrernet_culane_dla34_ema.py ----
IMG_W, IMG_H = 800, 320          # network input
ORI_W, ORI_H = 1640, 590         # CULane frame
CUT_HEIGHT = 270                 # rows cropped from the top
N_OFFSETS = 72                   # rows per lane
N_STRIPS = N_OFFSETS - 1
CONF_THRESHOLD = 0.43            # EMA model (0.41 for the non-EMA model)
NMS_THRES = 50                   # pixels (mean |dx| on the 800-wide input)
NMS_TOPK = 4
PRIOR_YS = np.linspace(1, 0, N_OFFSETS, dtype=np.float32)
CATEGORIES = ['test0_normal', 'test1_crowd', 'test2_hlight', 'test3_shadow', 'test4_noline',
              'test5_arrow', 'test6_curve', 'test7_cross', 'test8_night']


def preprocess_bgr(img_bgr):
    """Same as the repo test pipeline: crop rows 270:590, resize to 800x320 (bilinear),
    keep BGR order, scale to [0,1].  Returns float32 CHW."""
    img = img_bgr[CUT_HEIGHT:ORI_H, 0:ORI_W]
    img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32) / 255.0


def _dev_iou(a, b, thr):
    """Port of devIoU() in nms_kernel.cu (True -> b is suppressed by a)."""
    start_a = int(a[2] * N_STRIPS + 0.5)
    start_b = int(b[2] * N_STRIPS + 0.5)
    start = max(start_a, start_b)
    end_a = int(start_a + a[4] - 1 + 0.5 - (1 if (a[4] - 1) < 0 else 0))
    end_b = int(start_b + b[4] - 1 + 0.5 - (1 if (b[4] - 1) < 0 else 0))
    end = min(end_a, end_b, N_OFFSETS - 1)
    if end < start:
        return False
    dist = np.abs(a[5 + start:6 + end] - b[5 + start:6 + end]).sum()
    return dist < thr * (end - start + 1)


def lane_nms(boxes, scores, overlap=NMS_THRES, top_k=NMS_TOPK):
    """Greedy lane NMS, identical decision rule to the repo's CUDA extension.
    boxes: (N, 5+72) = [logit0, logit1, start_y, start_x, length, xs*799 ...]."""
    order = np.argsort(-scores, kind='stable')
    removed = np.zeros(len(order), dtype=bool)
    keep = []
    for ii, i in enumerate(order):
        if removed[ii]:
            continue
        keep.append(i)
        if len(keep) == top_k:
            break
        for jj in range(ii + 1, len(order)):
            if not removed[jj] and _dev_iou(boxes[i], boxes[order[jj]], overlap):
                removed[jj] = True
    return np.array(keep, dtype=np.int64)


class Lane:
    """lane_utils.Lane: cubic spline x(y) on normalised coords."""
    def __init__(self, points, invalid_value=-2.0, metadata=None):
        self.points = points
        self.invalid_value = invalid_value
        self.function = InterpolatedUnivariateSpline(points[:, 1], points[:, 0], k=min(3, len(points) - 1))
        self.min_y = points[:, 1].min() - 0.01
        self.max_y = points[:, 1].max() + 0.01
        self.metadata = metadata or {}

    def __call__(self, lane_ys):
        lane_xs = self.function(lane_ys)
        lane_xs[(lane_ys < self.min_y) | (lane_ys > self.max_y)] = self.invalid_value
        return lane_xs


def predictions_to_lanes(pred_xs, anchor_params, lengths, extend_bottom=True):
    prior_ys = PRIOR_YS.astype(np.float64)
    lanes = []
    for lane_xs, lane_param, length in zip(pred_xs, anchor_params, lengths):
        start = min(max(0, int(round((1 - float(lane_param[0])) * N_STRIPS))), N_STRIPS)
        length = int(round(float(length)))
        end = min(start + length - 1, len(prior_ys) - 1)
        if extend_bottom:
            edge = (lane_xs[:start] >= 0.0) & (lane_xs[:start] <= 1.0)
            start -= int(np.cumprod(edge[::-1]).sum())
        lane_ys = prior_ys[start:end + 1][::-1]
        lxs = lane_xs[start:end + 1].astype(np.float64)[::-1]
        lane_ys = (lane_ys * (ORI_H - CUT_HEIGHT) + CUT_HEIGHT) / ORI_H
        if len(lxs) <= 1:
            continue
        lanes.append(np.stack([lxs, lane_ys], axis=1))
    return lanes


def decode_one(cls_logits, anchor_params, lengths, xs, conf_threshold=CONF_THRESHOLD,
               use_nms=True, extend_bottom=True):
    """Decode ONE image (final refine stage) -> list of (N,2) normalised [x, y] arrays.
    cls_logits (192,2), anchor_params (192,3), lengths (192,1), xs (192,72)."""
    cls_logits = cls_logits.astype(np.float32)
    e = np.exp(cls_logits - cls_logits.max(-1, keepdims=True))
    scores = (e / e.sum(-1, keepdims=True))[:, 1]
    keep = scores >= conf_threshold
    if not keep.any():
        return [], np.zeros(0, np.float32)
    scores, xs, lengths, anchor_params = scores[keep], xs[keep], lengths[keep], anchor_params[keep]
    logits = cls_logits[keep]
    if use_nms:
        nms_preds = np.concatenate([
            logits,
            np.stack([1 - anchor_params[:, 0], anchor_params[:, 1]], axis=1),
            lengths * N_STRIPS,
            xs * (IMG_W - 1)], axis=1).astype(np.float32)
        k = lane_nms(nms_preds, scores)
        xs, scores, lengths, anchor_params = xs[k], scores[k], lengths[k], anchor_params[k]
    lengths = np.round(lengths * N_STRIPS)[:, 0]
    return predictions_to_lanes(xs, anchor_params, lengths, extend_bottom), scores


def lanes_to_culane_string(lanes, y_step=2):
    """CULaneMetric.get_prediction_string (spline-resampled every 2 px)."""
    ys = np.arange(0, ORI_H, y_step) / ORI_H
    out = []
    for pts in lanes:
        lane = Lane(pts)
        xs = lane(ys)
        valid = (xs >= 0) & (xs < 1)
        lane_xs = (xs * ORI_W)[valid][::-1]
        lane_ys = (ys[valid] * ORI_H)[::-1]
        if len(lane_xs) < 2:
            continue
        out.append(" ".join("{:.5f} {:.5f}".format(x, y) for x, y in zip(lane_xs, lane_ys)))
    return "\n".join(out)


def lanes_to_pixels(lanes, y_step=2):
    """Lanes in original-image pixel coords (for drawing)."""
    s = lanes_to_culane_string(lanes, y_step)
    return [np.array(list(map(float, l.split()))).reshape(-1, 2) for l in s.split("\n") if l.strip()]


# ------------------------------- CULane metric -------------------------------
def interp(points, n=5):
    x = [x for x, _ in points]
    y = [y for _, y in points]
    tck, u = splprep([x, y], s=0, t=5, k=min(3, len(points) - 1))
    u = np.linspace(0.0, 1.0, num=(len(u) - 1) * n + 1)
    return np.array(splev(u, tck)).T


def draw_lane(lane, img=None, img_shape=None, width=30, color=(255, 255, 255)):
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    lane = lane.astype(np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img, tuple(int(v) for v in p1), tuple(int(v) for v in p2), color, thickness=width)
    return img


def discrete_cross_iou(xs, ys, width=30, img_shape=(590, 1640, 3)):
    xs = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in xs]
    ys = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in ys]
    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = (x & y).sum() / (x | y).sum()
    return ious


def culane_metric(pred, anno, cat, width=30, iou_thresholds=(0.5,), img_shape=(590, 1640, 3)):
    interp_pred = [interp(p, n=5) for p in pred]
    interp_anno = [interp(a, n=5) for a in anno]
    ious = discrete_cross_iou(interp_pred, interp_anno, width=width, img_shape=img_shape)
    row_ind, col_ind = linear_sum_assignment(1 - ious)
    pred_ious = np.zeros(len(pred))
    pred_ious[row_ind] = ious[row_ind, col_ind]
    return {'n_gt': len(anno), 'cat': cat, 'hits': [pred_ious > t for t in iou_thresholds]}


def load_culane_img_data(path):
    with open(path, 'r') as f:
        img_data = [list(map(float, line.split())) for line in f.readlines()]
    img_data = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)] for lane in img_data]
    return [lane for lane in img_data if len(lane) >= 2]


def load_categories(categories_dir):
    data_cats = {}
    for c in CATEGORIES:
        p = Path(categories_dir) / f"{c}.txt"
        if p.exists():
            with open(p) as f:
                data_cats.update({k.strip(): c for k in f if k.strip()})
    return data_cats


def _metric_job(args):
    pred_path, anno_path, cat, iou_thresholds = args
    return culane_metric(load_culane_img_data(pred_path), load_culane_img_data(anno_path), cat,
                         width=30, iou_thresholds=list(iou_thresholds))


def eval_predictions(pred_dir, anno_dir, rel_paths, categories_dir,
                     iou_thresholds=(0.1, 0.5, 0.75), workers=None, verbose=True):
    """rel_paths: list like 'driver_100_30frame/xxx.MP4/00000.jpg' (no leading slash)."""
    from multiprocessing import Pool
    data_cats = load_categories(categories_dir)
    jobs = []
    for p in rel_paths:
        txt = p.replace('.jpg', '.lines.txt')
        jobs.append((os.path.join(pred_dir, txt), os.path.join(anno_dir, txt),
                     data_cats.get(p, 'test0_normal'), tuple(iou_thresholds)))
    workers = workers or os.cpu_count()
    with Pool(workers) as pool:
        results = pool.map(_metric_job, jobs, chunksize=64)
    eps = 1e-8
    out = {}
    for k, thr in enumerate(iou_thresholds):
        for i in range(len(CATEGORIES) + 1):
            cats = CATEGORIES if i == 0 else [CATEGORIES[i - 1]]
            rs = [r for r in results if r['cat'] in cats]
            if not rs:
                continue
            n_gt = sum(r['n_gt'] for r in rs)
            hits = np.concatenate([r['hits'][k] for r in rs]) if rs else np.zeros(0, bool)
            tp = int(hits.sum()); fp = int(len(hits) - tp); fn = n_gt - tp
            prec = tp / (tp + fp + eps); rec = tp / (n_gt + eps); f1 = 2 * prec * rec / (prec + rec + eps)
            name = 'all' if i == 0 else cats[0]
            out[(thr, name)] = dict(N=len(rs), TP=tp, FP=fp, FN=fn, Precision=prec, Recall=rec, F1=f1)
            if verbose and (i == 0 or thr == 0.5):
                print(f"IoU={thr:<4} {name:13s} N={len(rs):5d} TP={tp:6d} FP={fp:6d} FN={fn:6d} "
                      f"P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
    return out