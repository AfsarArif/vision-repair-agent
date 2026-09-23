"""Stage-wise metrics shared by evals/run_eval.py and unit tests."""

from __future__ import annotations

from collections import defaultdict


def box_xywh_to_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return (x, y, x + w, y + h)


def iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection-over-union for boxes as (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def greedy_match(
    predictions: list[dict],
    ground_truth: list[dict],
    iou_threshold: float = 0.5,
    class_aware: bool = True,
) -> list[tuple[int, int, float]]:
    """Greedy IoU match. Each dict needs `bbox` as xyxy and optional `cls`.

    Returns list of (pred_index, gt_index, iou).
    """
    pairs: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(predictions):
        for gi, gt in enumerate(ground_truth):
            if class_aware and pred.get("cls") != gt.get("cls"):
                continue
            score = iou_xyxy(tuple(pred["bbox"]), tuple(gt["bbox"]))
            if score >= iou_threshold:
                pairs.append((score, pi, gi))
    pairs.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append((pi, gi, score))
    return matches


def average_precision_at_iou(
    predictions: list[dict],
    ground_truth: list[dict],
    iou_threshold: float = 0.5,
    class_aware: bool = True,
) -> float:
    """VOC-style AP at a single IoU threshold.

    `predictions` must include `score` (higher = more confident) and `bbox` xyxy.
    """
    if not ground_truth:
        return 1.0 if not predictions else 0.0

    ordered = sorted(predictions, key=lambda p: float(p.get("score", 0.0)), reverse=True)
    gt_matched = [False] * len(ground_truth)
    tp: list[int] = []
    fp: list[int] = []

    for pred in ordered:
        best_iou = 0.0
        best_gi = -1
        for gi, gt in enumerate(ground_truth):
            if gt_matched[gi]:
                continue
            if class_aware and pred.get("cls") != gt.get("cls"):
                continue
            score = iou_xyxy(tuple(pred["bbox"]), tuple(gt["bbox"]))
            if score > best_iou:
                best_iou = score
                best_gi = gi
        if best_gi >= 0 and best_iou >= iou_threshold:
            gt_matched[best_gi] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    cum_tp = 0
    cum_fp = 0
    n_gt = len(ground_truth)
    precisions: list[float] = []
    recalls: list[float] = []
    for t, f in zip(tp, fp):
        cum_tp += t
        cum_fp += f
        precisions.append(cum_tp / max(cum_tp + cum_fp, 1))
        recalls.append(cum_tp / n_gt)

    return _voc_ap(recalls, precisions)


def mean_average_precision(
    predictions_by_image: dict[str, list[dict]],
    ground_truth_by_image: dict[str, list[dict]],
    iou_threshold: float = 0.5,
    class_aware: bool = True,
) -> dict[str, float]:
    """mAP@iou across images. Returns per-class AP plus `mAP`."""
    by_class_pred: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    by_class_gt: dict[str, list[tuple[str, dict]]] = defaultdict(list)

    if class_aware:
        for image_id, preds in predictions_by_image.items():
            for pred in preds:
                by_class_pred[str(pred.get("cls", "unknown"))].append((image_id, pred))
        for image_id, gts in ground_truth_by_image.items():
            for gt in gts:
                by_class_gt[str(gt.get("cls", "unknown"))].append((image_id, gt))
        classes = sorted(set(by_class_pred) | set(by_class_gt))
        aps: dict[str, float] = {}
        for cls in classes:
            pred_map: dict[str, list[dict]] = defaultdict(list)
            gt_map: dict[str, list[dict]] = defaultdict(list)
            for image_id, pred in by_class_pred.get(cls, []):
                pred_map[image_id].append(pred)
            for image_id, gt in by_class_gt.get(cls, []):
                gt_map[image_id].append(gt)
            aps[cls] = _ap_over_images(pred_map, gt_map, iou_threshold, class_aware=False)
        aps["mAP"] = sum(aps.values()) / len(aps) if aps else 0.0
        return aps

    ap = _ap_over_images(
        predictions_by_image, ground_truth_by_image, iou_threshold, class_aware=False
    )
    return {"defect": ap, "mAP": ap}


def _ap_over_images(
    predictions_by_image: dict[str, list[dict]],
    ground_truth_by_image: dict[str, list[dict]],
    iou_threshold: float,
    class_aware: bool,
) -> float:
    all_preds: list[dict] = []
    all_gt: list[dict] = []
    # Offset boxes per image so greedy AP pooling does not match across images.
    shift = 0.0
    for image_id in sorted(set(predictions_by_image) | set(ground_truth_by_image)):
        for pred in predictions_by_image.get(image_id, []):
            x1, y1, x2, y2 = pred["bbox"]
            all_preds.append({**pred, "bbox": (x1 + shift, y1 + shift, x2 + shift, y2 + shift)})
        for gt in ground_truth_by_image.get(image_id, []):
            x1, y1, x2, y2 = gt["bbox"]
            all_gt.append({**gt, "bbox": (x1 + shift, y1 + shift, x2 + shift, y2 + shift)})
        shift += 10_000.0
    return average_precision_at_iou(all_preds, all_gt, iou_threshold, class_aware=class_aware)


def _voc_ap(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def detection_prf(
    predictions_by_image: dict[str, list[dict]],
    ground_truth_by_image: dict[str, list[dict]],
    iou_threshold: float = 0.5,
    classes: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Per-class precision / recall / F1 at a fixed operating point, plus macro-F1.

    Boxes are xyxy. Matching is class-aware and greedy by IoU per image. When
    `classes` is given, predictions and gold outside that set are ignored (use
    this to score only the classes two datasets share).
    """
    keep = set(classes) if classes is not None else None
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for image_id in set(predictions_by_image) | set(ground_truth_by_image):
        preds = [
            p for p in predictions_by_image.get(image_id, [])
            if keep is None or p.get("cls") in keep
        ]
        gts = [
            g for g in ground_truth_by_image.get(image_id, [])
            if keep is None or g.get("cls") in keep
        ]
        matches = greedy_match(preds, gts, iou_threshold=iou_threshold, class_aware=True)
        matched_p = {pi for pi, _, _ in matches}
        matched_g = {gi for _, gi, _ in matches}
        for pi, pred in enumerate(preds):
            counts[str(pred.get("cls"))]["tp" if pi in matched_p else "fp"] += 1
        for gi, gt in enumerate(gts):
            if gi not in matched_g:
                counts[str(gt.get("cls"))]["fn"] += 1

    per_class: dict[str, dict[str, float]] = {}
    for cls in sorted(keep if keep is not None else counts):
        c = counts[cls]
        precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            **c,
        }
    macro = sum(v["f1"] for v in per_class.values()) / len(per_class) if per_class else 0.0
    tp = sum(v["tp"] for v in per_class.values())
    fp = sum(v["fp"] for v in per_class.values())
    fn = sum(v["fn"] for v in per_class.values())
    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    return {
        "per_class": per_class,
        "macro_f1": round(macro, 4),
        "micro_f1": round(micro_f1, 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
    }


def recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 5) -> float:
    """Fraction of expected source ids found in the top-k retrieved ids."""
    if not expected_ids:
        return 1.0
    top = set(retrieved_ids[:k])
    hits = sum(1 for e in expected_ids if e in top)
    return hits / len(expected_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    expected = set(expected_ids)
    for i, item in enumerate(retrieved_ids, start=1):
        if item in expected:
            return 1.0 / i
    return 0.0


def exact_match(predicted: str | None, gold: str | None) -> bool:
    if predicted is None or gold is None:
        return predicted is None and gold is None
    return _normalize_text(predicted) == _normalize_text(gold)


def _normalize_text(value: str) -> str:
    return "".join(value.split()).upper()
