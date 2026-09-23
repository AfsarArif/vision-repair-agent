"""Transfer eval: DeepPCB-trained YOLO scored on the PKU-Market-PCB holdout.

DeepPCB is 640x640 binarized copper (copper black, substrate white); PKU is
~3000x1600 color photos of green solder-masked boards. Nothing here is fit on
PKU: the only choice is among fixed preprocess variants, and every variant is
reported.

Scoring uses ``detection_prf(..., classes=SHARED_CLASSES)``: the five classes
both datasets label. ``missing_hole`` (PKU only) cannot be predicted by the
six-class DeepPCB head, so its recall is 0 by construction and it is reported
separately; ``pin_hole`` predictions (DeepPCB only) have no PKU gold and are
counted separately as out-of-vocabulary detections.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from repair_agent.eval.metrics import box_xywh_to_xyxy, detection_prf

SHARED_CLASSES: tuple[str, ...] = ("open", "short", "mousebite", "spur", "spurious_copper")
PKU_ONLY_CLASSES: tuple[str, ...] = ("missing_hole",)
DEEPPCB_ONLY_CLASSES: tuple[str, ...] = ("pin_hole",)

# base modes x optional "_tile" suffix
_BASE_MODES = ("none", "binarize", "binarize_gray")
PREPROCESS_VARIANTS: tuple[str, ...] = (
    "none",
    "tile",
    "binarize",
    "binarize_tile",
    "binarize_gray_tile",
)

TILE_SIZE = 640
TILE_OVERLAP = 128
MERGE_IOU = 0.5

Detector = Callable[..., list[dict]]


def parse_preprocess(preprocess: str) -> tuple[str, bool]:
    """Return (base_mode, tiled). Accepts e.g. 'none', 'tile', 'binarize_tile'."""
    name = preprocess.strip().lower()
    if name == "tile":
        return "none", True
    tiled = name.endswith("_tile")
    base = name[: -len("_tile")] if tiled else name
    if base not in _BASE_MODES:
        raise ValueError(
            f"Unknown preprocess {preprocess!r}; expected one of {PREPROCESS_VARIANTS}"
        )
    return base, tiled


def binarize(img: np.ndarray, channel: str = "green") -> np.ndarray:
    """Otsu-binarize a color PCB photo into DeepPCB-style copper-black / substrate-white.

    ``channel="green"`` thresholds the G channel: on green solder mask, copper
    (traces, pour, pads) is brighter in G than bare laminate, so pixels above
    the Otsu threshold become black. ``channel="gray"`` uses luminance instead
    (on PKU this mostly keeps only bare pads). Returns a 3-channel BGR uint8.
    """
    if img.ndim == 2:
        plane = img
    elif channel == "gray":
        plane = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif channel == "green":
        plane = img[:, :, 1]
    else:
        raise ValueError(f"Unknown binarize channel: {channel}")
    plane = cv2.GaussianBlur(np.ascontiguousarray(plane), (5, 5), 0)
    _, binary = cv2.threshold(plane, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def apply_preprocess(img: np.ndarray, base_mode: str) -> np.ndarray:
    if base_mode == "none":
        return img
    if base_mode == "binarize":
        return binarize(img, channel="green")
    if base_mode == "binarize_gray":
        return binarize(img, channel="gray")
    raise ValueError(f"Unknown preprocess base mode: {base_mode}")


def tile_origins(length: int, tile: int = TILE_SIZE, overlap: int = TILE_OVERLAP) -> list[int]:
    """Start offsets covering [0, length) with fixed-size tiles; last tile flush to the edge."""
    if length <= tile:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def nms_merge(dets: list[dict], iou_threshold: float = MERGE_IOU) -> list[dict]:
    """Class-aware greedy NMS over xyxy detections (used to merge tile overlaps)."""
    kept: list[dict] = []
    by_cls: dict[str, list[dict]] = {}
    for det in dets:
        by_cls.setdefault(str(det.get("cls")), []).append(det)
    for group in by_cls.values():
        boxes = np.array([d["bbox"] for d in group], dtype=np.float64).reshape(-1, 4)
        scores = np.array([float(d.get("score", 0.0)) for d in group])
        areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
            boxes[:, 3] - boxes[:, 1], 0, None
        )
        order = np.argsort(-scores, kind="stable")
        while order.size:
            i = order[0]
            kept.append(group[i])
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            union = areas[i] + areas[rest] - inter
            iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
            order = rest[iou < iou_threshold]
    kept.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
    return kept


def make_yolo_detector(device: str | None = None) -> Detector:
    """Per-image detector with the same output as `detect_yolo`, plus a `.batch` fast path.

    `.batch(images, weights, conf)` sends all tiles of one image to Ultralytics in
    one call on `device` (auto: cuda > mps > cpu). Same model cache as detect_yolo.
    """
    from repair_agent.taxonomy import yolo_index_to_class
    from repair_agent.tools.yolo_detector import _load_model

    if device is None:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    def _batch(images: list[np.ndarray], weights: str, conf: float = 0.25) -> list[list[dict]]:
        model = _load_model(str(Path(weights).resolve()))
        results = model.predict(images, conf=conf, device=device, verbose=False)
        out: list[list[dict]] = []
        for result in results:
            dets: list[dict] = []
            if result.boxes is not None and len(result.boxes):
                xyxy = result.boxes.xyxy.cpu().numpy()
                cls = result.boxes.cls.cpu().numpy().astype(int)
                score = result.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), c, sc in zip(xyxy, cls, score):
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    dets.append(
                        {
                            "cls": yolo_index_to_class(int(c)),
                            "bbox": (x1, y1, x2 - x1, y2 - y1),
                            "score": round(float(sc), 4),
                        }
                    )
            out.append(dets)
        return out

    def _single(img: np.ndarray, weights: str, conf: float = 0.25) -> list[dict]:
        return _batch([img], weights, conf)[0]

    _single.batch = _batch  # type: ignore[attr-defined]
    _single.device = device  # type: ignore[attr-defined]
    return _single


def _run_detector(
    detector: Detector, images: list[np.ndarray], weights: str, conf: float
) -> list[list[dict]]:
    batch = getattr(detector, "batch", None)
    if batch is not None:
        return batch(images, weights, conf=conf)
    return [detector(im, weights, conf=conf) for im in images]


def _to_xyxy(raw: list[dict], dx: int = 0, dy: int = 0) -> list[dict]:
    out = []
    for d in raw:
        x1, y1, x2, y2 = box_xywh_to_xyxy(tuple(d["bbox"]))
        out.append(
            {
                "bbox": (x1 + dx, y1 + dy, x2 + dx, y2 + dy),
                "cls": d.get("cls"),
                "score": float(d.get("score", 0.0)),
            }
        )
    return out


def predict_image(
    img: np.ndarray,
    detector: Detector,
    weights: str,
    conf: float,
    preprocess: str,
    tile: int = TILE_SIZE,
    overlap: int = TILE_OVERLAP,
) -> list[dict]:
    """Run the detector with a preprocess variant. Returns xyxy boxes in full-image coords."""
    base, tiled = parse_preprocess(preprocess)
    prepared = apply_preprocess(img, base)
    if not tiled:
        return _to_xyxy(_run_detector(detector, [prepared], weights, conf)[0])
    h, w = prepared.shape[:2]
    origins = [
        (x0, y0)
        for y0 in tile_origins(h, tile, overlap)
        for x0 in tile_origins(w, tile, overlap)
    ]
    crops = [
        np.ascontiguousarray(prepared[y0 : y0 + tile, x0 : x0 + tile]) for x0, y0 in origins
    ]
    dets: list[dict] = []
    for (x0, y0), raw in zip(origins, _run_detector(detector, crops, weights, conf)):
        dets.extend(_to_xyxy(raw, dx=x0, dy=y0))
    return nms_merge(dets)


def _load_gold(gold_path: Path, limit: int | None) -> list[dict]:
    rows = []
    for line in gold_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def eval_transfer(
    gold_path: Path,
    weights: str,
    conf: float = 0.25,
    preprocess: str = "none",
    limit: int | None = None,
    detector: Detector | None = None,
    device: str | None = None,
) -> dict:
    """Macro-F1 over SHARED_CLASSES of a DeepPCB detector on PKU gold JSONL.

    `detector(img, weights, conf=...)` must return `{cls, bbox (xywh), score}`
    like `repair_agent.tools.yolo_detector.detect_yolo`. The default is
    `make_yolo_detector(device)`: same weights, cache, and output format as
    detect_yolo, but batches the tiles of one image on cuda/mps when available.
    """
    parse_preprocess(preprocess)  # validate early
    if detector is None:
        if not Path(weights).is_file():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        detector = make_yolo_detector(device)

    from repair_agent.tools.cv_tools import decode_image

    rows = _load_gold(Path(gold_path), limit)
    pred_map: dict[str, list[dict]] = {}
    gt_map: dict[str, list[dict]] = {}
    started = time.perf_counter()
    for row in rows:
        image_id = row["image_id"]
        img = decode_image(Path(row["image"]).read_bytes())
        gt_map[image_id] = [
            {"bbox": tuple(b["bbox"]), "cls": b["cls"]} for b in row.get("boxes", [])
        ]
        pred_map[image_id] = predict_image(img, detector, weights, conf, preprocess)
    elapsed = time.perf_counter() - started

    shared = detection_prf(pred_map, gt_map, classes=SHARED_CLASSES)

    # Class-agnostic localization: every gold box (incl. missing_hole) vs every prediction.
    def _agnostic(m: dict[str, list[dict]]) -> dict[str, list[dict]]:
        return {k: [{**b, "cls": "defect"} for b in v] for k, v in m.items()}

    agnostic = detection_prf(_agnostic(pred_map), _agnostic(gt_map), classes=("defect",))

    n_missing_hole = sum(1 for v in gt_map.values() for b in v if b["cls"] == "missing_hole")
    n_pin_hole = sum(1 for v in pred_map.values() for p in v if p["cls"] == "pin_hole")
    pred_counts: dict[str, int] = {}
    for v in pred_map.values():
        for p in v:
            pred_counts[str(p["cls"])] = pred_counts.get(str(p["cls"]), 0) + 1

    return {
        "stage": "transfer",
        "gold": str(gold_path),
        "weights": str(weights),
        "n_images": len(rows),
        "conf": conf,
        "preprocess": preprocess,
        "tiling": {"tile": TILE_SIZE, "overlap": TILE_OVERLAP, "merge_iou": MERGE_IOU}
        if parse_preprocess(preprocess)[1]
        else None,
        "device": getattr(detector, "device", None),
        "classes": list(SHARED_CLASSES),
        "macro_f1": shared["macro_f1"],
        "micro_f1": shared["micro_f1"],
        "micro_precision": shared["micro_precision"],
        "micro_recall": shared["micro_recall"],
        "per_class": shared["per_class"],
        "missing_hole": {
            "n_gold": n_missing_hole,
            "recall": 0.0 if n_missing_hole else None,
            "note": "PKU-only class; the DeepPCB YOLO head has no missing_hole output, "
            "so recall is 0 by construction. Excluded from macro_f1.",
        },
        "pin_hole_predictions": {
            "n_pred": n_pin_hole,
            "note": "DeepPCB-only class with no PKU gold; excluded from macro_f1.",
        },
        "class_agnostic": agnostic["per_class"]["defect"],
        "pred_counts": pred_counts,
        "seconds": round(elapsed, 1),
    }


def eval_transfer_variants(
    gold_path: Path,
    weights: str,
    conf: float = 0.25,
    variants: tuple[str, ...] = PREPROCESS_VARIANTS,
    limit: int | None = None,
    detector: Detector | None = None,
    device: str | None = None,
) -> dict:
    """Run every preprocess variant; returns {"stage", "variants": [...], "best_by_macro_f1"}."""
    reports = [
        eval_transfer(
            gold_path, weights, conf=conf, preprocess=v, limit=limit, detector=detector,
            device=device,
        )
        for v in variants
    ]
    best = max(reports, key=lambda r: r["macro_f1"]) if reports else None
    return {
        "stage": "transfer",
        "variants": reports,
        "best_by_macro_f1": best["preprocess"] if best else None,
        "note": "Variants are fixed a priori; none were tuned on PKU. Report all of them.",
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m repair_agent.eval.transfer [--preprocess all|<variant>] ..."""
    import argparse

    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="DeepPCB YOLO -> PKU transfer macro-F1")
    parser.add_argument(
        "--gold", type=Path, default=root / "data" / "processed" / "pku" / "gold_holdout.jsonl"
    )
    parser.add_argument(
        "--weights",
        default=str(root / "data" / "processed" / "deeppcb" / "weights" / "best.pt"),
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--preprocess", default="all", help=f"'all' or one of {PREPROCESS_VARIANTS}"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--out", type=Path, default=root / "evals" / "results" / "transfer.json"
    )
    args = parser.parse_args(argv)
    variants = PREPROCESS_VARIANTS if args.preprocess == "all" else (args.preprocess,)
    report = eval_transfer_variants(
        args.gold, args.weights, conf=args.conf, variants=tuple(variants),
        limit=args.limit, device=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in report["variants"]:
        per = " ".join(f"{c}={v['f1']:.3f}" for c, v in r["per_class"].items())
        print(f"{r['preprocess']:<20} macro_f1={r['macro_f1']:.4f}  {per}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
