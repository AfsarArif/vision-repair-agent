#!/usr/bin/env python3
"""Stage-wise eval: synthetic template-diff, RAG Recall@k, optional DeepPCB CV."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from repair_agent.config import settings  # noqa: E402
from repair_agent.eval.metrics import (  # noqa: E402
    box_xywh_to_xyxy,
    exact_match,
    iou_xyxy,
    mean_average_precision,
    mean_reciprocal_rank,
    recall_at_k,
)
from repair_agent.eval.synthetic import apply_open, apply_short, make_trace_template  # noqa: E402
from repair_agent.tools.cv_tools import decode_image  # noqa: E402
from repair_agent.tools.template_diff import localize_defects  # noqa: E402
from repair_agent.tools.yolo_detector import detect_yolo, resolve_weights  # noqa: E402

RESULTS_DIR = ROOT / "evals" / "results"


def _write_report(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {path}")
    return path


def eval_synthetic() -> dict:
    cases = [
        ("open", apply_open),
        ("short", apply_short),
    ]
    ious = []
    hits = 0
    for name, apply_fn in cases:
        template = make_trace_template()
        test, gold_xyxy = apply_fn(template)
        detections = localize_defects(test, template, min_area=10, diff_threshold=20)
        best = 0.0
        for det in detections:
            pred_xyxy = box_xywh_to_xyxy(tuple(det["bbox"]))
            best = max(best, iou_xyxy(pred_xyxy, gold_xyxy))
        ious.append(best)
        if best >= 0.5:
            hits += 1
        print(f"synthetic {name}: best_iou={best:.3f} n_dets={len(detections)}")
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    return {
        "stage": "synthetic",
        "mean_iou": round(mean_iou, 4),
        "localization_hit_rate_iou50": round(hits / len(cases), 4),
        "n_cases": len(cases),
        "note": "Class-agnostic template-diff on generated pairs. No DeepPCB download required.",
    }


def _source_ids(docs: list[dict]) -> list[str]:
    ids: list[str] = []
    for doc in docs:
        meta = doc.get("metadata") or {}
        if meta.get("source_id"):
            ids.append(str(meta["source_id"]))
        source = Path(str(meta.get("source", "")))
        if source.stem:
            ids.append(source.stem.lower())
            if "adapters" in source.parts:
                ids.append(f"adapter-{source.stem.lower()}")
    # preserve order, drop dupes
    seen: set[str] = set()
    ordered: list[str] = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def eval_rag(queries_path: Path) -> dict:
    from repair_agent.rag.retriever import aretrieve

    rows = [
        json.loads(line)
        for line in queries_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    recalls = []
    mrrs = []
    import asyncio

    async def _run():
        for row in rows:
            docs = await aretrieve(row["query"], k=row.get("k", 5))
            retrieved = _source_ids(docs)
            expected = row["expected_source_ids"]
            r = recall_at_k(retrieved, expected, k=row.get("k", 5))
            m = mean_reciprocal_rank(retrieved, expected)
            recalls.append(r)
            mrrs.append(m)
            print(f"rag q={row['id']} recall@k={r:.2f} mrr={m:.2f} got={retrieved[:5]}")

    asyncio.run(_run())
    n = len(rows) or 1
    return {
        "stage": "rag",
        "n_queries": len(rows),
        "recall_at_5": round(sum(recalls) / n, 4),
        "mrr": round(sum(mrrs) / n, 4),
        "queries_file": str(queries_path),
    }


def _as_xyxy_preds(raw: list[dict]) -> list[dict]:
    return [
        {
            "bbox": box_xywh_to_xyxy(tuple(d["bbox"])),
            "cls": d.get("cls"),
            "score": d.get("score", 0.0),
        }
        for d in raw
    ]


def eval_cv(
    gold_path: Path,
    backend: str,
    weights: str | None = None,
    conf: float = 0.25,
    limit: int | None = None,
) -> dict:
    resolved = resolve_weights(weights) if backend == "yolo" else None
    if backend == "yolo" and resolved is None:
        raise FileNotFoundError(
            "YOLO backend requested but no weights file found. "
            "Train with scripts/train_detector.py --run or pass --weights."
        )

    pred_map: dict[str, list[dict]] = defaultdict(list)
    gt_map: dict[str, list[dict]] = defaultdict(list)
    n = 0
    for line in gold_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if limit is not None and n >= limit:
            break
        row = json.loads(line)
        image_id = row["image_id"]
        test = decode_image(Path(row["image"]).read_bytes())
        gts = [{"bbox": tuple(b["bbox"]), "cls": b["cls"]} for b in row.get("boxes", [])]
        gt_map[image_id] = gts
        if backend == "yolo":
            raw = detect_yolo(test, str(resolved), conf=conf)
        elif backend in ("template_diff", "heuristic"):
            template = decode_image(Path(row["template"]).read_bytes())
            raw = localize_defects(test, template)
        else:
            raise ValueError(f"Unknown CV eval backend: {backend}")
        pred_map[image_id] = _as_xyxy_preds(raw)
        n += 1

    class_aware = backend == "yolo"
    metrics = mean_average_precision(pred_map, gt_map, class_aware=class_aware)
    metrics.update(
        {
            "stage": "cv",
            "n_images": n,
            "backend": backend,
            "class_aware": class_aware,
            "gold": str(gold_path),
            "weights": str(resolved) if resolved else None,
            "conf": conf,
        }
    )
    return metrics


def eval_ultralytics_val(data_yaml: Path, weights: Path, split: str = "test") -> dict:
    # Optional `train` extra — not imported at module load so pytest stays light.
    from ultralytics import YOLO

    model = YOLO(str(weights))
    result = model.val(data=str(data_yaml), split=split, plots=False, verbose=False)
    raw = getattr(result, "results_dict", None) or {}
    metrics = {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}
    metrics.update(
        {
            "stage": "cv_ultralytics",
            "split": split,
            "weights": str(weights),
            "data": str(data_yaml),
        }
    )
    return metrics


def eval_ocr(gold_path: Path) -> dict:
    from repair_agent.tools.ocr_tools import decode_and_preprocess, extract_text, find_designator

    n = 0
    hits = 0
    for line in gold_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        img = Path(row["image"]).read_bytes()
        text = extract_text(decode_and_preprocess(img))
        pred = find_designator(text)
        if exact_match(pred, row.get("designator")):
            hits += 1
        n += 1
    return {
        "stage": "ocr",
        "n": n,
        "exact_match": round(hits / n, 4) if n else 0.0,
        "note": "FPIC/designator gold only. Do not score DeepPCB.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["synthetic", "rag", "cv", "ocr", "all"],
        default="synthetic",
    )
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument(
        "--rag-queries",
        type=Path,
        default=ROOT / "evals" / "rag_queries.jsonl",
    )
    parser.add_argument(
        "--backend",
        choices=["heuristic", "template_diff", "yolo"],
        default=None,
        help="CV backend for --stage cv (default: yolo if weights exist, else template_diff)",
    )
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None, help="Score at most N gold images")
    parser.add_argument(
        "--ultralytics-val",
        action="store_true",
        help="Also run Ultralytics val() on data/processed/deeppcb/deeppcb.yaml",
    )
    args = parser.parse_args()

    stages = ["synthetic", "rag", "cv", "ocr"] if args.stage == "all" else [args.stage]
    reports = []
    for stage in stages:
        if stage == "synthetic":
            reports.append(eval_synthetic())
        elif stage == "rag":
            if not args.rag_queries.exists():
                print(f"missing {args.rag_queries}")
                return 1
            try:
                reports.append(eval_rag(args.rag_queries))
            except Exception as exc:
                print(f"RAG eval skipped: {exc}")
                reports.append({"stage": "rag", "skipped": True, "error": str(exc)})
        elif stage == "cv":
            gold = args.gold or (ROOT / "data" / "processed" / "deeppcb" / "gold_test.jsonl")
            weights_arg = str(args.weights) if args.weights else (settings.YOLO_WEIGHTS or None)
            resolved = resolve_weights(weights_arg)
            backend = args.backend
            if backend is None:
                backend = "yolo" if resolved is not None else "template_diff"
            if not gold.exists():
                print(f"missing {gold}; run prepare_deeppcb.py or use --stage synthetic")
                reports.append({"stage": "cv", "skipped": True, "gold": str(gold)})
            else:
                try:
                    reports.append(
                        eval_cv(
                            gold,
                            backend=backend,
                            weights=weights_arg,
                            conf=args.conf,
                            limit=args.limit,
                        )
                    )
                except (FileNotFoundError, ImportError) as exc:
                    print(f"CV eval skipped: {exc}")
                    reports.append({"stage": "cv", "skipped": True, "error": str(exc)})
            if args.ultralytics_val:
                yaml_path = ROOT / "data" / "processed" / "deeppcb" / "deeppcb.yaml"
                if resolved is None or not yaml_path.exists():
                    reports.append({"stage": "cv_ultralytics", "skipped": True})
                else:
                    reports.append(eval_ultralytics_val(yaml_path, resolved))
        elif stage == "ocr":
            if not args.gold or not args.gold.exists():
                print("OCR eval needs --gold JSONL with image + designator fields")
                reports.append({"stage": "ocr", "skipped": True})
            else:
                reports.append(eval_ocr(args.gold))
        else:
            raise ValueError(f"Unknown eval stage: {stage}")

    _write_report("latest", {"reports": reports})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
