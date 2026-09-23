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
from repair_agent.eval.ocr import eval_ocr  # noqa: E402
from repair_agent.eval.metrics import (  # noqa: E402
    box_xywh_to_xyxy,
    detection_prf,
    greedy_match,
    iou_xyxy,
    mean_average_precision,
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


def _doc_aliases(doc: dict) -> set[str]:
    """Every id a gold query may use for one retrieved chunk."""
    meta = doc.get("metadata") or {}
    aliases: set[str] = set()
    if meta.get("source_id"):
        aliases.add(str(meta["source_id"]))
    source = Path(str(meta.get("source", "")))
    if source.stem:
        aliases.add(source.stem.lower())
        if "adapters" in source.parts:
            aliases.add(f"adapter-{source.stem.lower()}")
    return aliases


def _primary_id(doc: dict) -> str:
    meta = doc.get("metadata") or {}
    return str(meta.get("source_id") or Path(str(meta.get("source", ""))).stem)


def eval_rag(queries_path: Path, reranker: str | None = None) -> dict:
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
            docs = await aretrieve(
                row["query"],
                k=row.get("k", 5),
                defect_class=row.get("defect_class"),
                reranker=reranker,
            )
            k = row.get("k", 5)
            expected = row["expected_source_ids"]
            # Rank by retrieved chunk (what the diagnosis prompt sees), not by alias id.
            alias_sets = [_doc_aliases(d) for d in docs[:k]]
            found = {e for e in expected if any(e in a for a in alias_sets)}
            r = len(found) / len(expected) if expected else 1.0
            m = next((1.0 / i for i, a in enumerate(alias_sets, 1) if a & set(expected)), 0.0)
            recalls.append(r)
            mrrs.append(m)
            got = [_primary_id(d) for d in docs[:k]]
            print(f"rag q={row['id']} recall@k={r:.2f} mrr={m:.2f} got={got}")

    asyncio.run(_run())
    n = len(rows) or 1
    return {
        "stage": "rag",
        "n_queries": len(rows),
        "recall_at_5": round(sum(recalls) / n, 4),
        "mrr": round(sum(mrrs) / n, 4),
        "reranker": settings.RAG_RERANKER if reranker is None else (reranker or None),
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


def _load_gold(gold_path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit is not None else rows


def _top_is_correct(preds: list[dict], gts: list[dict]) -> bool:
    """Does the agent's `defect_type` (highest-score box) match a gold box of that class?"""
    if not preds:
        return not gts
    top = max(preds, key=lambda p: p["score"])
    return bool(greedy_match([top], gts, iou_threshold=0.5, class_aware=True))


def eval_threshold_sweep(
    gold_path: Path,
    weights: str | None,
    limit: int | None = None,
    grid: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20)),
) -> dict:
    """Sweep YOLO confidence on the DeepPCB **val** split (never test).

    1. Detection operating point: micro/macro F1 vs box confidence.
    2. Self-correct gate: `defect_confidence < t` should flag images whose top
       detection is wrong. Reports trigger rate and flag precision/recall/F1.
    """
    resolved = resolve_weights(weights)
    if resolved is None:
        raise FileNotFoundError("YOLO weights not found; train first or pass --weights.")
    rows = _load_gold(gold_path, limit)
    pred_map: dict[str, list[dict]] = {}
    gt_map: dict[str, list[dict]] = {}
    for row in rows:
        test = decode_image(Path(row["image"]).read_bytes())
        pred_map[row["image_id"]] = _as_xyxy_preds(detect_yolo(test, str(resolved), conf=0.01))
        gt_map[row["image_id"]] = [{"bbox": tuple(b["bbox"]), "cls": b["cls"]} for b in row["boxes"]]

    detection = []
    for t in grid:
        kept = {k: [p for p in v if p["score"] >= t] for k, v in pred_map.items()}
        prf = detection_prf(kept, gt_map)
        detection.append(
            {
                "conf": t,
                "micro_f1": prf["micro_f1"],
                "macro_f1": prf["macro_f1"],
                "precision": prf["micro_precision"],
                "recall": prf["micro_recall"],
            }
        )
    best_det = max(detection, key=lambda d: (d["micro_f1"], -d["conf"]))

    # Gate uses boxes at the chosen operating point, like the running agent.
    op = best_det["conf"]
    per_image = []
    for image_id, preds in pred_map.items():
        kept = [p for p in preds if p["score"] >= op]
        gts = gt_map[image_id]
        n_match = len(greedy_match(kept, gts, iou_threshold=0.5, class_aware=True))
        per_image.append(
            {
                "top": max((p["score"] for p in kept), default=0.95),  # cv_node's "normal" score
                "min": min((p["score"] for p in kept), default=0.95),
                "top_ok": _top_is_correct(kept, gts),
                "perfect": n_match == len(kept) == len(gts),
            }
        )

    def _gate_curve(signal: str, target: str) -> list[dict]:
        n_bad = sum(1 for r in per_image if not r[target])
        curve = []
        for t in grid:
            flagged = [r for r in per_image if r[signal] < t]
            tp = sum(1 for r in flagged if not r[target])
            precision = tp / len(flagged) if flagged else 0.0
            recall = tp / n_bad if n_bad else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            curve.append(
                {
                    "threshold": t,
                    "trigger_rate": round(len(flagged) / len(per_image), 4),
                    "flag_precision": round(precision, 4),
                    "flag_recall": round(recall, 4),
                    "flag_f1": round(f1, 4),
                }
            )
        return curve

    n_wrong = sum(1 for r in per_image if not r["top_ok"])
    gate = _gate_curve("top", "top_ok")
    gate_min = _gate_curve("min", "perfect")
    best_gate = max(gate, key=lambda g: (g["flag_f1"], -g["threshold"]))
    best_gate_min = max(gate_min, key=lambda g: (g["flag_f1"], -g["threshold"]))
    return {
        "stage": "threshold_sweep",
        "split": "val",
        "gold": str(gold_path),
        "n_images": len(rows),
        "weights": str(resolved),
        "top1_accuracy": round(1 - n_wrong / len(per_image), 4) if per_image else 0.0,
        "image_perfect_rate": round(sum(r["perfect"] for r in per_image) / len(per_image), 4),
        "best_detection_conf": best_det,
        "best_gate_top_score": best_gate,
        "best_gate_min_score": best_gate_min,
        "detection_curve": detection,
        "gate_curve_top_score": gate,
        "gate_curve_min_score": gate_min,
    }


def eval_self_correct_ab(
    gold_path: Path,
    weights: str | None,
    limit: int | None = 50,
) -> dict:
    """Same images, self-correction off vs on (BUILD.md A/B).

    - `off`: YOLO at YOLO_CONF, no self-correct.
    - `replace`: the pre-fusion node — template-diff blobs replace YOLO boxes
      whenever the gate fires (class-less, so class-aware F1 can only drop).
    - `verify`: gated template verification (`tools/fusion.py`), as the agent runs now.
    """
    from repair_agent.tools.fusion import verify_with_template

    resolved = resolve_weights(weights)
    if resolved is None:
        raise FileNotFoundError("YOLO weights not found; train first or pass --weights.")
    rows = _load_gold(gold_path, limit)
    gt_map: dict[str, list[dict]] = {}
    policies: dict[str, dict[str, list[dict]]] = {"off": {}, "replace": {}, "verify": {}}
    triggered = 0
    for row in rows:
        image_id = row["image_id"]
        test = decode_image(Path(row["image"]).read_bytes())
        template = decode_image(Path(row["template"]).read_bytes())
        gt_map[image_id] = [{"bbox": tuple(b["bbox"]), "cls": b["cls"]} for b in row["boxes"]]
        candidates = detect_yolo(test, str(resolved), conf=settings.YOLO_CANDIDATE_CONF)
        kept = [d for d in candidates if d["score"] >= settings.YOLO_CONF]
        min_conf = min((d["score"] for d in candidates), default=0.95)
        fire = min_conf < settings.CONFIDENCE_THRESHOLD
        triggered += int(fire)
        blobs = localize_defects(test, template) if fire else []
        policies["off"][image_id] = _as_xyxy_preds(kept)
        policies["replace"][image_id] = _as_xyxy_preds(blobs if fire and blobs else kept)
        fused = (
            verify_with_template(
                candidates,
                blobs,
                op_conf=settings.YOLO_CONF,
                recover_conf=settings.YOLO_CANDIDATE_CONF,
                keep_conf=settings.CONFIDENCE_THRESHOLD,
                min_coverage=settings.TEMPLATE_MIN_COVERAGE,
            )
            if fire
            else kept
        )
        policies["verify"][image_id] = _as_xyxy_preds(fused)

    results = {}
    for name, pred_map in policies.items():
        prf = detection_prf(pred_map, gt_map)
        perfect = sum(
            1
            for image_id, gts in gt_map.items()
            if len(greedy_match(pred_map[image_id], gts, class_aware=True))
            == len(pred_map[image_id])
            == len(gts)
        )
        results[name] = {
            "micro_f1": prf["micro_f1"],
            "macro_f1": prf["macro_f1"],
            "precision": prf["micro_precision"],
            "recall": prf["micro_recall"],
            "mAP50_class_aware": round(mean_average_precision(pred_map, gt_map)["mAP"], 4),
            "mAP50_class_agnostic": round(
                mean_average_precision(pred_map, gt_map, class_aware=False)["mAP"], 4
            ),
            "image_perfect_rate": round(perfect / len(gt_map), 4),
        }
    return {
        "stage": "self_correct_ab",
        "gold": str(gold_path),
        "n_images": len(rows),
        "trigger_rate": round(triggered / len(rows), 4) if rows else 0.0,
        "settings": {
            "YOLO_CONF": settings.YOLO_CONF,
            "YOLO_CANDIDATE_CONF": settings.YOLO_CANDIDATE_CONF,
            "CONFIDENCE_THRESHOLD": settings.CONFIDENCE_THRESHOLD,
            "TEMPLATE_MIN_COVERAGE": settings.TEMPLATE_MIN_COVERAGE,
        },
        "policies": results,
    }


def eval_diagnosis(gold_path: Path, limit: int | None = 50, judge: bool = True) -> dict:
    """End-to-end groundedness on a DeepPCB test subset (BUILD.md "Diagnosis").

    Runs the real LangGraph agent (CV → RAG → self-correct → DeepSeek) with the
    template attached, then scores each report deterministically and, if
    `judge`, with an LLM judge. Needs DEEPSEEK_API_KEY.
    """
    import asyncio

    from langchain_openai import ChatOpenAI

    from repair_agent.agent.graph import get_agent_sync
    from repair_agent.agent.state import initial_agent_state
    from repair_agent.eval.groundedness import judge_messages, parse_judge, score_diagnosis

    if not settings.DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not set; diagnosis eval needs the LLM.")

    rows = _load_gold(gold_path, limit)
    agent = get_agent_sync()
    judge_llm = ChatOpenAI(
        model=settings.DEEPSEEK_LLM_MODEL,
        temperature=0,
        openai_api_key=settings.DEEPSEEK_API_KEY,
        openai_api_base=settings.DEEPSEEK_BASE_URL,
    )
    per_image: list[dict] = []

    async def _run():
        for row in rows:
            state = initial_agent_state(
                Path(row["image"]).read_bytes(),
                template_image_bytes=Path(row["template"]).read_bytes(),
                image_path=row["image"],
            )
            final = await agent.ainvoke(state, config={"configurable": {"thread_id": row["image_id"]}})
            docs = final.get("rag_documents") or []
            diagnosis = final.get("diagnosis") or ""
            record = {
                "image_id": row["image_id"],
                "defect_type": final.get("defect_type"),
                "gold_classes": sorted({b["cls"] for b in row["boxes"]}),
                "correction_mode": final.get("correction_mode"),
                **score_diagnosis(diagnosis, docs, final.get("defect_type")),
            }
            if judge:
                header = (
                    f"class={final.get('defect_type')} confidence={final.get('defect_confidence')} "
                    f"correction={final.get('correction_mode')}"
                )
                reply = await judge_llm.ainvoke(judge_messages(diagnosis, docs, header))
                record["judge"] = parse_judge(str(reply.content))
            record["diagnosis"] = diagnosis
            per_image.append(record)
            print(
                f"diag {row['image_id']} cls={record['defect_type']} grounded={record['grounded']} "
                f"judge={record.get('judge', {}).get('verdict')}"
            )

    asyncio.run(_run())
    n = len(per_image) or 1
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    detail = RESULTS_DIR / "diagnosis_per_image.jsonl"
    detail.write_text("\n".join(json.dumps(r) for r in per_image) + "\n", encoding="utf-8")
    report = {
        "stage": "diagnosis",
        "n_images": len(per_image),
        "deterministic_grounded_rate": round(sum(r["grounded"] for r in per_image) / n, 4),
        "bad_citation_rate": round(sum(bool(r["bad_citations"]) for r in per_image) / n, 4),
        "unsupported_reference_rate": round(
            sum(bool(r["unsupported_references"]) for r in per_image) / n, 4
        ),
        "class_named_rate": round(sum(r["class_named"] for r in per_image) / n, 4),
        "detected_class_in_gold_rate": round(
            sum(r["defect_type"] in r["gold_classes"] for r in per_image) / n, 4
        ),
        "per_image": str(detail),
        "note": "Gate: groundedness >= 0.80. Judge is the same model family as the writer; spot-check by hand.",
    }
    if judge:
        report["judge_grounded_rate"] = round(
            sum(r.get("judge", {}).get("verdict") == "grounded" for r in per_image) / n, 4
        )
    return report


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["synthetic", "rag", "cv", "sweep", "ab", "transfer", "diagnosis", "ocr", "all"],
        default="synthetic",
    )
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument(
        "--rag-queries",
        type=Path,
        default=ROOT / "evals" / "rag_queries.jsonl",
    )
    parser.add_argument(
        "--reranker",
        default=None,
        help='RAG cross-encoder (default: settings.RAG_RERANKER; "" for dense-only)',
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
        "--preprocess",
        default="all",
        help="Transfer stage: 'all' or one of repair_agent.eval.transfer.PREPROCESS_VARIANTS",
    )
    parser.add_argument("--split", default="test", help="OCR stage: FPIC subset dev | test")
    parser.add_argument("--ocr-engine", default="tesseract", choices=["tesseract", "easyocr"])
    parser.add_argument("--no-judge", action="store_true", help="Diagnosis stage: skip the LLM judge")
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
                reports.append(eval_rag(args.rag_queries, reranker=args.reranker))
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
        elif stage == "sweep":
            gold = args.gold or (ROOT / "data" / "processed" / "deeppcb" / "gold_val.jsonl")
            if "test" in gold.name:
                print("refusing to sweep thresholds on the test split")
                return 1
            weights_arg = str(args.weights) if args.weights else (settings.YOLO_WEIGHTS or None)
            report = eval_threshold_sweep(gold, weights_arg, limit=args.limit)
            _write_report("threshold_sweep", report)
            reports.append({k: v for k, v in report.items() if not k.endswith("_curve")})
        elif stage == "ab":
            gold = args.gold or (ROOT / "data" / "processed" / "deeppcb" / "gold_test.jsonl")
            weights_arg = str(args.weights) if args.weights else (settings.YOLO_WEIGHTS or None)
            limit = 50 if args.limit is None else (None if args.limit <= 0 else args.limit)
            report = eval_self_correct_ab(gold, weights_arg, limit=limit)
            _write_report(f"self_correct_ab_{gold.stem}_{report['n_images']}", report)
            reports.append(report)
        elif stage == "transfer":
            from repair_agent.eval.transfer import PREPROCESS_VARIANTS, eval_transfer_variants

            gold = args.gold or (ROOT / "data" / "processed" / "pku" / "gold_holdout.jsonl")
            weights_arg = str(args.weights) if args.weights else (settings.YOLO_WEIGHTS or None)
            resolved = resolve_weights(weights_arg)
            if not gold.exists() or resolved is None:
                print(f"transfer skipped: need {gold} (prepare_pku.py) and YOLO weights")
                reports.append({"stage": "transfer", "skipped": True, "gold": str(gold)})
            else:
                variants = PREPROCESS_VARIANTS if args.preprocess == "all" else (args.preprocess,)
                report = eval_transfer_variants(
                    gold, str(resolved), conf=args.conf, variants=tuple(variants), limit=args.limit
                )
                _write_report("transfer", report)
                reports.append(report)
        elif stage == "diagnosis":
            gold = args.gold or (ROOT / "data" / "processed" / "deeppcb" / "gold_test.jsonl")
            limit = 50 if args.limit is None else (None if args.limit <= 0 else args.limit)
            try:
                report = eval_diagnosis(gold, limit=limit, judge=not args.no_judge)
                _write_report("diagnosis", report)
                reports.append(report)
            except RuntimeError as exc:
                print(f"diagnosis eval skipped: {exc}")
                reports.append({"stage": "diagnosis", "skipped": True, "error": str(exc)})
        elif stage == "ocr":
            gold = args.gold or (ROOT / "data" / "processed" / "fpic" / "gold_ocr.jsonl")
            if not gold.exists():
                print("OCR eval needs FPIC gold; see docs/BUILD.md Phase D (registration required)")
                reports.append(
                    {"stage": "ocr", "skipped": True, "reason": "no FPIC gold; run scripts/prepare_fpic.py"}
                )
            else:
                report = eval_ocr(gold, split=args.split, engine=args.ocr_engine, limit=args.limit)
                _write_report("ocr", report)
                reports.append(report)
        else:
            raise ValueError(f"Unknown eval stage: {stage}")

    _write_report("latest", {"reports": reports})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
