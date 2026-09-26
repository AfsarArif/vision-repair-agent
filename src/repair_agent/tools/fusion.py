"""Template-verified detection: fuse class-aware YOLO boxes with class-agnostic absdiff blobs.

The template diff cannot name a class, so it never adds boxes on its own. It
arbitrates YOLO's uncertain band instead:

- a candidate below the operating point is **recovered** when a diff blob covers it;
- a kept box below `keep_conf` is **dropped** when no diff blob supports it.

Boxes are xywh, matching `detect_yolo` and `localize_defects`.
"""

from __future__ import annotations


def _coverage(box: tuple, blob: tuple) -> float:
    """Fraction of `box` area covered by `blob` (both xywh)."""
    bx, by, bw, bh = box
    ox, oy, ow, oh = blob
    iw = max(0, min(bx + bw, ox + ow) - max(bx, ox))
    ih = max(0, min(by + bh, oy + oh) - max(by, oy))
    area = bw * bh
    return (iw * ih) / area if area > 0 else 0.0


def supported(box: tuple, blobs: list[dict], min_coverage: float) -> bool:
    return any(_coverage(tuple(box), tuple(b["bbox"])) >= min_coverage for b in blobs)


def verify_with_template(
    candidates: list[dict],
    diff_blobs: list[dict],
    op_conf: float,
    recover_conf: float = 0.1,
    keep_conf: float = 0.8,
    min_coverage: float = 0.1,
) -> list[dict]:
    """Return fused detections sorted by score.

    `candidates` are YOLO detections at a low confidence (>= `recover_conf`).
    Boxes scoring >= `keep_conf` are kept without diff support.
    """
    fused: list[dict] = []
    for det in candidates:
        score = float(det.get("score") or 0.0)
        if score < recover_conf:
            continue
        if score >= keep_conf:
            fused.append(det)
        elif supported(det["bbox"], diff_blobs, min_coverage):
            fused.append({**det, "template_verified": True})
        elif score >= op_conf and not diff_blobs:
            # No diff signal at all (e.g. misaligned template): do not drop.
            fused.append(det)
    fused.sort(key=lambda d: float(d.get("score") or 0.0), reverse=True)
    return fused
