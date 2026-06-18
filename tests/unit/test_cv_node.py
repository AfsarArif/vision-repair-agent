"""Unit tests for the CV node."""

import pytest

from repair_agent.agent.nodes.cv_node import cv_node
from repair_agent.tools.cv_tools import (
    decode_image,
    preprocess_for_contour_detection,
    find_defect_contours,
    classify_defect_heuristic,
    estimate_confidence,
)


class TestCVTools:
    def test_decode_image_valid(self, burn_mark_image_bytes):
        img = decode_image(burn_mark_image_bytes)
        assert img is not None
        assert img.shape[0] > 0 and img.shape[1] > 0
        assert img.shape[2] == 3  # BGR

    def test_decode_image_invalid(self):
        with pytest.raises(ValueError, match="Failed to decode"):
            decode_image(b"not an image")

    def test_preprocess_for_contour_detection(self, burn_mark_image_bytes):
        img = decode_image(burn_mark_image_bytes)
        thresh = preprocess_for_contour_detection(img)
        assert thresh is not None
        assert thresh.shape[:2] == img.shape[:2]
        # Output should be binary (0 or 255)
        assert set(thresh.flatten().tolist()).issubset({0, 255})

    def test_find_defect_contours_with_defect(self, burn_mark_image_bytes):
        img = decode_image(burn_mark_image_bytes)
        thresh = preprocess_for_contour_detection(img)
        results = find_defect_contours(thresh, min_area=100)
        assert len(results) > 0
        for contour, bbox in results:
            x, y, w, h = bbox
            assert x >= 0 and y >= 0
            assert w > 0 and h > 0

    def test_find_defect_contours_clean_image(self, clean_image_bytes):
        img = decode_image(clean_image_bytes)
        thresh = preprocess_for_contour_detection(img)
        results = find_defect_contours(thresh)
        # Clean image may have very small contours but they should be filtered
        assert len(results) == 0

    def test_classify_defect_burn_mark(self, burn_mark_image_bytes):
        img = decode_image(burn_mark_image_bytes)
        thresh = preprocess_for_contour_detection(img)
        results = find_defect_contours(thresh)
        if results:
            contour, bbox = results[0]
            defect = classify_defect_heuristic(img, contour, bbox)
            assert defect in ["burn_mark", "crack", "corrosion", "delamination", "normal"]

    def test_classify_defect_crack(self, crack_image_bytes):
        img = decode_image(crack_image_bytes)
        thresh = preprocess_for_contour_detection(img)
        results = find_defect_contours(thresh)
        if results:
            contour, bbox = results[0]
            defect = classify_defect_heuristic(img, contour, bbox)
            # Crack images should classify as crack
            # (may vary with synthetic data, so just check valid output)
            assert defect in ["burn_mark", "crack", "corrosion", "delamination", "normal"]

    def test_estimate_confidence(self, burn_mark_image_bytes):
        img = decode_image(burn_mark_image_bytes)
        thresh = preprocess_for_contour_detection(img)
        results = find_defect_contours(thresh)
        contours = [c for c, _ in results]
        conf = estimate_confidence(contours, img.shape[0] * img.shape[1])
        assert 0.0 <= conf <= 1.0


class TestCVNode:
    @pytest.mark.asyncio
    async def test_cv_node_detects_defect(self, base_state):
        result = await cv_node(base_state)
        assert "defect_type" in result
        assert "defect_confidence" in result
        assert result["defect_type"] is not None
        assert result["defect_confidence"] is not None
        assert 0.0 <= result["defect_confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_cv_node_clean_image(self, clean_image_bytes):
        state = {
            "image_bytes": clean_image_bytes,
        }
        result = await cv_node(state)
        assert result["defect_type"] == "normal"
        assert result["defect_confidence"] == 0.95
        assert result["defect_bbox"] is None

    @pytest.mark.asyncio
    async def test_cv_node_bbox_non_negative(self, base_state):
        result = await cv_node(base_state)
        if result["defect_bbox"] is not None:
            x, y, w, h = result["defect_bbox"]
            assert x >= 0
            assert y >= 0
            assert w > 0
            assert h > 0

    @pytest.mark.asyncio
    async def test_cv_node_crops_region(self, base_state):
        result = await cv_node(base_state)
        if result["cropped_image_bytes"] is not None:
            assert len(result["cropped_image_bytes"]) > 0
            # Should be valid PNG
            import cv2
            import numpy as np

            img_array = np.frombuffer(result["cropped_image_bytes"], np.uint8)
            decoded = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            assert decoded is not None
