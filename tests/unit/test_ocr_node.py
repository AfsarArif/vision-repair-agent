"""Unit tests for the OCR node and tools."""

import pytest

from repair_agent.agent.nodes.ocr_node import ocr_node
from repair_agent.tools.ocr_tools import (
    preprocess_for_ocr,
    extract_text,
    find_serial_number,
    estimate_ocr_confidence,
    decode_and_preprocess,
)


class TestOCRTools:
    def test_find_serial_number_standard(self):
        result = find_serial_number("Device Model X SN-ABC12345 Rev 2")
        assert result is not None
        assert "ABC12345" in result or "SN-ABC12345" in result

    def test_find_serial_number_slash_format(self):
        result = find_serial_number("S/N: XYZ-789012 Rev A")
        assert result is not None
        assert "XYZ-789012" in result or "S/N" in result

    def test_find_serial_number_none(self):
        result = find_serial_number("No serial number here")
        assert result is None

    def test_find_serial_number_dash_format(self):
        result = find_serial_number("Part 1234-ABCD Version 3")
        assert result is not None

    def test_find_serial_number_empty_string(self):
        result = find_serial_number("")
        assert result is None

    def test_decode_and_preprocess_valid(self, serial_number_image_bytes):
        preprocessed = decode_and_preprocess(serial_number_image_bytes)
        assert preprocessed is not None
        assert len(preprocessed.shape) == 2  # Should be grayscale
        # Check it's binary
        unique_vals = set(preprocessed.flatten().tolist())
        assert unique_vals.issubset({0, 255})

    def test_decode_and_preprocess_invalid(self):
        with pytest.raises(ValueError, match="Failed to decode"):
            decode_and_preprocess(b"not an image")

    def test_estimate_ocr_confidence(self, serial_number_image_bytes):
        preprocessed = decode_and_preprocess(serial_number_image_bytes)
        conf = estimate_ocr_confidence(preprocessed)
        assert 0.0 <= conf <= 1.0


class TestOCRNode:
    @pytest.mark.asyncio
    async def test_ocr_node_increments_attempts(self, base_state):
        # Use a state with cropped image bytes if available
        if base_state.get("cropped_image_bytes"):
            result = await ocr_node(base_state)
            assert result["correction_attempts"] == 1
            assert result["self_correction_triggered"] is True

    @pytest.mark.asyncio
    async def test_ocr_node_sets_ocr_confidence(self, base_state):
        if base_state.get("cropped_image_bytes"):
            result = await ocr_node(base_state)
            assert "ocr_confidence" in result
            assert result["ocr_confidence"] is not None
            assert 0.0 <= result["ocr_confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_ocr_node_falls_back_to_full_image(self, burn_mark_image_bytes):
        """OCR node should use full image when no cropped image is available."""
        state = {
            "image_bytes": burn_mark_image_bytes,
            "cropped_image_bytes": None,
            "correction_attempts": 0,
        }
        result = await ocr_node(state)
        assert result["correction_attempts"] == 1
        assert "serial_number" in result
        assert "ocr_confidence" in result
