"""Shared test fixtures for the repair agent test suite."""

import io
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from repair_agent.agent.state import AgentState


def create_test_image(
    width: int = 640,
    height: int = 480,
    color: tuple[int, int, int] = (128, 128, 128),
    add_defect: bool = False,
    defect_type: str = "burn_mark",
) -> bytes:
    """Create a synthetic test image, optionally with a simulated defect.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        color: Background BGR color.
        add_defect: If True, draw a simulated defect region.
        defect_type: Type of defect to simulate ('burn_mark', 'crack', 'corrosion').

    Returns:
        PNG-encoded image bytes.
    """
    img = np.full((height, width, 3), color, dtype=np.uint8)

    if add_defect:
        if defect_type == "burn_mark":
            # Dark brown/black region
            cv2.rectangle(img, (200, 150), (440, 330), (15, 10, 5), -1)
            cv2.circle(img, (320, 240), 60, (5, 5, 0), -1)
        elif defect_type == "crack":
            # Thin elongated line
            cv2.line(img, (100, 240), (540, 240), (0, 0, 0), 3)
            cv2.line(img, (100, 242), (540, 242), (0, 0, 0), 2)
        elif defect_type == "corrosion":
            # Greenish patch
            cv2.ellipse(img, (320, 240), (120, 80), 0, 0, 360, (80, 140, 60), -1)
            for _ in range(20):
                x, y = np.random.randint(200, 440), np.random.randint(160, 320)
                cv2.circle(img, (x, y), 8, (60, 120, 40), -1)

    _, encoded = cv2.imencode(".png", img)
    return encoded.tobytes()


def create_text_image(
    text: str = "SN-ABC12345",
    width: int = 400,
    height: int = 100,
) -> bytes:
    """Create a synthetic image with text for OCR testing.

    Args:
        text: Text to render on the image.
        width: Image width.
        height: Image height.

    Returns:
        PNG-encoded image bytes.
    """
    img = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)

    # Use PIL for text rendering since OpenCV has limited font support
    from PIL import Image, ImageDraw, ImageFont

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    font_size = 32
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    # Center text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    draw.text((x, y), text, fill=(0, 0, 0), font=font)

    # Convert back to OpenCV format
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode(".png", img)
    return encoded.tobytes()


@pytest.fixture
def clean_image_bytes() -> bytes:
    """Return a clean (defect-free) test image."""
    return create_test_image(add_defect=False)


@pytest.fixture
def burn_mark_image_bytes() -> bytes:
    """Return an image with a simulated burn mark."""
    return create_test_image(add_defect=True, defect_type="burn_mark")


@pytest.fixture
def crack_image_bytes() -> bytes:
    """Return an image with a simulated crack."""
    return create_test_image(add_defect=True, defect_type="crack")


@pytest.fixture
def corrosion_image_bytes() -> bytes:
    """Return an image with simulated corrosion."""
    return create_test_image(add_defect=True, defect_type="corrosion")


@pytest.fixture
def serial_number_image_bytes() -> bytes:
    """Return an image with printed serial number text."""
    return create_text_image("SN-ABC12345")


@pytest.fixture
def base_state(burn_mark_image_bytes) -> AgentState:
    """Return a minimal AgentState for testing."""
    return {
        "image_bytes": burn_mark_image_bytes,
        "image_path": None,
        "defect_type": None,
        "defect_confidence": None,
        "defect_bbox": None,
        "cropped_image_bytes": None,
        "serial_number": None,
        "ocr_confidence": None,
        "rag_documents": None,
        "rag_query": None,
        "diagnosis": None,
        "correction_attempts": 0,
        "self_correction_triggered": False,
        "messages": [],
    }
