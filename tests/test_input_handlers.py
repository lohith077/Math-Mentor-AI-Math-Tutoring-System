import pytest
from PIL import Image, ImageDraw, ImageFont
import io
from input_handlers.ocr_handler import extract_text_from_image
from input_handlers.asr_handler import normalize_math_phrases, transcribe_audio
from input_handlers.text_handler import normalize_text

def create_synthetic_image(text):
    image = Image.new("RGB", (200, 50), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), text, fill="black")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def test_ocr_extraction(monkeypatch):
    # Basic smoke test for OCR handler structure
    # Since accurate OCR involves models/system deps, we mock or test basics
    
    # Mock pytesseract to avoid system dependency errors in test environments
    import pytesseract
    monkeypatch.setattr(pytesseract, "image_to_data", lambda img, output_type: {"conf": ["85", "90", "95"]})
    monkeypatch.setattr(pytesseract, "image_to_string", lambda img: "2x + 3 = 7")
    
    img_bytes = create_synthetic_image("2x + 3 = 7")
    result = extract_text_from_image(img_bytes)
    assert isinstance(result, dict)
    assert "text" in result
    assert "confidence" in result
    assert "raw_data" in result

def test_text_normalization():
    assert normalize_text("2  x ^ 3") == "2x**3" 
    assert normalize_text("2x^3") == "2x**3"

def test_math_phrase_normalization():
    assert normalize_math_phrases("square root of 4 equals 2") == "sqrt( 4 = 2"
    assert normalize_math_phrases("3 times 2") == "3 * 2"
