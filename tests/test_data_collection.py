import pytest
from PIL import Image
import io
from src.utils.data_collection import validate_image


def test_validate_image_valid_jpeg():
    """Valid JPEG image should pass validation"""
    img = Image.new('RGB', (300, 300), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    result = validate_image(img_bytes, min_width=200, min_height=200)
    assert result['valid'] is True
    assert result['width'] == 300
    assert result['height'] == 300
    assert result['format'] == 'JPEG'


def test_validate_image_too_small():
    """Image below minimum resolution should fail"""
    img = Image.new('RGB', (150, 150), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    result = validate_image(img_bytes, min_width=200, min_height=200)
    assert result['valid'] is False
    assert 'too small' in result['reason'].lower()


def test_validate_image_corrupted():
    """Corrupted image should fail gracefully"""
    corrupted_bytes = io.BytesIO(b'not an image')
    
    result = validate_image(corrupted_bytes, min_width=200, min_height=200)
    assert result['valid'] is False
    assert 'corrupted' in result['reason'].lower()
