import pytest
from PIL import Image
import io
import numpy as np
from unittest.mock import Mock, patch
from src.utils.data_collection import validate_image, download_image, apply_augmentation


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


def test_download_image_success():
    """Successful download should return image bytes"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = b'fake image data'
    
    with patch('requests.get', return_value=mock_response):
        result = download_image('http://example.com/image.jpg', timeout=10)
        
        assert result['success'] is True
        assert result['data'] == b'fake image data'
        assert result['url'] == 'http://example.com/image.jpg'


def test_download_image_404():
    """404 error should return failure"""
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = Exception('404 Not Found')
    
    with patch('requests.get', return_value=mock_response):
        result = download_image('http://example.com/missing.jpg', timeout=10)
        
        assert result['success'] is False
        assert '404' in result['error']


def test_download_image_timeout():
    """Timeout should return failure"""
    import requests
    with patch('requests.get', side_effect=requests.exceptions.Timeout('Request timed out')):
        result = download_image('http://example.com/slow.jpg', timeout=5)
        
        assert result['success'] is False
        assert 'timeout' in result['error'].lower()


def test_apply_augmentation_returns_correct_count():
    """Should return requested number of augmented images"""
    original_img = Image.new('RGB', (400, 400), color='green')
    
    results = apply_augmentation(original_img, num_variations=3, seed=42)
    
    assert len(results) == 3
    for result in results:
        assert 'image' in result
        assert 'augmentation_type' in result
        assert isinstance(result['image'], Image.Image)


def test_apply_augmentation_images_are_different():
    """Augmented images should differ from original"""
    original_img = Image.new('RGB', (400, 400), color='red')
    original_array = np.array(original_img)
    
    results = apply_augmentation(original_img, num_variations=2, seed=42)
    
    for result in results:
        augmented_array = np.array(result['image'])
        # At least some pixels should be different
        assert not np.array_equal(original_array, augmented_array)


def test_apply_augmentation_preserves_dimensions():
    """Augmented images should have similar dimensions (±10%)"""
    original_img = Image.new('RGB', (500, 600), color='blue')
    
    results = apply_augmentation(original_img, num_variations=3, seed=42)
    
    for result in results:
        width, height = result['image'].size
        # Allow ±10% variation from crop/resize
        assert 450 <= width <= 550
        assert 540 <= height <= 660
