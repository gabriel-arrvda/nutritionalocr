import pytest
from PIL import Image
import io
import numpy as np
import hashlib
from unittest.mock import Mock, patch
from src.utils.data_collection import (
    validate_image,
    download_image,
    apply_augmentation,
    translate_nutrients,
)


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


def test_apply_augmentation_combines_one_or_two_techniques_and_metadata():
    """Each variation should report 1-2 techniques and include combined cases."""
    original_img = Image.new('RGB', (320, 320), color='yellow')

    results = apply_augmentation(original_img, num_variations=20, seed=7)

    technique_counts = []
    for result in results:
        augmentation_type = result['augmentation_type']
        assert isinstance(augmentation_type, str)
        applied = [part for part in augmentation_type.split('+') if part and part != 'fallback_rotation']
        technique_counts.append(len(applied))
        assert len(applied) in {1, 2}

    assert 2 in technique_counts


def test_apply_augmentation_seed_reproducibility():
    """Same seed should produce consistent augmentation metadata and image content."""
    original_img = Image.new('RGB', (300, 300), color='purple')

    first = apply_augmentation(original_img, num_variations=5, seed=99)
    second = apply_augmentation(original_img, num_variations=5, seed=99)

    first_types = [item['augmentation_type'] for item in first]
    second_types = [item['augmentation_type'] for item in second]
    assert first_types == second_types

    first_hashes = [
        hashlib.sha256(item['image'].tobytes()).hexdigest()
        for item in first
    ]
    second_hashes = [
        hashlib.sha256(item['image'].tobytes()).hexdigest()
        for item in second
    ]
    assert first_hashes == second_hashes


def test_apply_augmentation_fallback_metadata_and_non_rgb_fillcolor():
    """Fallback should be reflected in metadata and work for non-RGB modes."""
    original_img = Image.new('L', (128, 128), color=128)

    with patch('src.utils.data_collection.ImageChops.difference') as difference_mock:
        difference_result = Mock()
        difference_result.getbbox.return_value = None
        difference_mock.return_value = difference_result

        results = apply_augmentation(original_img, num_variations=1, seed=1)

    assert len(results) == 1
    assert 'fallback_rotation' in results[0]['augmentation_type']


def test_translate_nutrients_english_to_portuguese():
    """Should translate English nutrient names to Portuguese"""
    nutrients = {
        'calories': 150,
        'protein': 10.5,
        'carbohydrates': 20.0,
        'total_fat': 5.0
    }

    with patch('googletrans.Translator.translate') as mock_translate:
        mock_translate.side_effect = lambda text, src, dest: Mock(text=f'{text}_pt')

        result = translate_nutrients(nutrients, source_lang='en', target_lang='pt')

        assert result['success'] is True
        assert 'calories_pt' in result['translated']


def test_translate_nutrients_preserves_values():
    """Should preserve numeric values unchanged"""
    nutrients = {
        'protein': 12.5,
        'sodium': 300
    }

    with patch('googletrans.Translator.translate') as mock_translate:
        mock_translate.side_effect = lambda text, src, dest: Mock(text=f'{text}_pt')

        result = translate_nutrients(nutrients, source_lang='en', target_lang='pt')

        translated = result['translated']
        assert translated['protein_pt'] == 12.5
        assert translated['sodium_pt'] == 300


def test_translate_nutrients_api_failure():
    """Should handle translation API failures gracefully"""
    nutrients = {'calories': 100}

    with patch('googletrans.Translator.translate', side_effect=Exception('API Error')):
        result = translate_nutrients(nutrients, source_lang='en', target_lang='pt')

        assert result['success'] is False
        assert result['translated'] == nutrients
        assert 'API Error' in result['error']
