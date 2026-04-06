"""Data collection utility functions for nutrition label OCR"""

from typing import Dict, Any, BinaryIO, List, Optional
from PIL import Image, ImageEnhance, ImageFilter, ImageChops
import logging
import requests
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_image(
    image_bytes: BinaryIO,
    min_width: int = 200,
    min_height: int = 200
) -> Dict[str, Any]:
    """
    Validate image format, size and integrity.
    
    Args:
        image_bytes: Binary image data
        min_width: Minimum acceptable width in pixels
        min_height: Minimum acceptable height in pixels
        
    Returns:
        Dict with 'valid' (bool), 'width', 'height', 'format', and 'reason' keys
    """
    try:
        img = Image.open(image_bytes)
        width, height = img.size
        img_format = img.format
        
        if width < min_width or height < min_height:
            return {
                'valid': False,
                'width': width,
                'height': height,
                'format': img_format,
                'reason': f'Image too small: {width}x{height} (minimum: {min_width}x{min_height})'
            }
        
        if img_format not in ['JPEG', 'PNG', 'JPG']:
            return {
                'valid': False,
                'width': width,
                'height': height,
                'format': img_format,
                'reason': f'Unsupported format: {img_format}'
            }
        
        return {
            'valid': True,
            'width': width,
            'height': height,
            'format': img_format,
            'reason': 'Valid'
        }
        
    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        return {
            'valid': False,
            'width': 0,
            'height': 0,
            'format': None,
            'reason': f'Corrupted or invalid image: {str(e)}'
        }


def download_image(url: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Download image from URL.
    
    Args:
        url: Image URL
        timeout: Request timeout in seconds
        
    Returns:
        Dict with 'success' (bool), 'data' (bytes), 'url', and 'error' keys
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        return {
            'success': True,
            'data': response.content,
            'url': url,
            'error': None
        }
        
    except requests.exceptions.Timeout:
        logger.error(f"Download timeout for {url}")
        return {
            'success': False,
            'data': None,
            'url': url,
            'error': f'Request timeout after {timeout}s'
        }
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error for {url}: {e}")
        return {
            'success': False,
            'data': None,
            'url': url,
            'error': str(e)
        }
        
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        return {
            'success': False,
            'data': None,
            'url': url,
            'error': str(e)
        }


def apply_augmentation(
    image: Image.Image,
    num_variations: int = 3,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Apply random image augmentations and return generated variations.

    Augmentations include rotation, brightness/contrast adjustment,
    gaussian blur, and crop/resize.
    """
    rng = random.Random(seed)
    augmentations = ['rotation', 'brightness_contrast', 'gaussian_blur', 'crop_resize']
    width, height = image.size

    results: List[Dict[str, Any]] = []
    for _ in range(num_variations):
        augmentation_type = rng.choice(augmentations)
        augmented = image.copy()

        if augmentation_type == 'rotation':
            angle = rng.uniform(-20, 20)
            augmented = augmented.rotate(angle, resample=Image.Resampling.BICUBIC)
        elif augmentation_type == 'brightness_contrast':
            brightness = rng.uniform(0.7, 1.3)
            contrast = rng.uniform(0.7, 1.3)
            augmented = ImageEnhance.Brightness(augmented).enhance(brightness)
            augmented = ImageEnhance.Contrast(augmented).enhance(contrast)
        elif augmentation_type == 'gaussian_blur':
            blur_radius = rng.uniform(0.5, 2.0)
            augmented = augmented.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        elif augmentation_type == 'crop_resize':
            crop_ratio = rng.uniform(0.8, 0.95)
            crop_w = max(1, int(width * crop_ratio))
            crop_h = max(1, int(height * crop_ratio))
            max_left = max(0, width - crop_w)
            max_top = max(0, height - crop_h)
            left = rng.randint(0, max_left) if max_left > 0 else 0
            top = rng.randint(0, max_top) if max_top > 0 else 0
            augmented = augmented.crop((left, top, left + crop_w, top + crop_h))
            augmented = augmented.resize((width, height), resample=Image.Resampling.BICUBIC)

        if ImageChops.difference(image, augmented).getbbox() is None:
            augmented = augmented.rotate(15, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))

        results.append({
            'image': augmented,
            'augmentation_type': augmentation_type
        })

    return results
