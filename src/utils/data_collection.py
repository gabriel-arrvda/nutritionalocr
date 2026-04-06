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

    Each variation applies 1-2 randomly selected augmentation techniques.
    Techniques include rotation, brightness/contrast adjustment,
    gaussian blur, and crop/resize.
    """
    rng = random.Random(seed)
    augmentations = ['rotation', 'brightness_contrast', 'gaussian_blur', 'crop_resize']
    width, height = image.size

    def apply_single_augmentation(augmented_image: Image.Image, augmentation_name: str) -> Image.Image:
        if augmentation_name == 'rotation':
            angle = rng.uniform(-20, 20)
            return augmented_image.rotate(angle, resample=Image.Resampling.BICUBIC)
        if augmentation_name == 'brightness_contrast':
            brightness = rng.uniform(0.7, 1.3)
            contrast = rng.uniform(0.7, 1.3)
            adjusted = ImageEnhance.Brightness(augmented_image).enhance(brightness)
            return ImageEnhance.Contrast(adjusted).enhance(contrast)
        if augmentation_name == 'gaussian_blur':
            blur_radius = rng.uniform(0.5, 2.0)
            return augmented_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        if augmentation_name == 'crop_resize':
            crop_ratio = rng.uniform(0.8, 0.95)
            crop_w = max(1, int(width * crop_ratio))
            crop_h = max(1, int(height * crop_ratio))
            max_left = max(0, width - crop_w)
            max_top = max(0, height - crop_h)
            left = rng.randint(0, max_left) if max_left > 0 else 0
            top = rng.randint(0, max_top) if max_top > 0 else 0
            cropped = augmented_image.crop((left, top, left + crop_w, top + crop_h))
            return cropped.resize((width, height), resample=Image.Resampling.BICUBIC)
        return augmented_image

    def get_mode_aware_fillcolor(target_image: Image.Image) -> Any:
        return Image.new(target_image.mode, (1, 1), 0).getpixel((0, 0))

    results: List[Dict[str, Any]] = []
    for _ in range(num_variations):
        augmented = image.copy()
        applied_techniques: List[str] = []

        num_techniques = rng.randint(1, 2)
        selected_techniques = rng.sample(augmentations, k=num_techniques)
        for technique in selected_techniques:
            augmented = apply_single_augmentation(augmented, technique)
            applied_techniques.append(technique)

        fallback_used = ImageChops.difference(image, augmented).getbbox() is None
        if fallback_used:
            fillcolor = get_mode_aware_fillcolor(augmented)
            augmented = augmented.rotate(15, resample=Image.Resampling.BICUBIC, fillcolor=fillcolor)
            applied_techniques.append('fallback_rotation')

        results.append({
            'image': augmented,
            'augmentation_type': '+'.join(applied_techniques)
        })

    return results
