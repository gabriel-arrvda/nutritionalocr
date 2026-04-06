"""Data collection utility functions for nutrition label OCR"""

from typing import Dict, Any, BinaryIO
from PIL import Image
import logging
import requests

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
