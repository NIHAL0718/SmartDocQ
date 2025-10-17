"""Utility functions for image processing operations."""

import os
import io
import cv2
import numpy as np
from typing import Tuple, List, Optional, Union, BinaryIO
from PIL import Image, ImageEnhance, ImageFilter, UnidentifiedImageError

from ..core.logging import get_logger
from ..core.errors import InvalidDocumentFormatError

# Initialize logger
logger = get_logger("image_utils")

# Define allowed image extensions
ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
}


def is_valid_image_extension(extension: str) -> bool:
    """Check if a file extension is valid for images.
    
    Args:
        extension (str): File extension (with dot)
        
    Returns:
        bool: True if extension is valid, False otherwise
    """
    return extension.lower() in ALLOWED_IMAGE_EXTENSIONS


def get_image_mime_type(extension: str) -> str:
    """Get the MIME type for an image file extension.
    
    Args:
        extension (str): File extension (with dot)
        
    Returns:
        str: MIME type
    """
    return ALLOWED_IMAGE_EXTENSIONS.get(extension.lower(), "application/octet-stream")


def open_image(image_path: str) -> Image.Image:
    """Open an image file.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        Image.Image: PIL Image object
        
    Raises:
        FileNotFoundError: If the image file does not exist
        InvalidDocumentFormatError: If the file is not a valid image
    """
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Check file extension
    _, ext = os.path.splitext(image_path)
    if not is_valid_image_extension(ext):
        raise InvalidDocumentFormatError(ext)
    
    try:
        # Open the image
        image = Image.open(image_path)
        return image
    except UnidentifiedImageError:
        raise InvalidDocumentFormatError(ext)
    except Exception as e:
        logger.error(f"Error opening image: {image_path} - {e}")
        raise


def open_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Open an image from bytes.
    
    Args:
        image_bytes (bytes): Image bytes
        
    Returns:
        Image.Image: PIL Image object
        
    Raises:
        InvalidDocumentFormatError: If the bytes are not a valid image
    """
    try:
        # Open the image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except UnidentifiedImageError:
        raise InvalidDocumentFormatError("unknown")
    except Exception as e:
        logger.error(f"Error opening image from bytes: {e}")
        raise


def save_image(image: Image.Image, output_path: str, format: Optional[str] = None, quality: int = 95) -> str:
    """Save an image to a file.
    
    Args:
        image (Image.Image): PIL Image object
        output_path (str): Path to save the image to
        format (Optional[str]): Image format (e.g., 'JPEG', 'PNG')
        quality (int): Image quality (0-100, JPEG only)
        
    Returns:
        str: Path to the saved image
    """
    try:
        # Determine format from output path if not specified
        if format is None:
            _, ext = os.path.splitext(output_path)
            format = ext[1:].upper()
            if format == 'JPG':
                format = 'JPEG'
        
        # Save the image
        image.save(output_path, format=format, quality=quality)
        logger.info(f"Image saved: {output_path}")
        
        return output_path
    except Exception as e:
        logger.error(f"Error saving image: {output_path} - {e}")
        raise


def image_to_bytes(image: Image.Image, format: str = 'JPEG', quality: int = 95) -> bytes:
    """Convert an image to bytes.
    
    Args:
        image (Image.Image): PIL Image object
        format (str): Image format (e.g., 'JPEG', 'PNG')
        quality (int): Image quality (0-100, JPEG only)
        
    Returns:
        bytes: Image bytes
    """
    try:
        # Convert image to bytes
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=quality)
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Error converting image to bytes: {e}")
        raise


def resize_image(image: Image.Image, width: Optional[int] = None, height: Optional[int] = None, 
                maintain_aspect_ratio: bool = True) -> Image.Image:
    """Resize an image.
    
    Args:
        image (Image.Image): PIL Image object
        width (Optional[int]): Target width
        height (Optional[int]): Target height
        maintain_aspect_ratio (bool): Whether to maintain aspect ratio
        
    Returns:
        Image.Image: Resized image
    """
    # Get original dimensions
    orig_width, orig_height = image.size
    
    # Calculate new dimensions
    if width is None and height is None:
        return image
    elif width is None:
        if maintain_aspect_ratio:
            width = int(orig_width * (height / orig_height))
        else:
            width = orig_width
    elif height is None:
        if maintain_aspect_ratio:
            height = int(orig_height * (width / orig_width))
        else:
            height = orig_height
    elif maintain_aspect_ratio:
        # Calculate aspect ratios
        width_ratio = width / orig_width
        height_ratio = height / orig_height
        
        # Use the smaller ratio to ensure the image fits within the target dimensions
        if width_ratio < height_ratio:
            height = int(orig_height * width_ratio)
        else:
            width = int(orig_width * height_ratio)
    
    # Resize the image
    resized_image = image.resize((width, height), Image.LANCZOS)
    
    return resized_image


def crop_image(image: Image.Image, left: int, top: int, right: int, bottom: int) -> Image.Image:
    """Crop an image.
    
    Args:
        image (Image.Image): PIL Image object
        left (int): Left coordinate
        top (int): Top coordinate
        right (int): Right coordinate
        bottom (int): Bottom coordinate
        
    Returns:
        Image.Image: Cropped image
    """
    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))
    
    return cropped_image


def rotate_image(image: Image.Image, angle: float, expand: bool = True) -> Image.Image:
    """Rotate an image.
    
    Args:
        image (Image.Image): PIL Image object
        angle (float): Rotation angle in degrees
        expand (bool): Whether to expand the image to fit the rotated content
        
    Returns:
        Image.Image: Rotated image
    """
    # Rotate the image
    rotated_image = image.rotate(angle, expand=expand, resample=Image.BICUBIC)
    
    return rotated_image


def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
    """Adjust image brightness.
    
    Args:
        image (Image.Image): PIL Image object
        factor (float): Brightness factor (0.0-2.0, 1.0 is original)
        
    Returns:
        Image.Image: Adjusted image
    """
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(image)
    adjusted_image = enhancer.enhance(factor)
    
    return adjusted_image


def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    """Adjust image contrast.
    
    Args:
        image (Image.Image): PIL Image object
        factor (float): Contrast factor (0.0-2.0, 1.0 is original)
        
    Returns:
        Image.Image: Adjusted image
    """
    # Adjust contrast
    enhancer = ImageEnhance.Contrast(image)
    adjusted_image = enhancer.enhance(factor)
    
    return adjusted_image


def adjust_sharpness(image: Image.Image, factor: float) -> Image.Image:
    """Adjust image sharpness.
    
    Args:
        image (Image.Image): PIL Image object
        factor (float): Sharpness factor (0.0-2.0, 1.0 is original)
        
    Returns:
        Image.Image: Adjusted image
    """
    # Adjust sharpness
    enhancer = ImageEnhance.Sharpness(image)
    adjusted_image = enhancer.enhance(factor)
    
    return adjusted_image


def convert_to_grayscale(image: Image.Image) -> Image.Image:
    """Convert an image to grayscale.
    
    Args:
        image (Image.Image): PIL Image object
        
    Returns:
        Image.Image: Grayscale image
    """
    # Convert to grayscale
    grayscale_image = image.convert('L')
    
    return grayscale_image


def apply_blur(image: Image.Image, radius: float = 2.0) -> Image.Image:
    """Apply Gaussian blur to an image.
    
    Args:
        image (Image.Image): PIL Image object
        radius (float): Blur radius
        
    Returns:
        Image.Image: Blurred image
    """
    # Apply blur
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius))
    
    return blurred_image


def apply_threshold(image: Image.Image, threshold: int = 128) -> Image.Image:
    """Apply threshold to an image.
    
    Args:
        image (Image.Image): PIL Image object
        threshold (int): Threshold value (0-255)
        
    Returns:
        Image.Image: Thresholded image
    """
    # Convert to grayscale if not already
    if image.mode != 'L':
        image = convert_to_grayscale(image)
    
    # Apply threshold
    thresholded_image = image.point(lambda x: 255 if x > threshold else 0, mode='1')
    
    return thresholded_image


def enhance_image_for_ocr(image: Image.Image) -> Image.Image:
    """Enhance an image for OCR processing.
    
    Args:
        image (Image.Image): PIL Image object
        
    Returns:
        Image.Image: Enhanced image
    """
    # Convert to grayscale
    image = convert_to_grayscale(image)
    
    # Increase contrast
    image = adjust_contrast(image, 1.5)
    
    # Apply slight sharpening
    image = adjust_sharpness(image, 1.5)
    
    # Apply adaptive thresholding using OpenCV
    # Convert PIL image to OpenCV format
    img_np = np.array(image)
    
    # Apply adaptive thresholding
    img_cv = cv2.adaptiveThreshold(
        img_np,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # Block size
        2    # Constant subtracted from mean
    )
    
    # Convert back to PIL image
    enhanced_image = Image.fromarray(img_cv)
    
    return enhanced_image


def detect_text_regions(image: Image.Image) -> List[Tuple[int, int, int, int]]:
    """Detect text regions in an image.
    
    Args:
        image (Image.Image): PIL Image object
        
    Returns:
        List[Tuple[int, int, int, int]]: List of text region bounding boxes (left, top, right, bottom)
    """
    # Convert PIL image to OpenCV format
    img_np = np.array(image)
    if len(img_np.shape) == 3:
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_np
    
    # Apply adaptive thresholding
    img_thresh = cv2.adaptiveThreshold(
        img_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,  # Block size
        2    # Constant subtracted from mean
    )
    
    # Find contours
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to find text regions
    text_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out small regions
        if w > 20 and h > 5 and w * h > 100:
            text_regions.append((x, y, x + w, y + h))
    
    return text_regions


def deskew_image(image: Image.Image) -> Image.Image:
    """Deskew (straighten) an image.
    
    Args:
        image (Image.Image): PIL Image object
        
    Returns:
        Image.Image: Deskewed image
    """
    # Convert PIL image to OpenCV format
    img_np = np.array(image)
    if len(img_np.shape) == 3:
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_np
    
    # Apply thresholding
    _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find all contours
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour
    if not contours:
        return image
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Find the minimum area rectangle that encloses the contour
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]
    
    # Adjust the angle
    if angle < -45:
        angle = 90 + angle
    
    # Rotate the image to deskew it
    (h, w) = img_gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Convert back to PIL image
    if len(img_np.shape) == 3:
        rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    
    deskewed_image = Image.fromarray(rotated)
    
    return deskewed_image


def remove_noise(image: Image.Image) -> Image.Image:
    """Remove noise from an image.
    
    Args:
        image (Image.Image): PIL Image object
        
    Returns:
        Image.Image: Denoised image
    """
    # Convert PIL image to OpenCV format
    img_np = np.array(image)
    if len(img_np.shape) == 3:
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_np
        img_cv = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    # Apply denoising
    denoised = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)
    
    # Convert back to PIL image
    denoised = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    denoised_image = Image.fromarray(denoised)
    
    return denoised_image


def get_image_dimensions(image: Image.Image) -> Tuple[int, int]:
    """Get the dimensions of an image.
    
    Args:
        image (Image.Image): PIL Image object
        
    Returns:
        Tuple[int, int]: Image dimensions (width, height)
    """
    return image.size


def get_image_format(image: Image.Image) -> str:
    """Get the format of an image.
    
    Args:
        image (Image.Image): PIL Image object
        
    Returns:
        str: Image format
    """
    return image.format or "UNKNOWN"


def get_image_mode(image: Image.Image) -> str:
    """Get the mode of an image.
    
    Args:
        image (Image.Image): PIL Image object
        
    Returns:
        str: Image mode (e.g., 'RGB', 'L', 'CMYK')
    """
    return image.mode


def get_image_info(image: Image.Image) -> dict:
    """Get information about an image.
    
    Args:
        image (Image.Image): PIL Image object
        
    Returns:
        dict: Image information
    """
    width, height = get_image_dimensions(image)
    
    return {
        "width": width,
        "height": height,
        "format": get_image_format(image),
        "mode": get_image_mode(image),
        "info": image.info,
    }