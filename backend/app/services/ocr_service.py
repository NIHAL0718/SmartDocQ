"""Service for OCR (Optical Character Recognition) functionality."""

import os
import time
from typing import Dict, Any, Optional
import pytesseract
import easyocr
import json
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import logging
import yaml

from app.core.logging import log_execution_time, log_step_execution_time

# Get OCR-specific logger
ocr_logger = logging.getLogger("smartdocq.ocr")

# Configure Tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Uncomment and adjust for Windows

# Initialize EasyOCR reader for multiple languages
reader = None  # Lazy initialization to save memory

# Language mapping for Tesseract and EasyOCR
LANGUAGE_MAP = {
    "english": {"tesseract": "eng", "easyocr": "en"},
    "hindi": {"tesseract": "hin", "easyocr": "hi"},
    "telugu": {"tesseract": "tel", "easyocr": "te"},
    "tamil": {"tesseract": "tam", "easyocr": "ta"},
}

# Ensure EasyOCR directories and config files exist
def ensure_easyocr_config():
    """Create necessary EasyOCR directories and config files if they don't exist."""
    try:
        # Create user_network directory if it doesn't exist
        user_network_dir = os.path.expanduser("~/.EasyOCR/user_network")
        os.makedirs(user_network_dir, exist_ok=True)
        
        # Create fast.yaml if it doesn't exist
        fast_yaml_path = os.path.join(user_network_dir, "fast.yaml")
        if not os.path.exists(fast_yaml_path):
            fast_config = {
                "imgH": 64,
                "imgW": 600,
                "input_channel": 1,
                "output_channel": 512,
                "hidden_size": 256,
                "batch_max_length": 25,
                "character": "0123456789abcdefghijklmnopqrstuvwxyz",
                "sensitive": True,
                "backbone": "ResNet",
                "sequence_modeling": "BiLSTM",
                "prediction": "CTC",
                "transformation": "TPS"
            }
            
            with open(fast_yaml_path, 'w') as f:
                yaml.dump(fast_config, f, default_flow_style=False)
            
            ocr_logger.info(f"Created EasyOCR fast.yaml configuration at {fast_yaml_path}")
    except Exception as e:
        ocr_logger.error(f"Error creating EasyOCR config: {str(e)}")

# Ensure config exists
ensure_easyocr_config()


@log_execution_time(logger_name="smartdocq.ocr", level=logging.INFO)
def process_image_ocr(
    ocr_id: str,
    file_path: str,
    language: str = None,  # Now optional, will be auto-detected if None
    enhance_image: bool = False,
    use_easyocr: bool = True,
) -> Dict[str, Any]:
    """Process an image with OCR to extract text.
    
    Args:
        ocr_id: Unique identifier for the OCR job
        file_path: Path to the image file
        language: Language of the text in the image (optional, auto-detected if None)
        enhance_image: Whether to enhance the image before OCR
        use_easyocr: Whether to use EasyOCR (True) or Tesseract (False)
        
    Returns:
        Dictionary with OCR results
    """
    try:
        ocr_logger.info(f"Starting OCR processing for job {ocr_id} with file {os.path.basename(file_path)}")
        ocr_logger.info(f"OCR parameters: language={language}, enhance_image={enhance_image}, use_easyocr={use_easyocr}")
        
        # Start timing the processing
        start_time = time.time()
        
        # Check if file exists
        if not os.path.exists(file_path):
            ocr_logger.error(f"File not found: {file_path}")
            return {
                "status": "error",
                "message": f"File not found: {file_path}",
                "ocr_id": ocr_id,
            }
        
        # Get file extension
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Handle PDF files
        if file_extension == ".pdf":
            ocr_logger.info(f"Processing PDF file: {os.path.basename(file_path)}")
            return process_pdf_ocr(ocr_id, file_path, language, enhance_image, use_easyocr)
        
        # Open the image file
        with log_step_execution_time("smartdocq.ocr")("image_loading"):
            image = Image.open(file_path)
            ocr_logger.info(f"Image loaded: {image.size[0]}x{image.size[1]} pixels, mode={image.mode}")
        
        # Prepare the image
        if enhance_image:
            with log_step_execution_time("smartdocq.ocr")("image_enhancement"):
                image = prepare_image(image, enhance=True)
                ocr_logger.info("Image enhancement completed")
        
        # Auto-detect language if not specified
        detected_language = language
        if detected_language is None:
            with log_step_execution_time("smartdocq.ocr")("language_detection"):
                detected_language = detect_language(image)
                ocr_logger.info(f"Language detection completed: {detected_language}")
        
        # Map language to OCR engine format
        lang_code = get_language_code(detected_language, use_easyocr)
        
        # Extract text using the selected OCR engine
        with log_step_execution_time("smartdocq.ocr")("text_extraction"):
            if use_easyocr:
                text, confidence = extract_text_easyocr(image, lang_code)
                ocr_logger.info(f"EasyOCR extraction completed with confidence: {confidence:.2f}")
            else:
                text, confidence = extract_text_tesseract(image, lang_code)
                ocr_logger.info(f"Tesseract extraction completed with confidence: {confidence:.2f}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare result
        result = {
            "status": "success",
            "ocr_id": ocr_id,
            "text": text,
            "confidence": confidence,
            "language": detected_language,
            "processing_time": processing_time,
            "engine": "EasyOCR" if use_easyocr else "Tesseract",
        }
        
        # Log performance metrics
        ocr_logger.info(f"OCR processing completed in {processing_time:.2f} seconds")
        ocr_logger.info(f"Text extracted: {len(text)} characters, {len(text.split())} words")
        
        # Save result to file
        try:
            with log_step_execution_time("smartdocq.ocr")("result_saving"):
                # Ensure directory exists
                os.makedirs("uploads/ocr", exist_ok=True)
                
                # Save result to JSON file
                result_file = f"uploads/ocr/{ocr_id}_result.json"
                with open(result_file, "w") as f:
                    json.dump(result, f)
                ocr_logger.info(f"Result saved to {result_file}")
        except Exception as save_error:
            ocr_logger.error(f"Error saving OCR result to file: {str(save_error)}")
        
        return result
    
    except Exception as e:
        # Log the error
        ocr_logger.error(f"Error processing OCR for {ocr_id}: {str(e)}", exc_info=True)
        
        # Return error information
        return {
            "status": "error",
            "ocr_id": ocr_id,
            "error": str(e),
        }


@log_execution_time(logger_name="smartdocq.ocr", level=logging.INFO)
def prepare_image(image_input, enhance: bool = False) -> Image.Image:
    """Prepare an image for OCR processing with advanced enhancement.
    
    Args:
        image_input: Either a file path (str) or a PIL Image object
        enhance: Whether to enhance the image
        
    Returns:
        Processed PIL Image object
    """
    # Handle input that can be either a file path or a PIL Image
    with log_step_execution_time("smartdocq.ocr")("image_loading"):
        if isinstance(image_input, str):
            # It's a file path
            image = Image.open(image_input)
        else:
            # It's already a PIL Image
            image = image_input
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
    
    # Apply enhancements if requested
    if enhance:
        with log_step_execution_time("smartdocq.ocr")("image_enhancement"):
            # Convert PIL image to OpenCV format for advanced processing
            img_np = np.array(image)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Check image dimensions and skip heavy processing for very large images
            h, w = img_cv.shape[:2]
            if h * w > 25000000:  # Skip heavy processing for very large images (e.g., > 25MP)
                ocr_logger.warning(f"Using simplified enhancement for large image: {w}x{h}")
                # Simple grayscale conversion and basic thresholding for large images
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return Image.fromarray(thresh)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding with improved parameters
            block_size = 15  # Increased from 11 for better results with varying lighting
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, block_size, 2
            )
            
            # Apply noise reduction with optimized parameters
            denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
            
            # Apply morphological operations to remove noise
            kernel = np.ones((2, 2), np.uint8)
            opening = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
            
            # Apply dilation to make text thicker and more readable
            kernel = np.ones((1, 1), np.uint8)
            dilated = cv2.dilate(opening, kernel, iterations=1)
            
            # Convert back to PIL image
            image = Image.fromarray(dilated)
            ocr_logger.info(f"Image enhanced successfully: {w}x{h}")
    
    return image


@log_execution_time(logger_name="smartdocq.ocr", level=logging.INFO)
def extract_text_tesseract(image: Image.Image, language: str) -> tuple[str, float]:
    """Extract text from an image using Tesseract OCR.
    
    Args:
        image: PIL Image object
        language: Tesseract language code
        
    Returns:
        Tuple of (extracted text, confidence score)
    """
    try:
        # Convert PIL Image to OpenCV format for preprocessing
        with log_step_execution_time("smartdocq.ocr")("image_preprocessing"):
            img_np = np.array(image)
            h, w = img_np.shape[:2] if len(img_np.shape) > 2 else img_np.shape
            ocr_logger.debug(f"Processing image with Tesseract: {w}x{h} pixels")
            
            # Resize very large images to improve processing speed
            if w > 3000 or h > 3000:
                scale_factor = min(3000 / w, 3000 / h) if w > h else min(3000 / h, 3000 / w)
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
                ocr_logger.debug(f"Resized image for Tesseract: {w}x{h} -> {new_w}x{new_h}")
        
        # Extract text with Tesseract with optimized parameters
        with log_step_execution_time("smartdocq.ocr")("tesseract_text_extraction"):
            # Add optimized configuration for better performance
            config = "--oem 1 --psm 3"  # Use LSTM OCR Engine mode with auto page segmentation
            data = pytesseract.image_to_data(img_np, lang=language, output_type=pytesseract.Output.DICT, config=config)
        
        # Combine text and calculate average confidence
        text_parts = []
        confidence_values = []
        
        for i in range(len(data["text"])):
            # Filter out low-confidence results and empty text
            if int(data["conf"][i]) > 20 and data["text"][i].strip():  # Increased threshold from 0 to 20
                text_parts.append(data["text"][i])
                confidence_values.append(int(data["conf"][i]))
        
        # Join text parts with appropriate spacing
        text = " ".join(text_parts)
        
        # Calculate average confidence (0-1 scale)
        avg_confidence = sum(confidence_values) / len(confidence_values) / 100 if confidence_values else 0
        
        ocr_logger.info(f"Tesseract extracted {len(text_parts)} text segments with avg confidence: {avg_confidence:.2f}")
        
        return text, avg_confidence
    except Exception as e:
        ocr_logger.error(f"Error in Tesseract OCR: {str(e)}", exc_info=True)
        return "", 0.0


# Cache for EasyOCR readers to avoid recreating them for each request
reader_cache = {}

@log_execution_time(logger_name="smartdocq.ocr", level=logging.INFO)
def extract_text_easyocr(image: Image.Image, language: str) -> tuple[str, float]:
    """Extract text from an image using EasyOCR with improved language handling.
    
    Args:
        image: PIL Image object
        language: EasyOCR language code
        
    Returns:
        Tuple of (extracted text, confidence score)
    """
    global reader_cache
    
    # Prepare language list
    languages = [language]
    
    # Always include English as a fallback unless it's already the primary language
    if language != "en":
        languages.append("en")
    
    # Create a cache key based on the languages
    cache_key = ",".join(sorted(languages))
    
    # Use cached reader if available, otherwise create a new one
    with log_step_execution_time("smartdocq.ocr")("reader_initialization"):
        if cache_key not in reader_cache:
            ocr_logger.info(f"Creating new EasyOCR reader for languages: {languages}")
            try:
                reader_cache[cache_key] = easyocr.Reader(languages, gpu=False, quantize=True, recog_network='fast')  # Use fast recognition for better performance
            except KeyError as e:
                # Handle the 'imgH' KeyError by using default network instead
                ocr_logger.warning(f"Error with 'fast' network configuration: {str(e)}. Falling back to default network.")
                reader_cache[cache_key] = easyocr.Reader(languages, gpu=False, quantize=True)  # Use default network
        
        reader = reader_cache[cache_key]
    
    # Resize large images to improve processing speed
    with log_step_execution_time("smartdocq.ocr")("image_preprocessing"):
        img_np = np.array(image)
        h, w = img_np.shape[:2] if len(img_np.shape) > 2 else img_np.shape
        
        ocr_logger.debug(f"Processing image with EasyOCR: {w}x{h} pixels")
        
        # Resize if the image is too large (width > 1000 pixels)
        if w > 1000:
            scale_factor = 1000 / w
            new_w = 1000
            new_h = int(h * scale_factor)
            img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
            ocr_logger.debug(f"Resized image for EasyOCR: {w}x{h} -> {new_w}x{new_h}")
    
    # Extract text with EasyOCR - pass the numpy array directly
    with log_step_execution_time("smartdocq.ocr")("easyocr_text_extraction"):
        # Use paragraph=True for faster processing when appropriate
        use_paragraph = h * w > 500000  # For larger images, use paragraph mode for speed
        results = reader.readtext(img_np, paragraph=use_paragraph)
    
    # Combine text and calculate average confidence
    text_parts = []
    confidence_values = []
    
    for (_, text, confidence) in results:
        # Skip very low confidence results
        if confidence < 0.2:
            continue
            
        text_parts.append(text)
        confidence_values.append(confidence)
    
    # Join text parts with appropriate spacing
    text = " ".join(text_parts)
    
    # Calculate average confidence
    avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
    
    ocr_logger.info(f"EasyOCR extracted {len(text_parts)} text segments with avg confidence: {avg_confidence:.2f}")
    
    return text, avg_confidence


def process_pdf_ocr(ocr_id: str, file_path: str, language: str = None, enhance_image: bool = False, use_easyocr: bool = True) -> Dict[str, Any]:
    """Process a PDF file with OCR.
    
    Args:
        ocr_id: Unique identifier for the OCR job
        file_path: Path to the PDF file
        language: Language of the text in the PDF (optional, auto-detected if None)
        enhance_image: Whether to enhance the images before OCR
        use_easyocr: Whether to use EasyOCR or Tesseract
        
    Returns:
        Dictionary with OCR results
    """
    try:
        import fitz  # PyMuPDF
        
        start_time = time.time()
        
        # Open the PDF
        pdf = fitz.open(file_path)
        
        # Process each page
        all_text = []
        confidence_values = []
        detected_languages = []
        
        for page_num in range(len(pdf)):
            # Get page
            page = pdf[page_num]
            
            # Check if page has text (no OCR needed)
            page_text = page.get_text()
            if page_text.strip():
                all_text.append(page_text)
                confidence_values.append(1.0)  # High confidence for embedded text
                # For embedded text, we'll use the specified language or default to English
                detected_languages.append(language or "english")
                continue
            
            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Prepare the image
            if enhance_image:
                img = prepare_image(img, enhance=True)
            
            # Auto-detect language if not specified
            page_language = language
            if page_language is None:
                page_language = detect_language(img)
                detected_languages.append(page_language)
            else:
                detected_languages.append(page_language)
            
            # Map language to OCR engine format
            lang_code = get_language_code(page_language, use_easyocr)
            
            # Extract text using the selected OCR engine
            if use_easyocr:
                text, confidence = extract_text_easyocr(img, lang_code)
            else:
                text, confidence = extract_text_tesseract(img, lang_code)
            
            all_text.append(text)
            confidence_values.append(confidence)
        
        # Combine text from all pages
        combined_text = "\n\n".join(all_text)
        
        # Calculate average confidence
        avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
        
        # Determine the most common detected language
        if detected_languages:
            from collections import Counter
            most_common_language = Counter(detected_languages).most_common(1)[0][0]
        else:
            most_common_language = language or "english"
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "ocr_id": ocr_id,
            "text": combined_text,
            "confidence": avg_confidence,
            "language": most_common_language,
            "processing_time": processing_time,
            "engine": "EasyOCR" if use_easyocr else "Tesseract",
            "page_count": len(pdf),
        }
    
    except Exception as e:
        # Log the error
        print(f"Error processing PDF OCR for {ocr_id}: {str(e)}")
        
        # Return error information
        return {
            "status": "error",
            "ocr_id": ocr_id,
            "error": str(e),
        }


def get_language_code(language: str, use_easyocr: bool) -> str:
    """Get the appropriate language code for the OCR engine.
    
    Args:
        language: Language name (e.g., "english", "hindi")
        use_easyocr: Whether using EasyOCR (True) or Tesseract (False)
        
    Returns:
        Language code for the selected OCR engine
    """
    language = language.lower()
    
    # Get language mapping
    lang_map = LANGUAGE_MAP.get(language, {"tesseract": "eng", "easyocr": "en"})
    
    # Return appropriate code
    return lang_map["easyocr"] if use_easyocr else lang_map["tesseract"]


@log_execution_time(logger_name="smartdocq.ocr", level=logging.INFO)
def enhance_image_for_ocr(image_path: str, output_path: Optional[str] = None) -> str:
    """Enhance an image for better OCR results.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the enhanced image (if None, modifies the original)
        
    Returns:
        Path to the enhanced image
    """
    try:
        # Set output path if not provided
        if output_path is None:
            file_name, file_ext = os.path.splitext(image_path)
            output_path = f"{file_name}_enhanced{file_ext}"
        
        with log_step_execution_time("smartdocq.ocr")("image_loading"):
            # Read the image with OpenCV
            img = cv2.imread(image_path)
            
            # Check image dimensions and skip heavy processing for very large images
            h, w = img.shape[:2]
            if h * w > 25000000:  # Skip heavy processing for very large images (e.g., > 25MP)
                ocr_logger.warning(f"Using simplified enhancement for large image: {w}x{h}")
                # Simple grayscale conversion and basic thresholding for large images
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cv2.imwrite(output_path, thresh)
                return output_path
        
        with log_step_execution_time("smartdocq.ocr")("grayscale_conversion"):
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        with log_step_execution_time("smartdocq.ocr")("adaptive_thresholding"):
            # Apply adaptive thresholding with improved parameters
            block_size = 15  # Increased from 11 for better results with varying lighting
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2
            )
        
        with log_step_execution_time("smartdocq.ocr")("noise_reduction"):
            # Apply noise reduction with optimized parameters
            denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        with log_step_execution_time("smartdocq.ocr")("morphological_operations"):
            # Apply dilation to make text thicker
            kernel = np.ones((1, 1), np.uint8)
            dilated = cv2.dilate(denoised, kernel, iterations=1)
        
        # Save the enhanced image
        cv2.imwrite(output_path, dilated)
        ocr_logger.info(f"Image enhanced successfully: {w}x{h}, saved to {os.path.basename(output_path)}")
        
        return output_path
    except Exception as e:
        ocr_logger.error(f"Error enhancing image: {str(e)}", exc_info=True)
        # Return original path if enhancement fails
        return image_path


# Cache for language detection results to avoid redundant processing
language_detection_cache = {}

@log_execution_time(logger_name="smartdocq.ocr", level=logging.INFO)
def detect_language(image: Image.Image) -> str:
    """Detect the language of text in an image with improved accuracy.
    
    Args:
        image: PIL Image object
        
    Returns:
        Detected language name (e.g., "english", "hindi")
    """
    try:
        global reader_cache, language_detection_cache
        
        # Generate a simple hash of the image for caching
        # We'll use the image size and a sample of pixels as a simple fingerprint
        img_width, img_height = image.size
        img_mode = image.mode
        
        # Create a simple fingerprint from image properties and a sample of pixels
        # Take pixels from corners and center for a basic fingerprint
        try:
            pixels = [
                image.getpixel((0, 0)),
                image.getpixel((img_width-1, 0)),
                image.getpixel((0, img_height-1)),
                image.getpixel((img_width-1, img_height-1)),
                image.getpixel((img_width//2, img_height//2))
            ]
            # Convert pixels to strings and join them
            pixel_str = ",".join([str(p) for p in pixels])
            # Create a cache key from image properties
            cache_key = f"{img_width}x{img_height}_{img_mode}_{pixel_str}"
            
            # Check if we have a cached result for this image
            if cache_key in language_detection_cache:
                cached_lang = language_detection_cache[cache_key]
                ocr_logger.info(f"Using cached language detection result: {cached_lang}")
                return cached_lang
        except Exception as e:
            # If fingerprinting fails, log and continue without caching
            ocr_logger.warning(f"Failed to create image fingerprint for caching: {str(e)}")
            cache_key = None
        
        # Convert PIL Image to OpenCV format
        img_np = np.array(image)
        
        # Resize large images to improve processing speed for language detection
        h, w = img_np.shape[:2] if len(img_np.shape) > 2 else img_np.shape
        if w > 800:  # Use even smaller size for language detection
            scale_factor = 800 / w
            new_w = 800
            new_h = int(h * scale_factor)
            img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
            ocr_logger.debug(f"Resized image for language detection: {w}x{h} -> {new_w}x{new_h}")
        
        # Use cached reader for language detection if available
        lang_detect_key = "en,hi,ta,te"  # Sorted language codes
        if lang_detect_key not in reader_cache:
            ocr_logger.info("Creating new EasyOCR reader for language detection")
            try:
                reader_cache[lang_detect_key] = easyocr.Reader(["en", "hi", "te", "ta"], gpu=False, quantize=True, recog_network='fast')
            except FileNotFoundError as e:
                ocr_logger.warning(f"Fast recognition network not found: {str(e)}. Using default network instead.")
                reader_cache[lang_detect_key] = easyocr.Reader(["en", "hi", "te", "ta"], gpu=False, quantize=True)  # Use default network
        
        detector = reader_cache[lang_detect_key]
        
        # Extract text with EasyOCR - use paragraph=True for faster processing
        with log_step_execution_time("smartdocq.ocr")("language_detection_ocr"):
            results = detector.readtext(img_np, detail=1, paragraph=True)
        
        if not results:
            # No text detected, default to English
            ocr_logger.info("No text detected for language detection, defaulting to English")
            detected_lang = "english"
            # Cache the result if we have a valid cache key
            if cache_key:
                language_detection_cache[cache_key] = detected_lang
            return detected_lang
        
        # Count detected characters by language
        lang_scores = {
            "english": 0,
            "hindi": 0,
            "telugu": 0,
            "tamil": 0
        }
        
        # Enhanced heuristic: Use character ranges and confidence scores
        total_chars = 0
        for _, text, confidence in results:
            # Skip very low confidence results
            if confidence < 0.3:
                continue
                
            # Weight by confidence score
            weight = confidence
            
            for char in text:
                total_chars += 1
                # Check character ranges
                code = ord(char)
                if 0x0900 <= code <= 0x097F:  # Devanagari (Hindi)
                    lang_scores["hindi"] += weight
                elif 0x0C00 <= code <= 0x0C7F:  # Telugu
                    lang_scores["telugu"] += weight
                elif 0x0B80 <= code <= 0x0BFF:  # Tamil
                    lang_scores["tamil"] += weight
                elif 0x0000 <= code <= 0x007F:  # ASCII (English)
                    # Don't count punctuation and numbers for English
                    if not (char.isdigit() or char in '.,;:!?()[]{}"\'-=+/*&^%$#@'):
                        lang_scores["english"] += weight
        
        # Find language with highest score
        max_lang = max(lang_scores.items(), key=lambda x: x[1])
        
        # If no clear winner or very few characters detected, default to English
        if max_lang[1] == 0 or total_chars < 5:
            ocr_logger.info(f"Insufficient language data detected, defaulting to English (scores: {lang_scores})")
            detected_lang = "english"
        else:
            # Calculate percentage of detected text in the winning language
            total_score = sum(lang_scores.values())
            if total_score > 0:
                percentage = max_lang[1] / total_score
                # If the winning language doesn't have a clear majority, default to English
                if percentage < 0.6 and max_lang[0] != "english":
                    ocr_logger.info(f"No clear language majority ({percentage:.2f}), defaulting to English")
                    detected_lang = "english"
                else:
                    detected_lang = max_lang[0]
                    ocr_logger.info(f"Language detected: {detected_lang} with {percentage:.2f} confidence")
            else:
                detected_lang = "english"
        
        # Cache the result if we have a valid cache key
        if cache_key:
            language_detection_cache[cache_key] = detected_lang
            # Limit cache size to prevent memory issues
            if len(language_detection_cache) > 100:  # Arbitrary limit
                # Remove a random item from the cache
                try:
                    language_detection_cache.pop(next(iter(language_detection_cache)))
                except:
                    # If pop fails, just clear the cache
                    language_detection_cache.clear()
        
        return detected_lang
    
    except Exception as e:
        ocr_logger.error(f"Error detecting language: {str(e)}", exc_info=True)
        # Default to English on error
        return "english"