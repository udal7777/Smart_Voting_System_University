"""
Face recognition utilities using OpenCV's face detection and recognition
Enhanced with better face detection and robust error handling
"""
import base64
import logging
import os
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global flag to track successful initialization
face_detection_initialized = False

# Initialize face detection models
face_cascade = None
dnn_face_detector = None
use_dnn_detection = False

def initialize_face_detection():
    """Initialize face detection models with fallback options"""
    global face_cascade, dnn_face_detector, use_dnn_detection, face_detection_initialized
    
    logger.info("Initializing face detection system...")
    
    try:
        # First try to load Haar cascade - our reliable fallback
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Check if cascade loaded correctly
        if face_cascade.empty():
            # Try direct path as fallback
            logger.warning("Default cascade path failed, trying alternate path")
            alt_path = os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml')
            face_cascade = cv2.CascadeClassifier(alt_path)
            
            if face_cascade.empty():
                logger.error("Failed to load face cascade classifier")
                face_cascade = None
            else:
                logger.info("Successfully loaded face cascade from alternate path")
        else:
            logger.info("Successfully loaded face cascade classifier")
            
        face_detection_initialized = True
        return True
    
    except Exception as e:
        logger.error(f"Error initializing face detection: {str(e)}")
        face_cascade = None
        return False

# Initialize face detection on module import
face_detection_initialized = initialize_face_detection()

def decode_base64_image(base64_string):
    """
    Convert a base64 string to an image for processing
    
    Args:
        base64_string: The base64 encoded image data

    Returns:
        numpy.ndarray: The decoded image as a numpy array
    """
    try:
        # Remove header if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 string
        image_data = base64.b64decode(base64_string)
        
        # Convert to numpy array
        image = Image.open(BytesIO(image_data))
        return np.array(image)
    except Exception as e:
        logging.error(f"Error decoding image: {str(e)}")
        return None

def detect_faces_dnn(image_array):
    """
    Detect faces using DNN-based ML model
    
    Args:
        image_array: Numpy array representing the image
        
    Returns:
        list: Detected faces as [x, y, w, h]
    """
    if not use_dnn_detection or dnn_face_detector is None:
        return []
    
    try:
        # Get image dimensions
        h, w = image_array.shape[:2]
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(
            image_array, 1.0, (300, 300), 
            [104, 117, 123], False, False
        )
        
        # Set input and run forward pass
        dnn_face_detector.setInput(blob)
        detections = dnn_face_detector.forward()
        
        # Process results
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter weak detections
            if confidence > 0.7:  # Higher threshold for better accuracy
                # Get coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                
                # Extract region properties and add to faces
                faces.append([x1, y1, x2-x1, y2-y1])
        
        return faces
    except Exception as e:
        logger.error(f"DNN face detection failed: {str(e)}")
        return []

def get_face_features(image_array):
    """
    Extract facial features from an image using multi-scale detection
    for improved reliability
    
    Args:
        image_array: Numpy array representing the image

    Returns:
        numpy.ndarray: Face features as a vector
    """
    if not face_detection_initialized:
        if not initialize_face_detection():
            logger.error("Face detection system not initialized")
            return None
    
    try:
        # Ensure image is properly formatted
        if image_array is None or image_array.size == 0:
            logger.error("Empty image array provided")
            return None
            
        # Convert to grayscale for face detection
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            rgb = image_array
        else:
            gray = image_array
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Make sure our grayscale image is valid
        if gray is None or gray.size == 0:
            logger.error("Failed to convert image to grayscale")
            return None
        
        # Multi-scale face detection with Haar cascade (more reliable)
        faces = []
        
        # Make sure face cascade is loaded
        if face_cascade is None:
            logger.error("Face cascade classifier not loaded")
            return None
            
        # Try multiple scale factors for better detection rate
        scale_factors = [1.1, 1.2, 1.3, 1.5]
        min_neighbors_options = [3, 4, 5]
        
        # Try different detection parameters until we find a face
        for scale in scale_factors:
            for min_neighbors in min_neighbors_options:
                try:
                    # Apply histogram equalization for better feature contrast
                    equalized_gray = cv2.equalizeHist(gray)
                    
                    detected_faces = face_cascade.detectMultiScale(
                        equalized_gray,
                        scaleFactor=scale,
                        minNeighbors=min_neighbors,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    if len(detected_faces) > 0:
                        logger.debug(f"Face detected with scale={scale}, neighbors={min_neighbors}")
                        faces = detected_faces
                        break
                except Exception as cascade_error:
                    logger.error(f"Error in cascade detection: {str(cascade_error)}")
                    continue
                    
            if len(faces) > 0:
                break
        
        # If we failed to detect any faces, try one more time with very permissive parameters
        if len(faces) == 0:
            logger.warning("Trying last-resort face detection with permissive parameters")
            try:
                # Apply heavy preprocessing
                equalized_gray = cv2.equalizeHist(gray)
                blurred_gray = cv2.GaussianBlur(equalized_gray, (5, 5), 0)
                
                faces = face_cascade.detectMultiScale(
                    blurred_gray,
                    scaleFactor=1.1,
                    minNeighbors=2,
                    minSize=(20, 20),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
            except Exception as last_resort_error:
                logger.error(f"Last resort detection failed: {str(last_resort_error)}")
        
        # Check if we have any faces
        if len(faces) == 0:
            logger.warning("No faces detected in the image after multiple attempts")
            return None
        
        # Get the largest face detected
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        x, y, w, h = largest_face
        
        # Log face detection success
        logger.info(f"Face detected at ({x},{y}) with size {w}x{h}")
        
        # Add padding to include more of the face (20% padding)
        padding_x = int(w * 0.2)
        padding_y = int(h * 0.2)
        
        # Ensure coordinates are within image bounds
        x_start = max(0, x - padding_x)
        y_start = max(0, y - padding_y)
        x_end = min(gray.shape[1], x + w + padding_x)
        y_end = min(gray.shape[0], y + h + padding_y)
        
        # Extract the face ROI (Region of Interest) with padding
        face_roi = gray[y_start:y_end, x_start:x_end]
        
        # Resize to standard size for consistent comparison
        face_roi = cv2.resize(face_roi, (100, 100))
        
        # Apply advanced preprocessing to optimize feature extraction
        # Step 1: Normalize lighting with histogram equalization
        face_roi = cv2.equalizeHist(face_roi)
        
        # Step 2: Apply Gaussian blur to reduce noise (smaller kernel for sharper features)
        face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)
        
        # Step 3: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face_roi = clahe.apply(face_roi)
        
        # Step 4: Edge enhancement using Sobel operator
        sobel_x = cv2.Sobel(face_roi, cv2.CV_8U, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(face_roi, cv2.CV_8U, 0, 1, ksize=3)
        face_roi = cv2.addWeighted(face_roi, 0.7, cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0), 0.3, 0)
        
        # Flatten the image into a feature vector
        features = face_roi.flatten().astype(np.uint8)
        
        # Verify feature vector has data
        if features is None or len(features) == 0:
            logger.error("Failed to extract face features - empty feature vector")
            return None
            
        logger.info(f"Successfully extracted {len(features)} face features")
        return features
        
    except Exception as e:
        logger.error(f"Error extracting face features: {str(e)}")
        return None

def get_face_encoding(image_data_uri):
    """
    Get face encoding from an image
    
    Args:
        image_data_uri: Base64 encoded image data

    Returns:
        tuple: (encoded_features, None) if successful, or (None, error_message) if failed
    """
    try:
        # Decode the image
        image_array = decode_base64_image(image_data_uri)
        if image_array is None:
            return (None, "Failed to decode image")
        
        # Extract face features
        features = get_face_features(image_array)
        if features is None:
            return (None, "No face detected in the image")
        
        # Encode the features for storage
        encoded_features = base64.b64encode(features.tobytes()).decode()
        
        return (encoded_features, None)
    except Exception as e:
        logging.error(f"Error in face encoding: {str(e)}")
        return (None, "Failed to process face data")

def advanced_similarity(feature1, feature2):
    """
    Calculate similarity between two feature vectors using multiple metrics
    
    Args:
        feature1: First feature vector
        feature2: Second feature vector
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # 1. Calculate correlation coefficient
    correlation = np.corrcoef(feature1, feature2)[0, 1]
    
    # 2. Calculate cosine similarity
    norm1 = np.linalg.norm(feature1)
    norm2 = np.linalg.norm(feature2)
    if norm1 == 0 or norm2 == 0:
        cosine_sim = 0
    else:
        cosine_sim = np.dot(feature1, feature2) / (norm1 * norm2)
    
    # 3. Calculate Euclidean distance (convert to similarity)
    euclidean_dist = np.linalg.norm(feature1 - feature2)
    max_dist = np.sqrt(len(feature1)) * 255  # Maximum possible distance
    euclidean_sim = 1 - (euclidean_dist / max_dist)
    
    # Weighted combination of similarity metrics
    # Give more weight to correlation which is typically more reliable
    similarity = (0.5 * correlation) + (0.3 * cosine_sim) + (0.2 * euclidean_sim)
    
    # Ensure value is between 0 and 1
    return max(0, min(1, similarity))

def process_multiple_frames(frames_base64):
    """
    Process multiple face frames to create a more robust face encoding
    
    Args:
        frames_base64: List of base64 encoded face images
        
    Returns:
        tuple: (encoded_features, error_message)
    """
    if not frames_base64 or len(frames_base64) == 0:
        return (None, "No frames provided")
        
    logger.info(f"Processing {len(frames_base64)} face frames")
    
    all_features = []
    success_count = 0
    
    # Process each frame
    for idx, frame in enumerate(frames_base64):
        try:
            # Get face features from this frame
            image_array = decode_base64_image(frame)
            if image_array is not None:
                features = get_face_features(image_array)
                if features is not None:
                    all_features.append(features)
                    success_count += 1
                    logger.debug(f"Successfully extracted features from frame {idx+1}")
                else:
                    logger.warning(f"No face detected in frame {idx+1}")
            else:
                logger.warning(f"Failed to decode frame {idx+1}")
        except Exception as e:
            logger.error(f"Error processing frame {idx+1}: {str(e)}")
    
    # Check if we have enough successful frames
    if success_count == 0:
        return (None, "Failed to detect face in any frame")
    
    # Combine features from all frames (average them)
    # First ensure all feature vectors have the same length
    min_length = min(len(features) for features in all_features)
    
    # Truncate all features to the same length
    all_features = [features[:min_length] for features in all_features]
    
    # Calculate the average feature vector
    average_features = np.mean(all_features, axis=0).astype(np.uint8)
    
    # Encode the averaged features
    encoded_features = base64.b64encode(average_features.tobytes()).decode()
    
    logger.info(f"Successfully processed {success_count} frames and created combined face encoding")
    return (encoded_features, None)

def compare_faces(stored_encoding_base64, current_face_data):
    """
    Compare stored face encoding with current face(s) using robust comparison
    
    Args:
        stored_encoding_base64: Base64 encoded stored face features
        current_face_data: Base64 encoded image data or a JSON-encoded encoding from process_multiple_frames

    Returns:
        tuple: (match_result, error_message)
    """
    # Initialize face detection if needed
    if not face_detection_initialized:
        if not initialize_face_detection():
            return (False, "Face detection system not initialized")
            
    try:
        # Determine if current_face_data is already processed face encoding
        if isinstance(current_face_data, str) and current_face_data.startswith('data:image'):
            # This is a standard base64 image
            current_image = decode_base64_image(current_face_data)
            if current_image is None:
                return (False, "Failed to decode current image. Please try again.")
            
            # Extract features with enhanced face detection
            current_features = get_face_features(current_image)
            if current_features is None:
                return (False, "No face detected in the current image. Please position your face properly and ensure good lighting.")
        else:
            # Assume this is already an encoded feature set from process_multiple_frames
            try:
                # Try to decode directly (in case it's a direct encoding)
                current_features = np.frombuffer(base64.b64decode(current_face_data), dtype=np.uint8)
            except:
                # If that fails, it might be a raw base64 image string
                current_image = decode_base64_image(current_face_data)
                if current_image is not None:
                    current_features = get_face_features(current_image)
                    if current_features is None:
                        return (False, "No face detected in the provided image")
                else:
                    return (False, "Invalid face data format")
        
        logger.info(f"Current face features extracted: {len(current_features)} points")
        
        # Step 3: Decode stored features with robust error handling
        try:
            # Handle potential padding issues in base64
            padded_encoding = stored_encoding_base64
            if len(stored_encoding_base64) % 4 != 0:
                padded_encoding = stored_encoding_base64 + '=' * (4 - len(stored_encoding_base64) % 4)
                
            stored_features_bytes = base64.b64decode(padded_encoding)
            stored_features = np.frombuffer(stored_features_bytes, dtype=np.uint8)
            stored_features = stored_features.reshape(-1)  # Ensure it's a 1D array
            
            logger.info(f"Stored face features decoded: {len(stored_features)} points")
        except Exception as decode_error:
            logger.error(f"Error decoding stored features: {str(decode_error)}")
            # Try to recover corrupted data with more permissive decoding
            try:
                stored_features_bytes = base64.b64decode(stored_encoding_base64 + '==', validate=False)
                stored_features = np.frombuffer(stored_features_bytes, dtype=np.uint8)
                stored_features = stored_features.reshape(-1)
                logger.info("Recovered features with permissive decoding")
            except:
                return (False, "Invalid stored face data. Please register again.")
        
        # Step 4: Ensure we have enough data and features match in length
        min_length = min(len(stored_features), len(current_features))
        if min_length < 100:  # Sanity check for feature vector size
            logger.warning(f"Face feature vectors too small: {min_length} points")
            return (False, "Face feature extraction incomplete. Please try again with better positioning.")
            
        # Truncate to same length for comparison
        stored_features = stored_features[:min_length]
        current_features = current_features[:min_length]
        
        # Step 5: Calculate similarity using multiple metrics
        similarity = advanced_similarity(stored_features, current_features)
        logger.info(f"Raw similarity score: {similarity:.4f}")
        
        # Step 6: Apply multiple checks and adaptive thresholding
        # Base threshold - more permissive for better user experience
        base_threshold = 0.45
        
        # Adjust based on image quality
        feature_range = np.max(current_features) - np.min(current_features)
        contrast_factor = min(0.15, feature_range / 255)  # Higher contrast gets bonus
        
        # Adjust based on feature vector statistical properties
        std_dev = np.std(current_features)
        variance_factor = min(0.1, std_dev / 50)  # Higher variance (more details) gets bonus
        
        # Combined dynamic threshold
        adjusted_threshold = base_threshold - contrast_factor - variance_factor
        
        # Ensure threshold doesn't become too low
        adjusted_threshold = max(0.3, adjusted_threshold)
        
        # Log detailed comparison metrics
        logger.info(f"Face comparison details:")
        logger.info(f"  - Similarity score: {similarity:.4f}")
        logger.info(f"  - Base threshold: {base_threshold:.4f}")
        logger.info(f"  - Contrast adjustment: -{contrast_factor:.4f}")
        logger.info(f"  - Variance adjustment: -{variance_factor:.4f}")
        logger.info(f"  - Final threshold: {adjusted_threshold:.4f}")
        
        # Make match decision
        is_match = similarity > adjusted_threshold
        
        # Return result with detailed logging
        if is_match:
            logger.info(f"Face MATCH SUCCESSFUL: {similarity:.4f} > {adjusted_threshold:.4f}")
            return (True, None)
        else:
            # Additional checks for near-misses to improve user experience
            if similarity > (adjusted_threshold - 0.1):
                logger.info(f"Face near-match: {similarity:.4f} ≈ {adjusted_threshold:.4f}")
                return (False, "Face partially matched but not enough. Please try again with better lighting.")
            else:
                logger.info(f"Face match FAILED: {similarity:.4f} < {adjusted_threshold:.4f}")
                return (False, "Face doesn't match. Please ensure you're using the correct account.")
    
    except Exception as e:
        logger.error(f"Error comparing faces: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return (False, "Face comparison failed due to a technical issue. Please try again.")