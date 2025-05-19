# Vision Module for Bruise Detection
# Contains classes and functions for computer vision and image processing

import numpy as np
import cv2
from PIL import Image, ImageEnhance
import tensorflow as tf

class BruiseDetectionModel:
    """
    Simulates a deep learning model for bruise detection using multiple light sources
    """
    def __init__(self):
        self.model_version = "2.1.0"
        self.supported_light_sources = ["white", "als_415nm", "als_450nm"]
        self.input_shape = (512, 512, 3)  # Default input shape

    def preprocess_image(self, image, light_source="white"):
        """
        Apply preprocessing specific to the light source
        """
        if light_source == "white":
            return self._preprocess_white_light(image)
        elif light_source == "als_415nm":
            return self._preprocess_als_415nm(image)
        elif light_source == "als_450nm":
            return self._preprocess_als_450nm(image)
        else:
            raise ValueError(f"Unsupported light source: {light_source}")

    def _preprocess_white_light(self, image):
        """
        Preprocessing for white light images
        """
        # Convert to float and normalize
        image_norm = image.astype(np.float32) / 255.0

        # Apply channel-specific processing
        processed = np.copy(image_norm)

        # Enhance contrast slightly
        processed = np.clip(processed * 1.2, 0.0, 1.0)

        return processed

    def _preprocess_als_415nm(self, image):
        """
        Preprocessing for ALS 415nm (violet) images
        """
        # Convert to float and normalize
        image_norm = image.astype(np.float32) / 255.0

        # Apply channel-specific processing for ALS 415nm
        # Enhance blue and green channels where bruises fluoresce
        processed = np.copy(image_norm)
        processed[:, :, 1] = np.clip(processed[:, :, 1] * 1.4, 0.0, 1.0)  # Green channel
        processed[:, :, 2] = np.clip(processed[:, :, 2] * 1.5, 0.0, 1.0)  # Blue channel

        return processed

    def _preprocess_als_450nm(self, image):
        """
        Preprocessing for ALS 450nm (blue) images
        """
        # Convert to float and normalize
        image_norm = image.astype(np.float32) / 255.0

        # Apply channel-specific processing for ALS 450nm
        # Enhance green channel where bruises fluoresce
        processed = np.copy(image_norm)
        processed[:, :, 1] = np.clip(processed[:, :, 1] * 1.6, 0.0, 1.0)  # Green channel

        return processed

    def detect_bruises(self, image, light_source="white", skin_tone=None):
        """
        Detect bruises in the image

        Parameters:
        - image: numpy array of image
        - light_source: type of illumination
        - skin_tone: optional Fitzpatrick skin type (1-6)

        Returns:
        - segmentation_mask: binary mask of detected bruises
        - confidence: confidence score for detection
        - metadata: additional detection information
        """
        # Preprocess the image
        preprocessed = self.preprocess_image(image, light_source)

        # This would normally call the actual model, but we'll simulate results
        # In a real implementation, this would use TensorFlow/PyTorch to run inference

        # Simulate creating a segmentation mask (would be model output in real system)
        segmentation_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # For simulation, create some "bruise-like" areas
        center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
        radius = min(image.shape[0], image.shape[1]) // 6

        # Create a simple circular "bruise" for demonstration
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                dist = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
                if dist < radius:
                    # Add some noise to make it look natural
                    if np.random.random() > 0.2:
                        segmentation_mask[i, j] = 255

        # Simulate confidence score based on light source and skin tone
        if skin_tone is not None:
            # Lower confidence for darker skin tones with white light
            if light_source == "white" and skin_tone >= 4:
                confidence = 0.7 - (skin_tone - 4) * 0.1
            # Higher confidence for ALS with darker skin
            elif light_source.startswith("als_") and skin_tone >= 4:
                confidence = 0.85 + (skin_tone - 4) * 0.03
            else:
                confidence = 0.85
        else:
            # Default confidence by light source
            if light_source == "white":
                confidence = 0.80
            else:
                confidence = 0.90

        # Add noise to confidence
        confidence = min(0.99, max(0.5, confidence + np.random.normal(0, 0.03)))

        # Metadata about the detection
        metadata = {
            "model_version": self.model_version,
            "light_source": light_source,
            "skin_tone": skin_tone,
            "processing_time_ms": np.random.randint(80, 150),
            "bruise_area_pixels": np.count_nonzero(segmentation_mask),
            "estimated_age_hours": np.random.randint(24, 96)
        }

        return segmentation_mask, confidence, metadata

    def estimate_bruise_age(self, image, mask, skin_tone=None):
        """
        Estimate the age of a bruise based on its appearance

        Parameters:
        - image: RGB image containing the bruise
        - mask: segmentation mask indicating bruise location
        - skin_tone: optional Fitzpatrick skin type (1-6)

        Returns:
        - age_estimate: estimated age in hours
        - confidence: confidence in the estimation
        - color_features: extracted color features used for aging
        """
        # Extract color features from the bruise area
        bruise_pixels = image[mask > 0]

        if len(bruise_pixels) == 0:
            return None, 0.0, {}

        # Calculate color statistics
        mean_color = np.mean(bruise_pixels, axis=0)
        std_color = np.std(bruise_pixels, axis=0)

        # Calculate color ratios (useful for bruise aging)
        r_g_ratio = mean_color[0] / max(1, mean_color[1])
        r_b_ratio = mean_color[0] / max(1, mean_color[2])
        g_b_ratio = mean_color[1] / max(1, mean_color[2])

        # In a real system, these features would be fed to a regression model
        # Here we'll use a simplified heuristic based on color ratios

        # Simplified bruise aging logic:
        # - Fresh bruises (0-24h): more red/purple (high R, low G, moderate B)
        # - Early bruises (24-72h): more blue/purple (low R, low G, high B)
        # - Mid-age bruises (72-144h): more green/yellow (low R, high G, low B)
        # - Old bruises (144h+): more yellow/brown (moderate R, high G, low B)

        age_estimate = 0
        if r_g_ratio > 1.2 and r_b_ratio > 0.8:
            # Fresh bruise
            age_estimate = np.random.randint(0, 24)
            confidence = 0.85
        elif r_g_ratio < 0.9 and g_b_ratio < 0.9:
            # Early bruise
            age_estimate = np.random.randint(24, 72)
            confidence = 0.80
        elif g_b_ratio > 1.2 and r_g_ratio < 0.9:
            # Mid-age bruise
            age_estimate = np.random.randint(72, 144)
            confidence = 0.75
        else:
            # Old bruise
            age_estimate = np.random.randint(144, 240)
            confidence = 0.65

        # Adjust confidence based on skin tone (more difficult on darker skin)
        if skin_tone is not None and skin_tone >= 4:
            confidence *= 0.9

        # Color features extracted
        color_features = {
            "mean_rgb": mean_color.tolist(),
            "std_rgb": std_color.tolist(),
            "r_g_ratio": r_g_ratio,
            "r_b_ratio": r_b_ratio,
            "g_b_ratio": g_b_ratio
        }

        return age_estimate, confidence, color_features


def preprocess_image(image, light_source="white", skin_tone=None):
    """
    Apply preprocessing optimized for bruise detection based on light source and skin tone

    Parameters:
    - image: PIL Image or numpy array
    - light_source: 'white', 'als_415nm', or 'als_450nm'
    - skin_tone: Fitzpatrick scale (1-6) if known

    Returns:
    - processed_image: preprocessed image ready for model input
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()

    # Apply bilateral filtering to reduce noise while preserving edges
    img_filtered = cv2.bilateralFilter(img_array, 9, 75, 75)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Convert to LAB color space for better processing
    img_lab = cv2.cvtColor(img_filtered, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_lab[:,:,0] = clahe.apply(img_lab[:,:,0])
    img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

    # Light source specific processing
    if light_source == "als_415nm":
        # For 415nm (violet), enhance blue and suppress red
        img_enhanced = img_enhanced.astype(np.float32)
        img_enhanced[:,:,2] = np.clip(img_enhanced[:,:,2] * 1.3, 0, 255)  # Enhance blue
        img_enhanced[:,:,0] = np.clip(img_enhanced[:,:,0] * 0.8, 0, 255)  # Suppress red
        img_enhanced = img_enhanced.astype(np.uint8)

    elif light_source == "als_450nm":
        # For 450nm (blue), enhance green which often shows bruise fluorescence
        img_enhanced = img_enhanced.astype(np.float32)
        img_enhanced[:,:,1] = np.clip(img_enhanced[:,:,1] * 1.3, 0, 255)  # Enhance green
        img_enhanced = img_enhanced.astype(np.uint8)

    # Skin tone specific adjustments (if provided)
    if skin_tone is not None:
        if skin_tone >= 4:  # Darker skin tones
            # Increase contrast more for darker skin
            img_enhanced = img_enhanced.astype(np.float32)
            # Calculate local mean (rough estimate of skin tone)
            local_mean = cv2.GaussianBlur(img_enhanced, (31, 31), 0)
            # Calculate difference to enhance local contrast
            detail = img_enhanced - local_mean
            # Enhance the details
            img_enhanced = local_mean + detail * 1.5
            img_enhanced = np.clip(img_enhanced, 0, 255).astype(np.uint8)

    return img_enhanced


def apply_als_filter(image, wavelength=415, filter_color="orange"):
    """
    Simulate the effect of an ALS (alternate light source) with optical filter

    Parameters:
    - image: PIL Image or numpy array
    - wavelength: ALS wavelength in nm (typically 415 or 450)
    - filter_color: color of optical filter used (typically orange for bruise detection)

    Returns:
    - als_image: simulated ALS image
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()

    # Create a simulated ALS effect based on wavelength
    als_image = img_array.copy().astype(np.float32)

    if wavelength == 415:  # Violet light
        # Suppress red, enhance blue
        als_image[:,:,0] *= 0.5  # Reduce red
        als_image[:,:,2] *= 1.5  # Enhance blue
    elif wavelength == 450:  # Blue light
        # Suppress red, enhance green
        als_image[:,:,0] *= 0.6  # Reduce red
        als_image[:,:,1] *= 1.4  # Enhance green

    # Apply optical filter effect
    if filter_color == "orange":
        # Orange filter blocks blue, passes red and green
        als_image[:,:,2] *= 0.3  # Reduce blue
        als_image[:,:,0] *= 1.2  # Enhance red
    elif filter_color == "yellow":
        # Yellow filter partially blocks blue
        als_image[:,:,2] *= 0.5  # Reduce blue

    # Add wavelength-specific fluorescence effect for bruises
    # Create a synthetic bruise-like area
    height, width = als_image.shape[:2]
    center_x, center_y = width // 2, height // 2
    radius = min(height, width) // 6

    # Create a mask for the bruise area
    bruise_mask = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            dist = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
            if dist < radius:
                bruise_mask[i, j] = max(0, 1 - (dist / radius) ** 2)

    # Apply fluorescence effect based on wavelength
    if wavelength == 415:  # Violet light
        als_image[:,:,0] += bruise_mask[:,:,np.newaxis] * 40  # Add red fluorescence
        als_image[:,:,1] += bruise_mask[:,:,np.newaxis] * 30  # Add green fluorescence
    elif wavelength == 450:  # Blue light
        als_image[:,:,1] += bruise_mask[:,:,np.newaxis] * 60  # Add green fluorescence

    # Clip values and convert back to uint8
    als_image = np.clip(als_image, 0, 255).astype(np.uint8)

    return als_image
