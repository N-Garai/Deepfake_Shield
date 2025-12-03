"""
DeepFake Detection Model
Advanced image analysis for detecting AI-generated or manipulated images
"""

import numpy as np
from PIL import Image
import io
import base64
from scipy import ndimage
from scipy.fftpack import dct
import cv2


class DeepFakeDetector:
    """
    AI-powered deepfake detection model using multiple forensic techniques
    """
    
    def __init__(self):
        self.model_version = "1.0.0"
        
    def analyze_image(self, image_data):
        """
        Main analysis function that processes an image and returns detection results
        
        Args:
            image_data: Image data (PIL Image, numpy array, or base64 string)
            
        Returns:
            dict: Analysis results with scores and technical details
        """
        # Convert image to numpy array
        img_array = self._prepare_image(image_data)
        
        print(f"\n[DEBUG] Image shape: {img_array.shape}")
        
        # Run multiple detection algorithms
        compression_score = self._analyze_compression_artifacts(img_array)
        print(f"[DEBUG] Compression score: {compression_score:.2f}")
        
        edge_score = self._analyze_edge_consistency(img_array)
        print(f"[DEBUG] Edge score: {edge_score:.2f}")
        
        noise_score = self._analyze_noise_patterns(img_array)
        print(f"[DEBUG] Noise score: {noise_score:.2f}")
        
        color_score = self._analyze_color_distribution(img_array)
        print(f"[DEBUG] Color score: {color_score:.2f}")
        
        frequency_score = self._analyze_frequency_domain(img_array)
        print(f"[DEBUG] Frequency score: {frequency_score:.2f}")
        
        # Calculate weighted authenticity score
        authenticity_score = self._calculate_authenticity_score(
            compression_score,
            edge_score,
            noise_score,
            color_score,
            frequency_score
        )
        
        print(f"[DEBUG] Final authenticity score: {authenticity_score:.2f}\n")
        
        # Generate verdict
        verdict = self._generate_verdict(authenticity_score)
        
        return {
            'authenticity_score': round(authenticity_score, 2),
            'technical_details': {
                'compression_artifacts': round(compression_score, 2),
                'edge_consistency': round(edge_score, 2),
                'noise_patterns': round(noise_score, 2),
                'color_distribution': round(color_score, 2),
                'frequency_analysis': round(frequency_score, 2)
            },
            'verdict': verdict,
            'model_version': self.model_version
        }
    
    def _prepare_image(self, image_data):
        """Convert various image formats to numpy array"""
        if isinstance(image_data, np.ndarray):
            return image_data
        elif isinstance(image_data, Image.Image):
            return np.array(image_data)
        elif isinstance(image_data, str):
            # Assume base64 encoded
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            image = Image.open(io.BytesIO(image_bytes))
            return np.array(image)
        else:
            raise ValueError("Unsupported image format")
    
    def _analyze_compression_artifacts(self, img_array):
        """
        Analyze JPEG compression artifacts
        AI-generated images often have unusual compression patterns
        """
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Detect block boundaries (8x8 JPEG blocks)
        block_size = 8
        artifact_scores = []
        boundary_discontinuities = []
        
        for y in range(0, gray.shape[0] - block_size, block_size):
            for x in range(0, gray.shape[1] - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                
                # Calculate block variance
                variance = np.var(block)
                
                if variance > 10:  # Ignore flat blocks
                    # Check for blocky artifacts at boundaries
                    if x + block_size < gray.shape[1]:
                        right_block = gray[y:y+block_size, x+block_size:x+block_size*2]
                        if right_block.shape[1] == block_size:
                            boundary_diff = np.mean(np.abs(block[:, -1].astype(float) - right_block[:, 0].astype(float)))
                            boundary_discontinuities.append(boundary_diff)
                    
                    # Measure smoothness within block
                    horizontal_diff = np.mean(np.abs(np.diff(block, axis=1)))
                    vertical_diff = np.mean(np.abs(np.diff(block, axis=0)))
                    artifact_scores.append((horizontal_diff + vertical_diff) / 2)
        
        if len(artifact_scores) > 0 and len(boundary_discontinuities) > 0:
            avg_artifact = np.mean(artifact_scores)
            avg_boundary = np.mean(boundary_discontinuities)
            std_artifact = np.std(artifact_scores)
            
            # Real photos: higher boundary discontinuity, more variation
            # AI images: smoother boundaries, very uniform blocks
            boundary_ratio = avg_boundary / (avg_artifact + 1)
            uniformity = std_artifact / (avg_artifact + 1)
            
            # Higher score = more authentic (real photo characteristics)
            if boundary_ratio > 1.5 and uniformity > 0.5:
                score = 75 + min(25, boundary_ratio * 8)  # Clear real photo
            elif boundary_ratio < 0.5 and uniformity < 0.3:
                score = 15 + boundary_ratio * 30  # Clear AI (too smooth/uniform)
            elif boundary_ratio > 1.0:
                score = 60 + (boundary_ratio - 1.0) * 20
            elif boundary_ratio > 0.6:
                score = 45 + (boundary_ratio - 0.6) * 35
            else:
                score = 25 + boundary_ratio * 33
        else:
            score = 50  # Default to neutral if can't determine
        
        return min(100, max(0, score))
    
    def _analyze_edge_consistency(self, img_array):
        """
        Analyze edge consistency and sharpness
        AI-generated images often have inconsistent edges
        """
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate edge magnitude
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Analyze edge consistency
        edge_threshold = np.percentile(edge_magnitude, 90)
        strong_edges = edge_magnitude > edge_threshold
        
        # Check for unnatural edge patterns
        edge_variance = np.var(edge_magnitude[strong_edges]) if np.any(strong_edges) else 0
        edge_mean = np.mean(edge_magnitude[strong_edges]) if np.any(strong_edges) else 0
        
        # Natural images have higher edge variance
        # AI images have overly consistent, smooth edges
        if edge_mean > 0:
            consistency_ratio = edge_variance / edge_mean
            
            print(f"  [EDGE DEBUG] variance={edge_variance:.2f}, mean={edge_mean:.2f}, ratio={consistency_ratio:.3f}")
            
            # Real photos: consistency_ratio usually > 15
            # AI images: consistency_ratio usually < 8
            if consistency_ratio > 20:
                score = 90 + min(10, (consistency_ratio - 20) * 0.5)
            elif consistency_ratio > 12:
                score = 70 + (consistency_ratio - 12) * 2.5
            elif consistency_ratio > 6:
                score = 45 + (consistency_ratio - 6) * 4
            elif consistency_ratio > 3:
                score = 25 + (consistency_ratio - 3) * 6.5
            else:
                score = 10 + consistency_ratio * 5  # Very consistent edges - AI
        else:
            score = 50
        
        return min(100, max(0, score))
    
    def _analyze_noise_patterns(self, img_array):
        """
        Analyze noise patterns in the image
        AI-generated images often have unnaturally uniform noise
        """
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply high-pass filter to extract noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray.astype(float) - blurred.astype(float)
        
        # Analyze noise statistics
        noise_std = np.std(noise)
        noise_mean = np.abs(np.mean(noise))
        
        # Calculate local noise variance across multiple patch sizes
        kernel_sizes = [8, 16, 32]
        local_vars_all = []
        
        for kernel_size in kernel_sizes:
            local_vars = []
            for y in range(0, gray.shape[0] - kernel_size, kernel_size):
                for x in range(0, gray.shape[1] - kernel_size, kernel_size):
                    local_noise = noise[y:y+kernel_size, x:x+kernel_size]
                    local_vars.append(np.var(local_noise))
            if local_vars:
                local_vars_all.extend(local_vars)
        
        # Natural images have more varied local noise
        variance_of_variance = np.var(local_vars_all) if local_vars_all else 0
        mean_local_var = np.mean(local_vars_all) if local_vars_all else 0
        
        # Calculate coefficient of variation for noise
        noise_cv = (variance_of_variance ** 0.5) / (mean_local_var + 0.001)
        
        # Score based on noise characteristics
        # Real photos: higher noise std, higher variance of variance
        # AI images: very low noise or overly uniform noise
        
        print(f"  [NOISE DEBUG] std={noise_std:.3f}, cv={noise_cv:.3f}, mean_var={mean_local_var:.3f}")
        
        if noise_std < 1.0:
            # Extremely clean - very likely AI
            score = 10 + noise_std * 15
        elif noise_std < 3.0:
            # Very clean - check uniformity carefully
            if noise_cv < 0.5:
                score = 20 + noise_cv * 30  # Uniform and clean (AI)
            elif noise_cv < 1.0:
                score = 35 + (noise_cv - 0.5) * 40  # Somewhat uniform
            else:
                score = 55 + (noise_cv - 1.0) * 25  # Clean but varied
        elif noise_std > 12:
            score = 85 + min(15, (noise_std - 12) * 2)  # Strong natural sensor noise
        elif noise_std > 6:
            # Good amount of noise - check variation
            if noise_cv > 1.5:
                score = 80 + min(15, (noise_cv - 1.5) * 10)  # Highly varied (real)
            elif noise_cv > 0.8:
                score = 65 + (noise_cv - 0.8) * 21
            else:
                score = 50 + noise_cv * 18
        else:
            # Moderate noise (3-6) - most critical range
            if noise_cv > 1.2:
                score = 70 + min(20, (noise_cv - 1.2) * 15)  # Varied noise (real)
            elif noise_cv > 0.7:
                score = 50 + (noise_cv - 0.7) * 40
            elif noise_cv > 0.4:
                score = 35 + (noise_cv - 0.4) * 50
            else:
                score = 20 + noise_cv * 37  # Very uniform (AI)
        
        return min(100, max(0, score))
    
    def _analyze_color_distribution(self, img_array):
        """
        Analyze color distribution patterns
        AI-generated images may have unnatural color distributions
        """
        if len(img_array.shape) != 3:
            return 50  # Can't analyze grayscale
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Analyze hue distribution
        hue = hsv[:, :, 0]
        hue_hist, _ = np.histogram(hue, bins=180, range=(0, 180))
        hue_entropy = self._calculate_entropy(hue_hist)
        
        # Analyze saturation
        sat = hsv[:, :, 1]
        sat_mean = np.mean(sat)
        sat_std = np.std(sat)
        
        # Analyze luminance distribution
        luminance = lab[:, :, 0]
        lum_hist, _ = np.histogram(luminance, bins=256)
        lum_entropy = self._calculate_entropy(lum_hist)
        
        # Natural images have balanced entropy and variation
        entropy_score = (hue_entropy + lum_entropy) / 2
        variation_score = min(100, (sat_std / 255) * 200)
        
        # Combine scores with more weight on entropy
        score = (entropy_score * 12 + variation_score) / 2
        
        # Color distribution alone is not a strong indicator, be generous
        score = max(50, score)  # Minimum 50% for color
        
        return min(100, max(50, score))
    
    def _analyze_frequency_domain(self, img_array):
        """
        Analyze frequency domain characteristics using DCT
        AI-generated images often have unusual frequency patterns
        """
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Resize for efficient processing
        gray_small = cv2.resize(gray, (256, 256)).astype(float)
        
        # Apply DCT
        dct_img = dct(dct(gray_small.T, norm='ortho').T, norm='ortho')
        
        # Analyze frequency components in more detail
        low_freq = np.abs(dct_img[:64, :64])
        mid_freq = np.abs(dct_img[64:128, 64:128])
        high_freq = np.abs(dct_img[128:, 128:])
        very_high_freq = np.abs(dct_img[192:, 192:])
        
        low_energy = np.sum(low_freq)
        mid_energy = np.sum(mid_freq)
        high_energy = np.sum(high_freq)
        very_high_energy = np.sum(very_high_freq)
        
        total_energy = low_energy + mid_energy + high_energy + very_high_energy
        
        if total_energy > 0:
            low_ratio = low_energy / total_energy
            high_ratio = high_energy / total_energy
            very_high_ratio = very_high_energy / total_energy
            mid_ratio = mid_energy / total_energy
            
            # Calculate frequency distribution entropy
            freq_dist = np.array([low_ratio, mid_ratio, high_ratio, very_high_ratio])
            freq_entropy = -np.sum(freq_dist * np.log2(freq_dist + 1e-10))
            
            print(f"  [FREQ DEBUG] low={low_ratio:.3f}, vhigh={very_high_ratio:.4f}, entropy={freq_entropy:.3f}")
            
            # AI images tend to have:
            # 1. Too much low frequency (overly smooth) - low_ratio > 0.85
            # 2. Too little high frequency (lack of sensor noise) - very_high_ratio < 0.02
            # 3. Lower frequency entropy (less varied)
            
            # Real photos typically: low_ratio 0.55-0.80, very_high_ratio > 0.025
            if low_ratio > 0.90:
                score = 5  # Extremely smooth - definitely AI
            elif low_ratio > 0.85:
                score = 15 + (0.90 - low_ratio) * 200  # Very smooth - AI
            elif low_ratio > 0.80:
                # Check high freq to confirm
                if very_high_ratio < 0.015:
                    score = 25  # Smooth with no noise - AI
                else:
                    score = 35 + very_high_ratio * 500
            elif low_ratio < 0.50:
                score = 95  # Excellent detail distribution - real
            elif low_ratio < 0.65:
                # Good balance - check high freq
                if very_high_ratio > 0.04:
                    score = 85 + min(15, very_high_ratio * 250)  # Strong high freq - real
                elif very_high_ratio > 0.025:
                    score = 70 + (very_high_ratio - 0.025) * 1000
                else:
                    score = 50 + very_high_ratio * 800
            else:
                # Moderate low freq (0.65-0.80) - check high frequency carefully
                if very_high_ratio > 0.035:
                    score = 75 + min(20, very_high_ratio * 400)  # Good high freq
                elif very_high_ratio > 0.020:
                    score = 55 + (very_high_ratio - 0.020) * 1300
                elif very_high_ratio > 0.010:
                    score = 35 + (very_high_ratio - 0.010) * 2000
                else:
                    score = 15 + very_high_ratio * 2000  # Almost no high freq - AI
            
            # Apply entropy adjustment (lower entropy = more AI-like)
            if freq_entropy < 1.2:
                score = score * 0.75  # Penalize low entropy
            elif freq_entropy < 1.5:
                score = score * 0.9
        else:
            score = 50
        
        return min(100, max(0, score))
    
    def _calculate_entropy(self, histogram):
        """Calculate Shannon entropy of histogram"""
        histogram = histogram[histogram > 0]
        histogram = histogram / np.sum(histogram)
        entropy = -np.sum(histogram * np.log2(histogram))
        return entropy
    
    def _calculate_authenticity_score(self, compression, edge, noise, color, frequency):
        """
        Calculate weighted authenticity score from individual metrics
        Higher score = more likely to be authentic
        """
        # Adjusted weights - noise and frequency are strongest indicators
        weights = {
            'compression': 0.15,
            'edge': 0.20,
            'noise': 0.35,  # Increased - strongest indicator
            'color': 0.10,
            'frequency': 0.20  # Increased - strong indicator
        }
        
        score = (
            compression * weights['compression'] +
            edge * weights['edge'] +
            noise * weights['noise'] +
            color * weights['color'] +
            frequency * weights['frequency']
        )
        
        # Apply curve adjustment to make scoring less aggressive
        # This gives benefit of doubt to borderline cases
        if score >= 60:
            # Boost authentic-leaning scores
            score = 60 + (score - 60) * 1.2
        elif score <= 35:
            # Keep clearly fake scores low
            score = score * 0.9
        
        return min(100, max(0, score))
    
    def _generate_verdict(self, authenticity_score):
        """Generate human-readable verdict"""
        if authenticity_score >= 65:
            return {
                'classification': 'authentic',
                'title': 'Likely Authentic',
                'description': 'This image shows strong indicators of being genuine with natural characteristics typical of real photographs.',
                'confidence': 'high'
            }
        elif authenticity_score >= 50:
            return {
                'classification': 'suspicious',
                'title': 'Possibly Authentic',
                'description': 'This image shows mostly authentic characteristics with some minor inconsistencies. Likely a real photo with compression or editing.',
                'confidence': 'medium'
            }
        elif authenticity_score >= 35:
            return {
                'classification': 'suspicious',
                'title': 'Suspicious',
                'description': 'This image shows mixed indicators. Some characteristics suggest authenticity while others raise concerns about manipulation.',
                'confidence': 'medium'
            }
        else:
            return {
                'classification': 'fake',
                'title': 'Likely AI-Generated',
                'description': 'This image shows strong indicators of artificial generation or heavy manipulation.',
                'confidence': 'high'
            }


# Utility functions for batch processing
def analyze_image_from_path(image_path):
    """Analyze image from file path"""
    detector = DeepFakeDetector()
    image = Image.open(image_path)
    return detector.analyze_image(image)


def analyze_image_from_base64(base64_string):
    """Analyze image from base64 string"""
    detector = DeepFakeDetector()
    return detector.analyze_image(base64_string)


if __name__ == "__main__":
    # Test the model
    print("DeepFake Detection Model v1.0.0")
    print("Ready for image analysis")
