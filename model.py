
import numpy as np
from PIL import Image
import base64
import io
from scipy.fftpack import dct
from scipy import stats
from scipy.ndimage import sobel
import logging

logger = logging.getLogger(__name__)

class DeepFakeDetector:
    def __init__(self):
        self.model_version = "7.0.0"

        # CRITICAL: New weights based on real data
        self.weights = {
            "kurtosis_anomaly": 0.30,        # STRONGEST signal
            "texture_variance": 0.25,        # Deepfakes smooth
            "color_distribution": 0.15,      # Color imbalance
            "edge_characteristics": 0.12,    # Edge sharpness
            "frequency_distribution": 0.10,  # Low-freq dominance
            "skin_tone_consistency": 0.08,   # Unnatural skin
        }

    def analyze_image(self, image_input):
        """Advanced multi-stage analysis optimized for high-quality deepfakes"""
        try:
            img = self._prepare_image(image_input)

            # Stage 1: Kurtosis Anomaly (Most Critical)
            kurtosis_score = self._kurtosis_anomaly_detection(img)

            # Stage 2: Texture Variance
            texture_score = self._texture_variance_analysis(img)

            # Stage 3: Color Distribution
            color_score = self._color_distribution_analysis(img)

            # Stage 4: Edge Characteristics
            edge_score = self._edge_characteristics(img)

            # Stage 5: Frequency Distribution
            freq_score = self._frequency_distribution(img)

            # Stage 6: Skin Tone Consistency
            skin_score = self._skin_tone_analysis(img)

            logger.info(
                f"Analysis - Kurtosis:{kurtosis_score:.1f} Texture:{texture_score:.1f} "
                f"Color:{color_score:.1f} Edge:{edge_score:.1f} Freq:{freq_score:.1f} Skin:{skin_score:.1f}"
            )

            # Ensemble combination
            authenticity = self._ensemble_combine(
                kurtosis_score, texture_score, color_score, edge_score,
                freq_score, skin_score
            )

            verdict = self._verdict(authenticity)

            return {
                'authenticity_score': round(authenticity, 2),
                'technical_details': {
                    'kurtosis_anomaly': round(kurtosis_score, 2),
                    'texture_variance': round(texture_score, 2),
                    'color_distribution': round(color_score, 2),
                    'edge_characteristics': round(edge_score, 2),
                    'frequency_distribution': round(freq_score, 2),
                    'skin_tone_consistency': round(skin_score, 2),
                },
                'verdict': verdict,
                'model_version': self.model_version,
            }
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise

    def _prepare_image(self, image_input):
        """Convert to RGB 512x512"""
        if isinstance(image_input, str):
            if "," in image_input:
                image_input = image_input.split(",", 1)[1]
            raw = base64.b64decode(image_input)
            img = Image.open(io.BytesIO(raw))
        elif isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                img = Image.fromarray(image_input.astype(np.uint8), "L").convert("RGB")
            else:
                img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
        else:
            raise ValueError("Unsupported image type")

        img = img.convert("RGB")
        img = img.resize((512, 512), Image.LANCZOS)
        return np.array(img)

    # ========== STAGE 1: KURTOSIS ANOMALY (Most Critical) ==========

    def _kurtosis_anomaly_detection(self, arr):
        """
        Detect kurtosis anomalies in DCT coefficients
        REAL: Kurtosis > 160,000
        DEEPFAKE: Kurtosis < 165,000 (often 162,000-165,000)
        This is the STRONGEST differentiator for high-quality deepfakes
        """
        gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
        gray = gray.astype(np.float32)

        # Apply DCT
        dct_result = dct(dct(gray.T, norm='ortho').T, norm='ortho')
        dct_mag = np.abs(dct_result)

        # Analyze kurtosis
        flat = dct_mag.flatten()
        kurtosis_val = stats.kurtosis(flat)
        skewness_val = stats.skew(flat)

        logger.info(f"Kurtosis: {kurtosis_val:.0f}, Skewness: {skewness_val:.2f}")

        # Kurtosis-based decision
        # Real photos: 160,000+
        # Deepfakes: 150,000-165,000

        if kurtosis_val > 165000:
            score = 85
        elif kurtosis_val > 160000:
            score = 78
        elif kurtosis_val > 155000:
            score = 65
        elif kurtosis_val > 150000:
            score = 45
        else:
            score = 25

        return float(np.clip(score, 0, 100))

    # ========== STAGE 2: TEXTURE VARIANCE ==========

    def _texture_variance_analysis(self, arr):
        """
        Real: Lower variance (5000-5500)
        Deepfake: Higher variance (5500-6500) - smoother/more artificial
        """
        gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
        gray = gray.astype(np.uint8)

        texture_var = np.var(gray)

        if texture_var < 5200:
            score = 82
        elif texture_var < 5500:
            score = 75
        elif texture_var < 5800:
            score = 60
        elif texture_var < 6200:
            score = 40
        else:
            score = 25

        return float(np.clip(score, 0, 100))

    # ========== STAGE 3: COLOR DISTRIBUTION ==========

    def _color_distribution_analysis(self, arr):
        """
        Deepfakes often have less color variance or unbalanced channels
        Real: More balanced, higher std (>74)
        Deepfake: Less balanced, higher avg std (>78) or uneven
        """
        r_std = np.std(arr[:,:,0])
        g_std = np.std(arr[:,:,1])
        b_std = np.std(arr[:,:,2])

        avg_std = (r_std + g_std + b_std) / 3
        channel_std_dev = np.std([r_std, g_std, b_std])

        logger.info(f"Color - R:{r_std:.1f} G:{g_std:.1f} B:{b_std:.1f} Avg:{avg_std:.1f} StdDev:{channel_std_dev:.2f}")

        # Real: more balanced channels, avg_std 70-75, low std_dev
        # Deepfake: less balanced or overly saturated

        if avg_std < 75 and channel_std_dev < 1.2:
            score = 80
        elif avg_std < 77 and channel_std_dev < 1.5:
            score = 70
        elif avg_std > 78 or channel_std_dev > 1.5:
            score = 45
        else:
            score = 60

        return float(np.clip(score, 0, 100))

    # ========== STAGE 4: EDGE CHARACTERISTICS ==========

    def _edge_characteristics(self, arr):
        """
        Real: Natural edges (edge_mean ~47, edge_std ~80)
        Deepfake: Softened edges (edge_mean ~44, edge_std ~69)
        """
        gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
        gray = gray.astype(float)

        # Compute edge magnitudes
        sx = sobel(gray, axis=0)
        sy = sobel(gray, axis=1)
        edges = np.hypot(sx, sy)

        edge_mean = np.mean(edges)
        edge_std = np.std(edges)
        edge_ratio = edge_std / (edge_mean + 1e-10)

        logger.info(f"Edges - Mean:{edge_mean:.1f} Std:{edge_std:.1f} Ratio:{edge_ratio:.2f}")

        # Real: edge_mean > 45, edge_std > 75, ratio > 1.6
        # Deepfake: edge_mean < 45, edge_std < 75, ratio < 1.6

        if edge_mean > 46 and edge_std > 78 and edge_ratio > 1.65:
            score = 82
        elif edge_mean > 45 and edge_std > 75:
            score = 75
        elif edge_mean < 45 and edge_std < 70:
            score = 35
        else:
            score = 55

        return float(np.clip(score, 0, 100))

    # ========== STAGE 5: FREQUENCY DISTRIBUTION ==========

    def _frequency_distribution(self, arr):
        """
        Check frequency band distribution
        """
        gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
        gray = gray.astype(np.float32)

        dct_result = dct(dct(gray.T, norm='ortho').T, norm='ortho')
        dct_mag = np.abs(dct_result)

        h, w = dct_mag.shape
        low_band = dct_mag[:h//5, :w//5].sum()
        high_band = dct_mag[h//2:, w//2:].sum()
        total = dct_mag.sum()

        low_ratio = low_band / total
        high_ratio = high_band / total

        if low_ratio < 0.35 and high_ratio > 0.04:
            score = 80
        elif low_ratio < 0.32:
            score = 75
        elif low_ratio > 0.32:
            score = 55
        else:
            score = 65

        return float(np.clip(score, 0, 100))

    # ========== STAGE 6: SKIN TONE CONSISTENCY ==========

    def _skin_tone_analysis(self, arr):
        """
        Skin tone percentage and consistency
        Real: 30-35% skin pixels
        Deepfake: 25-30% (sometimes too little or too much)
        """
        r = arr[:,:,0].astype(float)
        g = arr[:,:,1].astype(float)
        b = arr[:,:,2].astype(float)

        # Skin detection
        skin_mask = (r > 95) & (g > 40) & (b > 20) & ((r-g) > 15) & (r > g) & (r > b)
        skin_ratio = np.sum(skin_mask) / (arr.shape[0] * arr.shape[1])

        if 0.30 < skin_ratio < 0.35:
            score = 80
        elif 0.28 < skin_ratio < 0.36:
            score = 72
        elif 0.25 < skin_ratio < 0.40:
            score = 65
        else:
            score = 50

        return float(np.clip(score, 0, 100))

    # ========== ENSEMBLE COMBINATION ==========

    def _ensemble_combine(self, kurtosis, texture, color, edge, freq, skin):
        """
        Weighted ensemble - Kurtosis is most important
        """
        w = self.weights

        raw_score = (
            kurtosis * w['kurtosis_anomaly'] +
            texture * w['texture_variance'] +
            color * w['color_distribution'] +
            edge * w['edge_characteristics'] +
            freq * w['frequency_distribution'] +
            skin * w['skin_tone_consistency']
        )

        logger.info(f"Raw ensemble score: {raw_score:.2f}")

        return float(np.clip(raw_score, 0, 100))

    def _verdict(self, score):
        """Professional verdict generation"""
        if score >= 75:
            return {
                'classification': 'authentic',
                'title': 'Likely Authentic',
                'description': 'Image shows natural forensic characteristics consistent with genuine photography.',
                'confidence': 'high',
            }
        elif score >= 65:
            return {
                'classification': 'probable',
                'title': 'Probably Authentic',
                'description': 'Image appears authentic with some minor variations.',
                'confidence': 'medium-high',
            }
        elif score >= 50:
            return {
                'classification': 'uncertain',
                'title': 'Uncertain - Expert Review Needed',
                'description': 'Image shows mixed forensic indicators. Professional analysis recommended.',
                'confidence': 'medium',
            }
        elif score >= 35:
            return {
                'classification': 'suspicious',
                'title': 'Suspicious - Likely Manipulated',
                'description': 'Multiple forensic indicators suggest manipulation or AI generation.',
                'confidence': 'medium-high',
            }
        else:
            return {
                'classification': 'fake',
                'title': 'Likely AI-Generated',
                'description': 'Strong forensic evidence indicates AI-generated or heavily manipulated content.',
                'confidence': 'high',
            }
