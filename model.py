import numpy as np
from PIL import Image
import base64
import io
import cv2
from scipy.fftpack import dct
import logging

logger = logging.getLogger(__name__)

class DeepFakeDetector:
    def __init__(self):
        self.model_version = "5.0.0"
        # PERFECTLY BALANCED WEIGHTS
        self.weights = {
            "compression": 0.12,    # 12%
            "edge": 0.12,           # 12%
            "noise": 0.24,          # 24% - Important but not overpowering
            "color": 0.12,          # 12%
            "frequency": 0.40,      # 40% - Important but not aggressive
        }

    def analyze_image(self, image_input):
        """Main analysis function"""
        try:
            img = self._prepare_image(image_input)

            # Run ALL metrics with balanced algorithms
            compression = self._compression_score_v5(img)
            edge = self._edge_score_v5(img)
            noise = self._noise_score_v5(img)
            color = self._color_score_v5(img)
            freq = self._frequency_score_v5(img)

            logger.info(
                f"Scores - Comp:{compression:.1f} Edge:{edge:.1f} "
                f"Noise:{noise:.1f} Color:{color:.1f} Freq:{freq:.1f}"
            )

            authenticity = self._combine_scores_v5(compression, edge, noise, color, freq)
            verdict = self._verdict(authenticity)

            return {
                'authenticity_score': round(authenticity, 2),
                'technical_details': {
                    'compression_artifacts': round(compression, 2),
                    'edge_consistency': round(edge, 2),
                    'noise_patterns': round(noise, 2),
                    'color_distribution': round(color, 2),
                    'frequency_analysis': round(freq, 2),
                },
                'verdict': verdict,
                'model_version': self.model_version,
            }
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise

    def _prepare_image(self, image_input):
        """Convert to RGB 512x512 numpy array"""
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

    # ========== v5.0 ALGORITHMS - PERFECTLY BALANCED ==========

    def _compression_score_v5(self, arr):
        """
        BALANCED compression artifact detection
        Real: Natural variation with visible block boundaries
        AI: Unnaturally smooth with subtle/missing boundaries
        """
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)

        block_size = 8
        block_variances = []
        boundary_diffs = []

        for y in range(0, gray.shape[0] - block_size, block_size):
            for x in range(0, gray.shape[1] - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                bv = np.var(block)
                block_variances.append(bv)

                # Right boundary difference
                if x + block_size * 2 <= gray.shape[1]:
                    right = gray[y:y+block_size, x+block_size:x+block_size*2]
                    bd = np.abs(np.mean(block[:,-1]) - np.mean(right[:,0]))
                    boundary_diffs.append(bd)

        if not block_variances or not boundary_diffs:
            return 50.0

        bv_array = np.array(block_variances)
        bd_array = np.array(boundary_diffs)

        within_var = np.var(bv_array)
        mean_var = np.mean(bv_array)
        mean_boundary = np.mean(bd_array)
        std_boundary = np.std(bd_array)

        if mean_var > 0:
            uniformity = within_var / mean_var
        else:
            uniformity = 0

        # BALANCED SCORING - not too aggressive
        if uniformity > 0.4 and mean_boundary > 2.5:
            score = 82  # Clear real
        elif uniformity > 0.25 and mean_boundary > 1.8:
            score = 70
        elif uniformity > 0.15 and mean_boundary > 1.0:
            score = 58
        elif uniformity > 0.08 and mean_boundary > 0.7:
            score = 48
        elif mean_var < 1.5 and std_boundary < 1.5:  # Very smooth
            score = 28
        elif uniformity < 0.05 and mean_boundary < 1.2:
            score = 35
        else:
            score = 50

        return float(np.clip(score, 0, 100))

    def _edge_score_v5(self, arr):
        """
        BALANCED edge detection - multi-scale
        Real: Varied edges with natural complexity
        AI: More uniform edges but not always suspicious
        """
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)

        edge_scores = []

        for ksize in [3, 5, 7]:
            sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
            sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
            mag = np.sqrt(sx**2 + sy**2 + 1e-6)

            threshold = np.percentile(mag, 80)
            strong_edges = mag[mag > threshold]

            if len(strong_edges) > 50:
                mean_edge = np.mean(strong_edges)
                var_edge = np.var(strong_edges)

                if mean_edge > 0:
                    ratio = var_edge / (mean_edge**2)

                    # BALANCED - not overly strict
                    if ratio > 1.2:
                        edge_scores.append(85)
                    elif ratio > 0.8:
                        edge_scores.append(72)
                    elif ratio > 0.5:
                        edge_scores.append(60)
                    elif ratio > 0.25:
                        edge_scores.append(45)
                    elif ratio > 0.1:
                        edge_scores.append(32)
                    else:
                        edge_scores.append(20)

        if edge_scores:
            score = np.mean(edge_scores)
        else:
            score = 50

        return float(np.clip(score, 0, 100))

    def _noise_score_v5(self, arr):
        """
        BALANCED noise analysis - key differentiator but fair
        Real: Significant natural noise with variation
        AI: Lower noise levels but not always disqualifying
        """
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)

        noise_levels = []
        uniformity_scores = []
        laplace_scores = []

        for blur_sigma in [0.5, 1.0, 1.5]:
            blurred = cv2.GaussianBlur(gray, (15, 15), blur_sigma)
            noise = np.abs(gray - blurred)

            noise_std = np.std(noise)
            noise_levels.append(noise_std)

            # Laplacian for texture details
            laplacian = cv2.Laplacian(gray, cv2.CV_32F)
            laplace_std = np.std(laplacian)
            laplace_scores.append(laplace_std)

            # Local noise variance (uniformity)
            patch_vars = []
            for y in range(0, gray.shape[0]-16, 32):
                for x in range(0, gray.shape[1]-16, 32):
                    patch_noise = noise[y:y+16, x:x+16]
                    patch_vars.append(np.var(patch_noise))

            if patch_vars:
                pv_array = np.array(patch_vars)
                uniformity = np.std(pv_array) / (np.mean(pv_array) + 0.1)
                uniformity_scores.append(uniformity)

        avg_noise = np.mean(noise_levels)
        avg_uniformity = np.mean(uniformity_scores) if uniformity_scores else 0
        avg_laplace = np.mean(laplace_scores)

        logger.info(f"Noise - Avg:{avg_noise:.2f} Uniformity:{avg_uniformity:.2f} Laplace:{avg_laplace:.2f}")

        # BALANCED SCORING
        # Real photos
        if avg_noise > 6.0 and avg_uniformity > 0.7 and avg_laplace > 8:
            noise_score = 88
        elif avg_noise > 4.5 and avg_uniformity > 0.5 and avg_laplace > 6:
            noise_score = 78
        elif avg_noise > 3.0 and avg_uniformity > 0.3 and avg_laplace > 4:
            noise_score = 68
        elif avg_noise > 1.8 and avg_uniformity > 0.15 and avg_laplace > 2:
            noise_score = 58
        # AI/Synthetic
        elif avg_noise < 0.5 and avg_laplace < 2:
            noise_score = 18  # Very clean = likely AI
        elif avg_noise < 1.0 and avg_uniformity < 0.15:
            noise_score = 28
        elif avg_noise < 1.5 and avg_uniformity < 0.25:
            noise_score = 38
        elif avg_noise < 2.5 and avg_uniformity < 0.35:
            noise_score = 48
        else:
            noise_score = 55

        return float(np.clip(noise_score, 0, 100))

    def _color_score_v5(self, arr):
        """
        BALANCED color analysis
        Real: Complex, varied color patterns with high entropy
        AI: More uniform colors but not always suspicious
        """
        if arr.ndim != 3:
            return 50.0

        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)

        h = hsv[:,:,0]
        s = hsv[:,:,1]
        v = hsv[:,:,2]

        l = lab[:,:,0]
        a = lab[:,:,1]
        b = lab[:,:,2]

        h_std = np.std(h)
        s_std = np.std(s)
        v_std = np.std(v)

        l_std = np.std(l)
        a_std = np.std(a)
        b_std = np.std(b)

        # Histograms
        h_hist = np.histogram(h, bins=32)[0]
        s_hist = np.histogram(s, bins=32)[0]

        h_entropy = self._entropy(h_hist)
        s_entropy = self._entropy(s_hist)

        complexity = (h_std + s_std + v_std + a_std + b_std) / 5
        entropy = (h_entropy + s_entropy) / 2

        # BALANCED scoring
        if complexity > 50 and entropy > 3.2:
            color_score = 80
        elif complexity > 40 and entropy > 2.8:
            color_score = 70
        elif complexity > 30 and entropy > 2.3:
            color_score = 60
        elif complexity > 20 and entropy > 1.8:
            color_score = 50
        elif complexity > 15 and entropy > 1.3:
            color_score = 40
        else:
            color_score = 30

        return float(np.clip(color_score, 0, 100))

    def _frequency_score_v5(self, arr):
        """
        BALANCED frequency analysis - sophisticated but not aggressive
        Real: More natural frequency distribution
        AI: Tends toward low-frequency dominance
        """
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)

        h, w = gray.shape
        size = min(h, w)
        gray = gray[:size, :size]

        # DCT analysis
        dct_result = dct(dct(gray.T, norm='ortho').T, norm='ortho')
        dct_mag = np.abs(dct_result)

        total_size = dct_mag.shape[0]

        # Better frequency band definitions
        q1 = total_size // 5
        q2 = total_size // 3
        q3 = (total_size * 2) // 3

        dc_band = dct_mag[0:q1, 0:q1]
        low_band = dct_mag[0:q2, 0:q2]
        mid_band = dct_mag[q1:q3, q1:q3]
        high_band = dct_mag[q2:, q2:]

        dc_energy = np.sum(dc_band)
        low_energy = np.sum(low_band) - dc_energy
        mid_energy = np.sum(mid_band)
        high_energy = np.sum(high_band)

        total_energy = dc_energy + low_energy + mid_energy + high_energy + 1e-10

        dc_ratio = dc_energy / total_energy
        low_ratio = low_energy / total_energy
        mid_ratio = mid_energy / total_energy
        high_ratio = high_energy / total_energy

        low_freq_total = dc_ratio + low_ratio
        high_freq_total = mid_ratio + high_ratio

        logger.info(f"Freq v5 - Low:{low_freq_total:.3f} High:{high_freq_total:.3f} DC:{dc_ratio:.3f}")

        # Entropy
        freqs = np.array([dc_ratio, low_ratio, mid_ratio, high_ratio])
        freq_entropy = self._entropy(freqs * 1000)

        # BALANCED DETECTION - not overly strict
        score = 50

        # Clear AI signatures (must be VERY clear)
        if low_freq_total > 0.78 and high_freq_total < 0.15 and dc_ratio < 0.15:
            logger.info("Strong AI signature detected")
            score = 18

        # Clear real signatures (more permissive)
        elif low_freq_total < 0.62 and high_freq_total > 0.28:
            logger.info("Strong real signature detected")
            score = 80

        # Gradual scoring with more nuance
        elif low_freq_total > 0.72:
            if high_freq_total < 0.18:
                score = 25
            elif high_freq_total < 0.22:
                score = 35
            else:
                score = 45

        elif low_freq_total > 0.65:
            if high_freq_total > 0.25:
                score = 60
            else:
                score = 50

        elif low_freq_total < 0.58:
            if high_freq_total > 0.32:
                score = 85
            else:
                score = 75

        else:  # 0.58 - 0.65
            if high_freq_total > 0.28:
                score = 72
            else:
                score = 60

        # Entropy adjustment (more subtle)
        if freq_entropy < 0.9:
            score = score * 0.9
        elif freq_entropy > 1.7:
            score = score * 1.05

        return float(np.clip(score, 0, 100))

    def _combine_scores_v5(self, c, e, n, col, f):
        """
        BALANCED weighted combination
        Fair to both real and AI detection
        """
        w = self.weights

        raw_score = (
            c * w['compression'] +
            e * w['edge'] +
            n * w['noise'] +
            col * w['color'] +
            f * w['frequency']
        )

        logger.info(f"Raw weighted score: {raw_score:.2f}")

        # Gentle polarization - not aggressive
        if raw_score < 25:
            final_score = raw_score * 0.95
        elif raw_score < 40:
            final_score = raw_score * 0.98
        elif raw_score > 80:
            final_score = 80 + (raw_score - 80) * 1.05
        elif raw_score > 70:
            final_score = 70 + (raw_score - 70) * 1.03
        else:
            final_score = raw_score

        return float(np.clip(final_score, 0, 100))

    def _verdict(self, score):
        """Generate verdict - FAIR THRESHOLDS"""
        if score >= 72:
            return {
                'classification': 'authentic',
                'title': 'Likely Authentic',
                'description': 'Image shows characteristics of a genuine photograph with natural forensic patterns.',
                'confidence': 'high',
            }
        elif score >= 60:
            return {
                'classification': 'uncertain',
                'title': 'Uncertain/Mixed Indicators',
                'description': 'Image shows mixed characteristics. Some patterns suggest authenticity while others raise concerns.',
                'confidence': 'medium',
            }
        elif score >= 42:
            return {
                'classification': 'suspicious',
                'title': 'Suspicious - Possibly Manipulated',
                'description': 'Multiple indicators suggest potential manipulation or significant editing.',
                'confidence': 'medium',
            }
        else:
            return {
                'classification': 'fake',
                'title': 'Likely AI-Generated',
                'description': 'Strong forensic evidence indicates this is AI-generated or heavily manipulated.',
                'confidence': 'high',
            }

    @staticmethod
    def _entropy(data):
        """Calculate Shannon entropy"""
        data = np.array(data, dtype=np.float64).flatten()
        data = data[data > 0]
        if len(data) == 0:
            return 0.0
        data = data / data.sum()
        return float(-(data * np.log2(data + 1e-10)).sum())
