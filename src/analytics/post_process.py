import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_filter

class PostProcessor:
    def __init__(self, sigma=2):
        self.sigma = sigma

    def compute_ssim_map(self, original, reconstructed):
        """
        Computes the Structural Similarity Index (SSIM) map.
        Returns 1 - SSIM as the anomaly map (higher value = more anomalous).
        """
        # Ensure inputs are numpy arrays in range [0, 1] or [0, 255]
        # Assuming inputs are [H, W] grayscale
        
        # Data range is important for SSIM. 
        # If float, assume [0, 1]. If int, assume [0, 255].
        data_range = 1.0 if original.dtype.kind == 'f' else 255
        
        score, diff_map = ssim(original, reconstructed, full=True, data_range=data_range)
        
        # We want the anomaly map, so we invert the similarity map
        # SSIM is 1 for identical, -1 for opposite. 
        # Diff map from ssim is the local similarity.
        # We want high values for anomalies.
        anomaly_map = 1 - diff_map
        
        return anomaly_map

    def apply_gaussian_blur(self, anomaly_map):
        """
        Applies Gaussian Blur to smooth the anomaly map.
        """
        return gaussian_filter(anomaly_map, sigma=self.sigma)

    def process(self, original, reconstructed):
        """
        Runs the full post-processing pipeline.
        Returns:
            raw_diff: L1 difference
            ssim_map: 1 - SSIM
            smoothed_map: Gaussian blurred SSIM map
        """
        # 1. Raw Difference (L1)
        raw_diff = np.abs(original - reconstructed)
        
        # 2. SSIM Map
        ssim_map = self.compute_ssim_map(original, reconstructed)
        
        # 3. Smoothed Map (using SSIM map as base, or raw diff? User said "Smoothed Heatmap")
        # Usually smoothing is applied to the final anomaly score. 
        # The user said "Smoothed Heatmap (Gaussian filtered)". 
        # I will smooth the SSIM map as it's the "structural loss".
        smoothed_map = self.apply_gaussian_blur(ssim_map)
        
        return raw_diff, ssim_map, smoothed_map
