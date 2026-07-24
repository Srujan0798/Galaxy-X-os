#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- Astronomical Image Preprocessing

Handles noise removal, brightness normalization, telescope artifact simulation,
and multi-scale processing specifically for astronomical imagery.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Union
from PIL import Image


class AstroPreprocessor:
    """
    Astronomical image preprocessing pipeline.

    Handles:
    - CCD/CMOS sensor noise (Gaussian + Poisson)
    - Cosmic ray hits (salt-and-pepper outliers)
    - Vignetting and telescope edge artifacts
    - Varying dynamic ranges across instruments
    - Background gradient removal
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        denoise_strength: float = 10.0,
        clahe_clip: float = 2.0,
        clahe_grid: Tuple[int, int] = (8, 8),
        remove_gradient: bool = True,
        normalize_background: bool = True,
    ):
        self.target_size = target_size
        self.denoise_strength = denoise_strength
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=clahe_grid,
        )
        self.remove_gradient = remove_gradient
        self.normalize_background = normalize_background

    def remove_cosmic_rays(self, image: np.ndarray, threshold: float = 5.0) -> np.ndarray:
        """Remove cosmic ray hits using Laplacian edge detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        mask = np.abs(laplacian) > threshold * np.std(laplacian)

        if not mask.any():
            return image

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        mask_uint8 = mask.astype(np.uint8) * 255
        cleaned = cv2.inpaint(image, mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return cleaned

    def estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate Gaussian noise sigma using median absolute deviation."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        h, w = gray.shape
        corners = [
            gray[:h // 4, :w // 4],
            gray[:h // 4, -w // 4:],
            gray[-h // 4:, :w // 4],
            gray[-h // 4:, -w // 4:],
        ]
        corner_noise = np.concatenate([c.flatten() for c in corners])
        mad = np.median(np.abs(corner_noise - np.median(corner_noise)))
        sigma = mad / 0.6745
        return max(sigma, 0.1)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply non-local means denoising with noise level estimation."""
        if image.dtype == np.float32 or image.dtype == np.float64:
            img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            img_uint8 = image

        denoised = cv2.fastNlMeansDenoisingColored(
            img_uint8,
            None,
            h=self.denoise_strength,
            hColor=self.denoise_strength * 0.5,
            templateWindowSize=7,
            searchWindowSize=21,
        )
        return denoised

    def remove_background_gradient(self, image: np.ndarray) -> np.ndarray:
        """Remove smooth background gradient (vignetting, light pollution)."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

        kernel_size = max(image.shape[:2]) // 8
        kernel_size = max(kernel_size, 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        background = cv2.GaussianBlur(background, (51, 51), 0)

        if image.ndim == 3:
            result = image.astype(np.float32)
            for i in range(3):
                ch = image[:, :, i].astype(np.float32)
                bg = background.astype(np.float32)
                ch_corrected = ch - bg + np.median(bg)
                result[:, :, i] = np.clip(ch_corrected, 0, 255)
            return result.astype(np.uint8)
        else:
            corrected = gray.astype(np.float32) - background.astype(np.float32)
            corrected = corrected + np.median(background)
            return np.clip(corrected, 0, 255).astype(np.uint8)

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_equalized = self.clahe.apply(l)
        lab_equalized = cv2.merge([l_equalized, a, b])
        result = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2RGB)
        return result

    def normalize_dynamic_range(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to full [0, 255] range using percentile clipping."""
        img_float = image.astype(np.float32) if image.dtype == np.uint8 else image
        p_low, p_high = np.percentile(img_float, [1, 99.5])

        if p_high > p_low:
            stretched = (img_float - p_low) / (p_high - p_low) * 255
        else:
            stretched = img_float

        return np.clip(stretched, 0, 255).astype(np.uint8)

    def remove_vignetting(self, image: np.ndarray) -> np.ndarray:
        """Correct vignetting (darkening at image edges)."""
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)

        radial_profile = 1 + 0.3 * (dist_from_center / max_dist) ** 2

        corrected = image.astype(np.float32) * radial_profile[:, :, np.newaxis]
        return np.clip(corrected, 0, 255).astype(np.uint8)

    def correct_hot_pixels(self, image: np.ndarray) -> np.ndarray:
        """Remove hot/stuck pixels using median filtering."""
        return cv2.medianBlur(image, 3)

    def preprocess(self, image: Union[np.ndarray, str, Path]) -> np.ndarray:
        """
        Full preprocessing pipeline for astronomical images.

        Pipeline order:
        1. Cosmic ray removal
        2. Hot pixel correction
        3. Non-local means denoising
        4. Background gradient removal
        5. Vignetting correction
        6. Dynamic range normalization
        7. CLAHE for contrast enhancement
        8. Resize to target
        """
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image).convert("RGB"))

        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        image = self.remove_cosmic_rays(image)
        image = self.correct_hot_pixels(image)
        image = self.denoise(image)

        if self.remove_gradient:
            image = self.remove_background_gradient(image)

        image = self.remove_vignetting(image)
        image = self.normalize_dynamic_range(image)
        image = self.apply_clahe(image)
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)

        return image


def preprocess_image(image_path: Union[str, Path], target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Quick preprocessing wrapper."""
    preprocessor = AstroPreprocessor(target_size=target_size)
    return preprocessor.preprocess(image_path)
