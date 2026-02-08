"""GPU-accelerated Filter Bank CCA (FBCCA) for SSVEP feature extraction.

Zero-parameter module: computes CCA correlations between multi-channel EEG
and sinusoidal templates at 40 SSVEP frequencies across 5 filter sub-bands.

SSVEP frequencies: 8.0, 8.2, 8.4, ..., 15.8 Hz (40 targets, 0.2 Hz steps)
Sub-bands: [6-90, 14-90, 22-90, 30-90, 38-90] Hz (bandpass via FFT)
Harmonics: 3 per frequency (f, 2f, 3f)

Output: (B, 200) = 5 sub-bands × 40 frequencies of weighted CCA correlations.
"""

import torch
import torch.nn as nn


# 40 SSVEP target frequencies (Hz)
SSVEP_FREQS = [8.0 + 0.2 * i for i in range(40)]

# 5 filter sub-bands: [low_cutoff, high_cutoff] in Hz
FILTER_BANDS = [
    (6.0, 90.0),
    (14.0, 90.0),
    (22.0, 90.0),
    (30.0, 90.0),
    (38.0, 90.0),
]

# Sub-band weights: w_k = k^{-1.25} + 0.25
BAND_WEIGHTS = [(k + 1) ** (-1.25) + 0.25 for k in range(len(FILTER_BANDS))]

N_HARMONICS = 3


class FBCCAFeatureExtractor(nn.Module):
    """GPU-accelerated FBCCA feature extractor (zero learnable parameters).

    For each EEG window, computes CCA correlation with sinusoidal templates
    at 40 SSVEP frequencies across 5 filter sub-bands.

    Args:
        sfreq: Sampling frequency of input EEG (Hz)
        n_timepoints: Number of timepoints per window
        n_harmonics: Number of harmonics for sinusoidal templates
    """

    def __init__(self, sfreq=200.0, n_timepoints=300, n_harmonics=N_HARMONICS):
        super().__init__()
        self.sfreq = sfreq
        self.n_timepoints = n_timepoints
        self.n_harmonics = n_harmonics
        self.n_freqs = len(SSVEP_FREQS)
        self.n_bands = len(FILTER_BANDS)

        # Pre-compute sinusoidal reference templates: (n_freqs, 2*n_harmonics, T)
        # For each frequency f and harmonic h: [sin(2π·h·f·t), cos(2π·h·f·t)]
        t = torch.arange(n_timepoints, dtype=torch.float32) / sfreq  # (T,)
        templates = []
        for freq in SSVEP_FREQS:
            refs = []
            for h in range(1, n_harmonics + 1):
                phase = 2.0 * torch.pi * h * freq * t  # (T,)
                refs.append(torch.sin(phase))
                refs.append(torch.cos(phase))
            templates.append(torch.stack(refs))  # (2*n_harmonics, T)
        templates = torch.stack(templates)  # (n_freqs, 2*n_harmonics, T)
        self.register_buffer("templates", templates)

        # Pre-compute FFT frequency bin masks for each sub-band
        n_fft = n_timepoints
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sfreq)  # (n_fft//2+1,)
        band_masks = []
        for low, high in FILTER_BANDS:
            mask = (freqs >= low) & (freqs <= high)
            band_masks.append(mask)
        self.register_buffer("band_masks", torch.stack(band_masks))  # (n_bands, n_fft//2+1)

        # Sub-band weights as buffer
        self.register_buffer("band_weights", torch.tensor(BAND_WEIGHTS, dtype=torch.float32))

    @torch.no_grad()
    def forward(self, eeg):
        """Extract FBCCA features from raw EEG.

        Args:
            eeg: (B, C, T) raw EEG at self.sfreq Hz

        Returns:
            features: (B, n_bands * n_freqs) = (B, 200)
        """
        B, C, T = eeg.shape
        assert T == self.n_timepoints, f"Expected T={self.n_timepoints}, got {T}"

        # Apply FFT-based bandpass filtering for all sub-bands at once
        # eeg: (B, C, T) -> FFT -> (B, C, n_fft//2+1) complex
        eeg_fft = torch.fft.rfft(eeg, dim=-1)  # (B, C, F)

        all_corr = []
        for band_idx in range(self.n_bands):
            # Apply bandpass: zero out-of-band bins
            mask = self.band_masks[band_idx]  # (F,)
            filtered_fft = eeg_fft * mask.unsqueeze(0).unsqueeze(0)  # (B, C, F)
            filtered = torch.fft.irfft(filtered_fft, n=T, dim=-1)  # (B, C, T)

            # Compute CCA correlation for each frequency
            band_corr = self._cca_all_freqs(filtered)  # (B, n_freqs)
            all_corr.append(band_corr)

        # Stack: (B, n_bands, n_freqs)
        all_corr = torch.stack(all_corr, dim=1)
        # Flatten to (B, n_bands * n_freqs)
        return all_corr.reshape(B, -1)

    def _cca_all_freqs(self, filtered_eeg):
        """Compute CCA correlation between filtered EEG and all frequency templates.

        For sinusoidal templates, CCA simplifies to:
        ρ = max singular value of: R_yy^{-1/2} @ R_yx @ R_xx^{-1/2}
        where X = EEG (C channels), Y = sinusoidal template (2*n_harmonics).

        Since Y is a fixed sinusoidal template, we can simplify further:
        ρ² = max eigenvalue of: (X @ Y^T) @ (Y @ Y^T)^{-1} @ (Y @ X^T) @ (X @ X^T)^{-1}

        But for numerical stability and simplicity, we use the SVD approach
        on the normalized cross-covariance matrix.

        Args:
            filtered_eeg: (B, C, T) bandpass-filtered EEG

        Returns:
            correlations: (B, n_freqs)
        """
        B, C, T = filtered_eeg.shape

        # Zero-mean the EEG (per channel)
        eeg_centered = filtered_eeg - filtered_eeg.mean(dim=-1, keepdim=True)  # (B, C, T)

        # Covariance of EEG: R_xx = (1/T) * X @ X^T, shape (B, C, C)
        # For numerical stability, add small regularization
        R_xx = torch.bmm(eeg_centered, eeg_centered.transpose(-1, -2)) / T  # (B, C, C)
        R_xx = R_xx + 1e-6 * torch.eye(C, device=R_xx.device, dtype=R_xx.dtype).unsqueeze(0)

        # Inverse square root of R_xx via eigendecomposition
        # R_xx = V @ diag(λ) @ V^T => R_xx^{-1/2} = V @ diag(λ^{-1/2}) @ V^T
        eigvals, eigvecs = torch.linalg.eigh(R_xx)  # (B, C), (B, C, C)
        eigvals = eigvals.clamp(min=1e-6)
        R_xx_inv_sqrt = eigvecs @ torch.diag_embed(eigvals.rsqrt()) @ eigvecs.transpose(-1, -2)

        correlations = []
        for f_idx in range(self.n_freqs):
            Y = self.templates[f_idx]  # (2*n_harmonics, T)
            # Zero-mean the template
            Y_centered = Y - Y.mean(dim=-1, keepdim=True)  # (2H, T)
            n_y = Y_centered.shape[0]

            # R_yy = (1/T) * Y @ Y^T, shape (2H, 2H)
            R_yy = (Y_centered @ Y_centered.T) / T
            R_yy = R_yy + 1e-6 * torch.eye(n_y, device=R_yy.device, dtype=R_yy.dtype)

            eigvals_y, eigvecs_y = torch.linalg.eigh(R_yy)
            eigvals_y = eigvals_y.clamp(min=1e-6)
            R_yy_inv_sqrt = eigvecs_y @ torch.diag_embed(eigvals_y.rsqrt()) @ eigvecs_y.T  # (2H, 2H)

            # Cross-covariance: R_xy = (1/T) * X @ Y^T, shape (B, C, 2H)
            R_xy = torch.bmm(eeg_centered, Y_centered.T.unsqueeze(0).expand(B, -1, -1)) / T

            # Canonical correlation matrix: R_xx^{-1/2} @ R_xy @ R_yy^{-1/2}
            # Shape: (B, C, 2H)
            M = R_xx_inv_sqrt @ R_xy @ R_yy_inv_sqrt.unsqueeze(0)

            # Max canonical correlation = max singular value of M
            # Use SVD; we only need the largest singular value
            s = torch.linalg.svdvals(M)  # (B, min(C, 2H))
            rho = s[:, 0].clamp(0.0, 1.0)  # (B,) max correlation
            correlations.append(rho)

        return torch.stack(correlations, dim=1)  # (B, n_freqs)

    @property
    def output_dim(self):
        return self.n_bands * self.n_freqs
