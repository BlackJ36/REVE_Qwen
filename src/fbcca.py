"""GPU-accelerated Filter Bank CCA (FBCCA) for SSVEP feature extraction.

Zero-parameter module: computes CCA correlations between multi-channel EEG
and sinusoidal templates at 40 SSVEP frequencies across 5 filter sub-bands.

SSVEP frequencies: 8.0, 8.2, 8.4, ..., 15.8 Hz (40 targets, 0.2 Hz steps)
Sub-bands: [6-90, 14-90, 22-90, 30-90, 38-90] Hz (bandpass via FFT)
Harmonics: 3 per frequency (f, 2f, 3f)

Output: (B, 200) = 5 sub-bands × 40 frequencies of weighted CCA correlations.

Fully batched: all 40 frequencies computed in parallel via einsum, no Python loops
over frequencies. R_yy^{-1/2} pre-computed in __init__.
"""

import torch
import torch.nn as nn


# 40 SSVEP target frequencies in data label order (5×8 grid, row-major)
# Row i, Col j: freq = 8.0 + j*1.0 + i*0.2, label = i*8 + j
SSVEP_FREQS = [8.0 + (i % 8) * 1.0 + (i // 8) * 0.2 for i in range(40)]

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

    All 40 frequencies are computed in parallel via batched matrix ops.
    R_yy^{-1/2} is pre-computed once in __init__ (templates are constant).

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
        self.n_ref = 2 * n_harmonics  # sin + cos per harmonic
        self.n_freqs = len(SSVEP_FREQS)
        self.n_bands = len(FILTER_BANDS)

        T = n_timepoints
        t = torch.arange(T, dtype=torch.float32) / sfreq  # (T,)

        # Build templates: (n_freqs, 2H, T), zero-mean
        templates = []
        for freq in SSVEP_FREQS:
            refs = []
            for h in range(1, n_harmonics + 1):
                phase = 2.0 * torch.pi * h * freq * t
                refs.append(torch.sin(phase))
                refs.append(torch.cos(phase))
            templates.append(torch.stack(refs))  # (2H, T)
        templates = torch.stack(templates)  # (F, 2H, T)
        # Zero-mean templates
        templates = templates - templates.mean(dim=-1, keepdim=True)
        self.register_buffer("templates", templates)

        # Pre-compute R_yy^{-1/2} for all 40 freqs: (F, 2H, 2H)
        # R_yy = (1/T) * Y @ Y^T
        R_yy = torch.bmm(templates, templates.transpose(-1, -2)) / T  # (F, 2H, 2H)
        R_yy = R_yy + 1e-6 * torch.eye(self.n_ref).unsqueeze(0)
        eigvals_y, eigvecs_y = torch.linalg.eigh(R_yy)  # (F, 2H), (F, 2H, 2H)
        eigvals_y = eigvals_y.clamp(min=1e-6)
        R_yy_inv_sqrt = eigvecs_y @ torch.diag_embed(eigvals_y.rsqrt()) @ eigvecs_y.transpose(-1, -2)
        self.register_buffer("R_yy_inv_sqrt", R_yy_inv_sqrt)  # (F, 2H, 2H)

        # FFT frequency bin masks for sub-bands
        freqs = torch.fft.rfftfreq(T, d=1.0 / sfreq)
        band_masks = []
        for low, high in FILTER_BANDS:
            band_masks.append((freqs >= low) & (freqs <= high))
        self.register_buffer("band_masks", torch.stack(band_masks))  # (n_bands, n_fft//2+1)

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

        # FFT once, reuse for all sub-bands
        eeg_fft = torch.fft.rfft(eeg, dim=-1)  # (B, C, n_fft)

        all_corr = []
        for band_idx in range(self.n_bands):
            mask = self.band_masks[band_idx]  # (n_fft,)
            filtered_fft = eeg_fft * mask.unsqueeze(0).unsqueeze(0)
            filtered = torch.fft.irfft(filtered_fft, n=T, dim=-1)  # (B, C, T)
            band_corr = self._cca_batched(filtered)  # (B, n_freqs)
            all_corr.append(band_corr)

        return torch.stack(all_corr, dim=1).reshape(B, -1)  # (B, 200)

    def _cca_batched(self, filtered_eeg):
        """Compute CCA correlations for all 40 frequencies in parallel.

        Avoids expensive eigh on (B, 62, 62) by using torch.linalg.solve (LU).
        All 40 frequencies solved in a single call via reshaping.

        CCA ρ² = max eigenvalue of R_yy^{-1/2} @ R_yx @ R_xx^{-1} @ R_xy @ R_yy^{-1/2}
        which is a tiny (6×6) eigenproblem per frequency.

        Args:
            filtered_eeg: (B, C, T) bandpass-filtered EEG

        Returns:
            correlations: (B, n_freqs)
        """
        B, C, T = filtered_eeg.shape
        F = self.n_freqs   # 40
        H = self.n_ref     # 6

        # Zero-mean EEG per channel
        X = filtered_eeg - filtered_eeg.mean(dim=-1, keepdim=True)  # (B, C, T)

        # R_xx = (1/T) X @ X^T + εI: (B, C, C)
        R_xx = torch.bmm(X, X.transpose(-1, -2)) / T
        R_xx = R_xx + 1e-4 * torch.eye(C, device=X.device, dtype=X.dtype).unsqueeze(0)

        # Cross-covariance for all 40 freqs: (B, F, C, H)
        R_xy_all = torch.einsum("bct, fht -> bfch", X, self.templates) / T

        # Solve R_xx @ Z = R_xy for all 40 freqs in one call
        # Reshape to (B, C, F*H) so solve sees a single right-hand side matrix
        R_xy_rhs = R_xy_all.permute(0, 2, 1, 3).reshape(B, C, F * H)  # (B, 62, 240)
        Z_flat = torch.linalg.solve(R_xx, R_xy_rhs)  # (B, 62, 240)
        Z = Z_flat.reshape(B, C, F, H).permute(0, 2, 1, 3)  # (B, F, C, H)

        # Kernel = R_xy^T @ Z = R_xy^T @ R_xx^{-1} @ R_xy: (B, F, H, H)
        kernel = torch.einsum("bfch, bfck -> bfhk", R_xy_all, Z)

        # K = R_yy^{-1/2} @ kernel @ R_yy^{-1/2}: (B, F, H, H)
        K = torch.einsum("fhi, bfij, fjk -> bfhk",
                         self.R_yy_inv_sqrt, kernel, self.R_yy_inv_sqrt)

        # ρ² = max eigenvalue of K (6×6 matrix per freq)
        # Symmetrize K to handle numerical asymmetry from zero-padded trials
        K = (K + K.transpose(-1, -2)) / 2
        K_flat = K.reshape(B * F, H, H)
        # Add small regularization for ill-conditioned matrices (e.g. zero-padded BETA)
        K_flat = K_flat + 1e-6 * torch.eye(H, device=K_flat.device).unsqueeze(0)
        CHUNK = 8192
        parts = []
        for i in range(0, B * F, CHUNK):
            ev = torch.linalg.eigvalsh(K_flat[i:i + CHUNK])[:, -1]
            parts.append(ev)
        max_eigval = torch.cat(parts)
        rho = max_eigval.clamp(min=0.0).sqrt().clamp(max=1.0)

        return rho.reshape(B, F)  # (B, 40)

    @property
    def output_dim(self):
        return self.n_bands * self.n_freqs
