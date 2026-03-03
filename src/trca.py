"""GPU-accelerated Filter Bank TRCA for SSVEP classification.

Task-Related Component Analysis learns spatial filters from calibration
data that maximize inter-trial reproducibility. Unlike FBCCA which relies
on sinusoidal templates (limited by frequency resolution), TRCA uses
data-driven templates and spatial filters, making it robust at short
trial durations (1-2s).

Filter Bank TRCA (FBTRCA) applies the same 5 sub-band decomposition as
FBCCA, running TRCA within each sub-band and combining via weighted sum.

Ensemble TRCA (eTRCA) uses spatial filters from ALL frequency classes
to evaluate each target, providing more robust classification.

References:
    Nakanishi et al. (2018): Enhancing Detection of SSVEPs for a
    High-Speed Brain Speller Using Task-Related Component Analysis.
    IEEE Trans. Biomed. Eng.
"""

import torch
import torch.nn as nn

from src.fbcca import FILTER_BANDS, BAND_WEIGHTS


def bandpass_fft(eeg, sfreq, low, high):
    """FFT-based bandpass filter (same approach as FBCCA).

    Args:
        eeg: (N, C, T) EEG data
        sfreq: sampling frequency
        low: low cutoff Hz
        high: high cutoff Hz

    Returns:
        filtered: (N, C, T) bandpass-filtered EEG
    """
    T = eeg.shape[-1]
    fft = torch.fft.rfft(eeg, dim=-1)
    freqs = torch.fft.rfftfreq(T, d=1.0 / sfreq).to(eeg.device)
    mask = (freqs >= low) & (freqs <= high)
    filtered_fft = fft * mask.unsqueeze(0).unsqueeze(0)
    return torch.fft.irfft(filtered_fft, n=T, dim=-1)


class TRCAModel:
    """Single-band TRCA model (fits on one sub-band's filtered data).

    For each frequency class k (0-39):
        1. Average calibration trials → template X̄_k
        2. Compute inter-trial covariance S_k and within-trial Q_k
        3. Solve generalized eigenproblem S_k w = λ Q_k w → spatial filter w_k

    Classification: argmax_k corr(w_k^T @ X_test, w_k^T @ X̄_k)
    Ensemble mode: argmax_k mean_j [corr(w_j^T @ X_test, w_j^T @ X̄_k)]
    """

    def __init__(self, n_classes=40, n_components=1, reg=1e-6):
        self.n_classes = n_classes
        self.n_components = n_components
        self.reg = reg
        self.spatial_filters = None  # (n_classes, C, n_components)
        self.templates = None        # (n_classes, C, T)

    def fit(self, eeg_cal, labels_cal):
        """Fit TRCA spatial filters and templates from calibration data.

        Args:
            eeg_cal: (N_cal, C, T) calibration EEG (should be bandpass-filtered)
            labels_cal: (N_cal,) integer labels 0-39
        """
        device = eeg_cal.device
        dtype = eeg_cal.dtype
        C = eeg_cal.shape[1]
        T = eeg_cal.shape[2]

        all_S = []
        all_Q = []
        templates = []

        for k in range(self.n_classes):
            mask = labels_cal == k
            trials_k = eeg_cal[mask]  # (N_k, C, T)
            N_k = trials_k.shape[0]

            if N_k < 2:
                # Not enough trials: zero template, identity filter
                templates.append(torch.zeros(C, T, device=device, dtype=dtype))
                all_S.append(torch.zeros(C, C, device=device, dtype=dtype))
                all_Q.append(torch.eye(C, device=device, dtype=dtype))
                continue

            # Zero-mean each trial
            trials_k = trials_k - trials_k.mean(dim=-1, keepdim=True)

            # Template: average across trials
            template_k = trials_k.mean(dim=0)  # (C, T)
            templates.append(template_k)

            # Inter-trial covariance: S = (ΣX_i)(ΣX_j)^T - Σ(X_i X_i^T)
            sum_trials = trials_k.sum(dim=0)  # (C, T)
            S = (sum_trials @ sum_trials.T)  # (C, C)
            self_cov = torch.einsum("nct, ndt -> cd", trials_k, trials_k)
            S = S - self_cov

            # Within-trial covariance: Q = Σ X_i X_i^T
            Q = self_cov

            all_S.append(S)
            all_Q.append(Q)

        # Batch solve generalized eigenvalue problems
        S_batch = torch.stack(all_S)  # (40, C, C)
        Q_batch = torch.stack(all_Q)  # (40, C, C)
        self.spatial_filters = self._solve_gep_batched(S_batch, Q_batch, C, device, dtype)
        self.templates = torch.stack(templates)  # (40, C, T)

    def _solve_gep_batched(self, S, Q, C, device, dtype):
        """Solve batched generalized eigenproblem S w = λ Q w.

        Uses Q^{-1} S formulation with full eigendecomp on small 62×62 matrices.

        Returns:
            W: (K, C, n_components) spatial filters
        """
        K = S.shape[0]
        I = torch.eye(C, device=device, dtype=dtype).unsqueeze(0)
        Q_reg = Q + self.reg * I

        # Solve Q^{-1} S via linear solve
        QinvS = torch.linalg.solve(Q_reg, S)  # (K, C, C)

        # Eigendecomposition (QinvS has real eigenvalues since S, Q are sym PSD)
        eigvals, eigvecs = torch.linalg.eig(QinvS)  # complex

        # Take top n_components by real eigenvalue
        real_eigvals = eigvals.real  # (K, C)
        _, top_idx = real_eigvals.topk(self.n_components, dim=-1)  # (K, n_comp)

        # Gather corresponding eigenvectors
        W = torch.zeros(K, C, self.n_components, device=device, dtype=dtype)
        for comp in range(self.n_components):
            idx = top_idx[:, comp]  # (K,)
            for k in range(K):
                W[k, :, comp] = eigvecs[k, :, idx[k]].real

        # Normalize
        W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        return W

    @torch.no_grad()
    def predict_correlations(self, eeg_test, ensemble=False):
        """Compute TRCA correlations for classification.

        Args:
            eeg_test: (B, C, T) test EEG (same band-filtering as fit)
            ensemble: if True, use ensemble TRCA (all spatial filters per class)

        Returns:
            correlations: (B, n_classes) Pearson correlations
        """
        B, C, T = eeg_test.shape
        X = eeg_test - eeg_test.mean(dim=-1, keepdim=True)  # (B, C, T)

        # Zero-mean templates
        tmpl = self.templates - self.templates.mean(dim=-1, keepdim=True)  # (40, C, T)

        if ensemble:
            return self._ensemble_correlations(X, tmpl)
        else:
            return self._standard_correlations(X, tmpl)

    def _standard_correlations(self, X, tmpl):
        """Standard TRCA: r_k = corr(w_k^T X, w_k^T X̄_k).

        Vectorized: computes all K classes in parallel via einsum.
        """
        W = self.spatial_filters  # (K, C, n_comp)

        # Apply each class's spatial filter to test data (diagonal: filter k → class k)
        # filtered_X[k, b, n, t] = W[k]^T @ X[b]
        fX = torch.einsum("kcn, bct -> kbnt", W, X)  # (K, B, n_comp, T)
        # filtered_tmpl[k, n, t] = W[k]^T @ tmpl[k]
        fT = torch.einsum("kcn, kct -> knt", W, tmpl)  # (K, n_comp, T)

        # Zero-mean for Pearson correlation
        fX = fX - fX.mean(dim=-1, keepdim=True)
        fT = fT - fT.mean(dim=-1, keepdim=True)

        # Correlation: (K, B, n_comp)
        num = torch.einsum("kbnt, knt -> kbn", fX, fT)
        den = fX.norm(dim=-1) * fT.norm(dim=-1).unsqueeze(1) + 1e-8

        # Average across components, transpose to (B, K)
        return (num / den).mean(dim=-1).T

    def _ensemble_correlations(self, X, tmpl):
        """Ensemble TRCA: r_k = mean_j [corr(w_j^T X, w_j^T X̄_k)].

        Vectorized: all J×K filter-class combinations computed in parallel.
        """
        W = self.spatial_filters  # (K, C, n_comp)

        # Apply ALL spatial filters to test data: (J, B, n_comp, T)
        fX = torch.einsum("jcn, bct -> jbnt", W, X)

        # Apply ALL spatial filters to ALL templates: (J, K, n_comp, T)
        fT = torch.einsum("jcn, kct -> jknt", W, tmpl)

        # Zero-mean
        fX = fX - fX.mean(dim=-1, keepdim=True)
        fT = fT - fT.mean(dim=-1, keepdim=True)

        # Correlation for all (j, b, k, n_comp) pairs
        num = torch.einsum("jbnt, jknt -> jbkn", fX, fT)  # (J, B, K, n_comp)
        den_X = fX.norm(dim=-1)    # (J, B, n_comp)
        den_T = fT.norm(dim=-1)    # (J, K, n_comp)
        den = torch.einsum("jbn, jkn -> jbkn", den_X, den_T) + 1e-8

        # Average across filters j and components n → (B, K)
        return (num / den).mean(dim=(0, 3))


class FBTRCAClassifier:
    """Filter Bank TRCA classifier.

    Applies same 5 sub-bands as FBCCA, runs TRCA within each band,
    and combines correlations via weighted sum.
    """

    def __init__(self, n_classes=40, n_components=1, reg=1e-6, sfreq=200.0):
        self.n_classes = n_classes
        self.n_components = n_components
        self.reg = reg
        self.sfreq = sfreq
        self.n_bands = len(FILTER_BANDS)
        self.band_weights = torch.tensor(BAND_WEIGHTS, dtype=torch.float32)
        self.models = []  # one TRCAModel per sub-band

    def fit(self, eeg_cal, labels_cal):
        """Fit FBTRCA from calibration data.

        Args:
            eeg_cal: (N_cal, C, T) raw calibration EEG (NOT pre-filtered)
            labels_cal: (N_cal,) integer labels 0-39
        """
        self.models = []
        for band_idx, (low, high) in enumerate(FILTER_BANDS):
            filtered = bandpass_fft(eeg_cal, self.sfreq, low, high)
            trca = TRCAModel(
                n_classes=self.n_classes,
                n_components=self.n_components,
                reg=self.reg,
            )
            trca.fit(filtered, labels_cal)
            self.models.append(trca)

    def _compute_weighted_corr(self, eeg_test, ensemble=False):
        """Compute weighted correlation scores across all sub-bands.

        Returns:
            weighted_corr: (B, n_classes) full correlation scores
        """
        B = eeg_test.shape[0]
        device = eeg_test.device
        weights = self.band_weights.to(device)

        weighted_corr = torch.zeros(B, self.n_classes, device=device, dtype=eeg_test.dtype)

        for band_idx, (low, high) in enumerate(FILTER_BANDS):
            filtered = bandpass_fft(eeg_test, self.sfreq, low, high)
            corr = self.models[band_idx].predict_correlations(filtered, ensemble=ensemble)
            weighted_corr += weights[band_idx] * corr

        return weighted_corr

    @torch.no_grad()
    def predict(self, eeg_test, ensemble=False):
        """Classify test trials using FBTRCA.

        Args:
            eeg_test: (B, C, T) raw test EEG (NOT pre-filtered)
            ensemble: if True, use ensemble TRCA within each sub-band

        Returns:
            top3_indices: (B, 3) int64
            top3_scores:  (B, 3) float32
        """
        weighted_corr = self._compute_weighted_corr(eeg_test, ensemble=ensemble)
        top3_scores, top3_indices = weighted_corr.topk(3, dim=-1)
        return top3_indices.to(torch.int64), top3_scores.float()

    @torch.no_grad()
    def predict_full(self, eeg_test, ensemble=False):
        """Classify test trials and return full 40-dim correlation scores.

        Used for knowledge distillation: the full score distribution serves
        as soft targets for training REVE's linear head.

        Args:
            eeg_test: (B, C, T) raw test EEG (NOT pre-filtered)
            ensemble: if True, use ensemble TRCA within each sub-band

        Returns:
            full_scores:  (B, 40) float32 — complete correlation scores
            top3_indices: (B, 3) int64
            top3_scores:  (B, 3) float32
        """
        weighted_corr = self._compute_weighted_corr(eeg_test, ensemble=ensemble)
        top3_scores, top3_indices = weighted_corr.topk(3, dim=-1)
        return weighted_corr.float(), top3_indices.to(torch.int64), top3_scores.float()


def leave_one_block_out_trca(eeg_data, labels, subject_ids, block_ids,
                             trial_duration_pts=600, sfreq=200.0,
                             ensemble=True, device="cuda", batch_size=256,
                             return_full_scores=False):
    """Evaluate FBTRCA using leave-one-block-out per subject.

    For each subject, each block is held out in turn while the remaining
    blocks serve as calibration data for fitting TRCA spatial filters.

    Args:
        eeg_data: (N, C, T) raw EEG
        labels: (N,) integer labels
        subject_ids: (N,) subject identifiers
        block_ids: (N,) block identifiers
        trial_duration_pts: number of timepoints to use
        sfreq: sampling frequency
        ensemble: if True, use ensemble TRCA
        device: torch device string
        batch_size: batch size for prediction
        return_full_scores: if True, also return full 40-dim scores for KD

    Returns:
        all_preds: (N, 3) top-3 predicted indices
        all_scores: (N, 3) top-3 correlation scores
        all_full_scores: (N, 40) full correlation scores (only if return_full_scores=True)
    """
    N = len(labels)
    total_T = eeg_data.shape[2]

    # Truncate to requested duration
    effective_T = min(trial_duration_pts, total_T)
    eeg = eeg_data[:, :, :effective_T]

    all_preds = torch.full((N, 3), -1, dtype=torch.int64)
    all_scores = torch.zeros(N, 3, dtype=torch.float32)
    if return_full_scores:
        all_full_scores = torch.zeros(N, 40, dtype=torch.float32)

    unique_subjects = subject_ids.unique().sort().values

    for sid in unique_subjects:
        s_mask = subject_ids == sid
        s_indices = s_mask.nonzero(as_tuple=True)[0]
        s_eeg = eeg[s_mask]          # (N_s, C, T)
        s_labels = labels[s_mask]     # (N_s,)
        s_blocks = block_ids[s_mask]  # (N_s,)

        unique_blocks = s_blocks.unique().sort().values

        for bid in unique_blocks:
            test_mask = s_blocks == bid
            cal_mask = ~test_mask

            cal_eeg = s_eeg[cal_mask].to(device)
            cal_labels = s_labels[cal_mask].to(device)
            test_eeg = s_eeg[test_mask].to(device)

            n_cal = cal_eeg.shape[0]
            n_test = test_eeg.shape[0]

            if n_cal < 80:  # need at least 2 trials/class for 40 classes
                continue

            # Fit FBTRCA on calibration data
            fbtrca = FBTRCAClassifier(sfreq=sfreq, reg=1e-6)
            fbtrca.fit(cal_eeg, cal_labels)

            # Predict test trials in batches
            test_preds_list = []
            test_scores_list = []
            test_full_list = []
            for start in range(0, n_test, batch_size):
                end = min(start + batch_size, n_test)
                batch = test_eeg[start:end]
                if return_full_scores:
                    full, preds, scores = fbtrca.predict_full(batch, ensemble=ensemble)
                    test_full_list.append(full.cpu())
                else:
                    preds, scores = fbtrca.predict(batch, ensemble=ensemble)
                test_preds_list.append(preds.cpu())
                test_scores_list.append(scores.cpu())

            test_preds = torch.cat(test_preds_list)
            test_scores = torch.cat(test_scores_list)

            # Map back to global indices
            global_test_indices = s_indices[test_mask.nonzero(as_tuple=True)[0]]
            all_preds[global_test_indices] = test_preds
            all_scores[global_test_indices] = test_scores
            if return_full_scores:
                all_full_scores[global_test_indices] = torch.cat(test_full_list)

        acc = (all_preds[s_mask, 0] == labels[s_mask]).float().mean().item()
        print(f"  S{sid.item():02d}: {s_mask.sum().item()} trials, "
              f"{'eTRCA' if ensemble else 'TRCA'} acc={acc:.1%}")

    if return_full_scores:
        return all_preds, all_scores, all_full_scores
    return all_preds, all_scores
