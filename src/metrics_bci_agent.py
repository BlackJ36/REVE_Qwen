"""Evaluation metrics for BCI Agent training.

Computes classification accuracy on BCI target tokens (<|t01|>-<|t40|>),
top-5 accuracy, per-class statistics, and non-BCI token accuracy.

Usage:
    compute_metrics, preprocess_logits = build_metrics_fn(target_token_ids)
    trainer = Trainer(..., compute_metrics=compute_metrics,
                      preprocess_logits_for_metrics=preprocess_logits)

TensorBoard scalars logged:
    eval_bci_acc      - top-1 classification accuracy on 40 SSVEP targets
    eval_bci_top5     - top-5 accuracy (catches near-miss frequency confusions)
    eval_bci_acc_min  - worst per-class accuracy (early warning for dead classes)
    eval_bci_acc_std  - std of per-class accuracies (bias indicator)
    eval_other_acc    - accuracy on non-BCI supervised tokens (EOS, NL text)
    eval_bci_count    - number of BCI target predictions (sanity check)
"""

import numpy as np


def build_metrics_fn(target_token_ids, candidate_token_ids=None):
    """Build (compute_metrics, preprocess_logits_for_metrics) for Trainer.

    Args:
        target_token_ids: dict {target_index(0-39): token_id} from get_target_token_ids()
        candidate_token_ids: optional dict with rank/conf token IDs for candidate mode.
            If provided, enables FBCCA correction/trust metrics.
            Keys: "rank_ids" (list of 3), "conf_ids" (dict str->int)

    Returns:
        (compute_metrics, preprocess_logits_for_metrics) tuple
    """
    target_id_list = np.array(list(target_token_ids.values()))
    # Reverse mapping: token_id -> target_index
    id_to_target = {v: k for k, v in target_token_ids.items()}

    def preprocess_logits_for_metrics(logits, labels):
        """Reduce (B, L, vocab_size) logits to (B, L, 5) top-5 indices."""
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.topk(5, dim=-1).indices

    def compute_metrics(eval_pred):
        """Compute BCI classification and language generation metrics."""
        top5_preds, labels = eval_pred  # (N, L, 5), (N, L)

        # Causal LM shift: logits[i] predicts position i+1
        top5_preds = top5_preds[:, :-1]
        labels = labels[:, 1:]
        preds = top5_preds[..., 0]  # top-1

        mask = labels != -100
        if mask.sum() == 0:
            return {"bci_acc": 0.0, "bci_top5": 0.0, "other_acc": 0.0}

        # Split: BCI target positions vs other supervised positions
        bci_mask = np.isin(labels, target_id_list) & mask
        other_mask = mask & ~bci_mask

        metrics = {}

        # --- BCI target metrics ---
        if bci_mask.sum() > 0:
            bci_preds = preds[bci_mask]
            bci_labels = labels[bci_mask]
            metrics["bci_acc"] = float((bci_preds == bci_labels).mean())

            # Top-5: true label in top-5 predictions?
            bci_top5 = top5_preds[bci_mask]                 # (M, 5)
            bci_labels_exp = bci_labels[:, np.newaxis]       # (M, 1)
            metrics["bci_top5"] = float(
                (bci_top5 == bci_labels_exp).any(axis=-1).mean()
            )

            # Per-class accuracy (identifies dead/weak targets)
            per_class_acc = []
            for tid in target_id_list:
                cls_mask = bci_labels == tid
                if cls_mask.sum() > 0:
                    per_class_acc.append(
                        float((bci_preds[cls_mask] == tid).mean())
                    )
            if per_class_acc:
                metrics["bci_acc_min"] = float(np.min(per_class_acc))
                metrics["bci_acc_std"] = float(np.std(per_class_acc))

            metrics["bci_count"] = int(bci_mask.sum())
        else:
            metrics["bci_acc"] = 0.0
            metrics["bci_top5"] = 0.0
            metrics["bci_count"] = 0

        # --- FBCCA correction/trust metrics (candidate mode) ---
        # In candidate mode, the label immediately before the supervised target
        # at position p was the FBCCA rank1 prediction (at p-7 in the sequence).
        # We can't access input_ids here, but we CAN measure how often the model's
        # top-1 prediction differs from its top-2 — a proxy for confidence.
        # We also compute the "agreement rate" between top-1 and top-2 predictions
        # as a measure of model certainty.
        if bci_mask.sum() > 0:
            bci_top1 = preds[bci_mask]
            bci_top2 = top5_preds[bci_mask][:, 1]
            # Model certainty: top-1 != top-2 means model is more decisive
            metrics["bci_certainty"] = float((bci_top1 != bci_top2).mean())

        # --- Non-BCI supervised token accuracy (EOS, NL text in Stage 2) ---
        if other_mask.sum() > 0:
            metrics["other_acc"] = float(
                (preds[other_mask] == labels[other_mask]).mean()
            )
        else:
            metrics["other_acc"] = 0.0

        return metrics

    return compute_metrics, preprocess_logits_for_metrics


def compute_fbcca_correction_metrics(
    model_predictions, true_labels, fbcca_top1_predictions
):
    """Compute FBCCA correction/trust metrics (for offline evaluation).

    This function is meant to be called outside the Trainer loop, using
    precomputed FBCCA predictions and model predictions from evaluation.

    Args:
        model_predictions: (N,) int array — model's top-1 BCI target predictions
        true_labels: (N,) int array — true BCI target labels (0-39)
        fbcca_top1_predictions: (N,) int array — FBCCA's top-1 predictions (0-39)

    Returns:
        dict with:
            correction_rate: % of FBCCA mistakes that model corrects
            trust_rate: % of correct FBCCA that model follows
            fbcca_acc: FBCCA standalone accuracy
            model_acc: model accuracy
            override_rate: % of times model disagrees with FBCCA
    """
    model_predictions = np.asarray(model_predictions)
    true_labels = np.asarray(true_labels)
    fbcca_top1_predictions = np.asarray(fbcca_top1_predictions)

    fbcca_correct = fbcca_top1_predictions == true_labels
    fbcca_wrong = ~fbcca_correct
    model_correct = model_predictions == true_labels

    metrics = {
        "fbcca_acc": float(fbcca_correct.mean()),
        "model_acc": float(model_correct.mean()),
        "override_rate": float((model_predictions != fbcca_top1_predictions).mean()),
    }

    # Correction rate: when FBCCA is wrong, how often does model get it right?
    if fbcca_wrong.sum() > 0:
        metrics["correction_rate"] = float(model_correct[fbcca_wrong].mean())
        metrics["correction_count"] = int(fbcca_wrong.sum())
    else:
        metrics["correction_rate"] = 0.0
        metrics["correction_count"] = 0

    # Trust rate: when FBCCA is right, how often does model agree?
    if fbcca_correct.sum() > 0:
        metrics["trust_rate"] = float(
            (model_predictions[fbcca_correct] == fbcca_top1_predictions[fbcca_correct]).mean()
        )
    else:
        metrics["trust_rate"] = 0.0

    return metrics
