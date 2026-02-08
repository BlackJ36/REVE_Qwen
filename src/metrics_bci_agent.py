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


def build_metrics_fn(target_token_ids):
    """Build (compute_metrics, preprocess_logits_for_metrics) for Trainer.

    Args:
        target_token_ids: dict {target_index(0-39): token_id} from get_target_token_ids()

    Returns:
        (compute_metrics, preprocess_logits_for_metrics) tuple
    """
    target_id_list = np.array(list(target_token_ids.values()))

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

        # --- Non-BCI supervised token accuracy (EOS, NL text in Stage 2) ---
        if other_mask.sum() > 0:
            metrics["other_acc"] = float(
                (preds[other_mask] == labels[other_mask]).mean()
            )
        else:
            metrics["other_acc"] = 0.0

        return metrics

    return compute_metrics, preprocess_logits_for_metrics
