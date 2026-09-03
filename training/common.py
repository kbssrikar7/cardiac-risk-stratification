"""Shared training utilities, extracted from the separate ablation/finalize
scripts written earlier this project (finalize_calibrated_xgb.py,
feature_selection.py, ablate_infarct_features.py, finalize_infarct_model.py,
retrain_on_fixed_radiomics.py, finalize_pruned_model.py all had an identical
copy of dynamic_resample() - one real implementation here instead).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42


def risk_score_label_mapping(df):
    """Maps each Risk_Score value (0-3) to its human-readable Risk_Category
    string, read directly from the data instead of assumed.

    Every finalize_*.py / pipeline.py script used
    `LabelEncoder().fit_transform(df["Risk_Score"].astype(str))` to build `y`,
    then reused `le.classes_` as the label_mapping - but classes_ from that
    call is just the sorted unique *Risk_Score strings* ('0','1','2','3'), not
    the actual risk category names. The trained model and every CV metric are
    unaffected (Risk_Score is already a clean 0-3 ordinal target), but every
    model pickled this way had label_mapping = {0: '0', 1: '1', ...} baked in,
    so any consumer mapping a predicted index back to a label via
    label_mapping[idx] got a bare digit string instead of e.g. 'Low Risk'.
    Found via E2E testing against the live /predict endpoint.
    """
    return (
        df[["Risk_Score", "Risk_Category"]]
        .drop_duplicates()
        .set_index("Risk_Score")["Risk_Category"]
        .to_dict()
    )


def dynamic_resample(X, y, random_state=RANDOM_STATE):
    unique, counts = np.unique(y, return_counts=True)
    maj_class, maj_count = unique[np.argmax(counts)], counts.max()
    under_strategy = {int(maj_class): int(maj_count * 0.5)}
    over_strategy = {int(c): int(int(maj_count * 0.5) * 0.75) for c in unique if c != maj_class}
    k_neighbors = max(1, counts.min() - 1)
    pipeline = ImbPipeline([
        ("u", RandomUnderSampler(sampling_strategy=under_strategy, random_state=random_state)),
        ("o", SMOTE(sampling_strategy=over_strategy, random_state=random_state, k_neighbors=k_neighbors)),
    ])
    return pipeline.fit_resample(X, y)


def greedy_correlation_prune(df_features, threshold=0.95):
    """Drops one member of any feature pair with |corr| > threshold, keeping
    whichever has lower mean correlation to the rest of the feature set."""
    corr = df_features.corr().abs()
    mean_corr = corr.mean()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = set()
    for col in upper.columns:
        for other in upper.index[upper[col] > threshold].tolist():
            if col in to_drop or other in to_drop:
                continue
            to_drop.add(col if mean_corr[col] >= mean_corr[other] else other)
    return [c for c in df_features.columns if c not in to_drop], sorted(to_drop)


def repeated_cv_eval(build_and_fit, X_df, y, label=None, random_state=RANDOM_STATE):
    """Repeated stratified 5x5 CV, the honest headline-metric protocol used
    throughout this project. `build_and_fit(X_train_res, y_train_res)` must
    return a fitted classifier exposing predict_proba."""
    X_s = StandardScaler().fit_transform(X_df)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=random_state)
    all_classes = sorted(np.unique(y))
    accs, f1s, baccs, aucs = [], [], [], []
    for tr_idx, val_idx in rskf.split(X_s, y):
        X_tr_res, y_tr_res = dynamic_resample(X_s[tr_idx], y[tr_idx], random_state=random_state)
        clf = build_and_fit(X_tr_res, y_tr_res)
        proba = clf.predict_proba(X_s[val_idx])
        preds = proba.argmax(1)
        y_val = y[val_idx]
        accs.append(accuracy_score(y_val, preds))
        f1s.append(f1_score(y_val, preds, average="macro", labels=all_classes, zero_division=0))
        baccs.append(balanced_accuracy_score(y_val, preds))
        try:
            aucs.append(roc_auc_score(label_binarize(y_val, classes=all_classes), proba, average="macro", multi_class="ovr"))
        except ValueError:
            pass  # a fold's val split can be missing a class at n=150; skip AUC for that fold only
    result = {"accuracy": accs, "f1_macro": f1s, "balanced_accuracy": baccs, "auc": aucs}
    if label:
        print(f"\n=== {label} (n_features={X_df.shape[1]}) ===")
        print(f"Accuracy:          {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
        print(f"F1 (macro):        {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
        print(f"Balanced accuracy: {np.mean(baccs):.3f} +/- {np.std(baccs):.3f}")
        print(f"AUC (macro OVR):   {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}  (n={len(aucs)} folds)")
    return result


def summarize(result):
    return {metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))} for metric, vals in result.items()}


def brier_calibration_report(probs_test, y_test, class_names):
    """Per-class Brier score - lower is better, 0.25 is the uninformative
    baseline for a binary one-vs-rest split."""
    report = {}
    for cls_idx, cls_name in enumerate(class_names):
        y_true_bin = (y_test == cls_idx).astype(int)
        report[str(cls_name)] = float(brier_score_loss(y_true_bin, probs_test[:, cls_idx]))
    return report
