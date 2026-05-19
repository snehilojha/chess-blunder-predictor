import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import shap
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix,
)


def shap_importance(model, X_test) -> dict:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    mean_abs = dict(zip(X_test.columns, abs(shap_values).mean(axis=0).tolist()))
    return dict(sorted(mean_abs.items(), key=lambda x: x[1], reverse=True))


def evaluate(model, X_test, y_test) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    return {
        "roc_auc": round(float(roc_auc), 4),
        "avg_precision": round(float(avg_prec), 4),
        "precision": round(float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0, 4),
        "recall": round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist()},
    }
