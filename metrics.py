from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def evaluate_predictions(y_true, y_pred, y_prob=None):
    yt = pd.Series(y_true).reset_index(drop=True)
    yp = pd.Series(y_pred).reset_index(drop=True)
    if len(yt) != len(yp) or len(yt) == 0:
        raise ValueError("y_true and y_pred must have the same non-zero length.")
    valid = yt.notna() & yp.notna()
    yt, yp = yt[valid], yp[valid]
    if len(yt) == 0:
        raise ValueError("No valid rows remain after removing missing labels.")
    labels = sorted(pd.unique(pd.concat([yt, yp]).astype(str)).tolist())
    yt_s, yp_s = yt.astype(str), yp.astype(str)
    average = "binary" if len(labels) == 2 and set(labels) <= {"0", "1"} else ("binary" if len(labels) == 2 and set(labels) <= {"False", "True"} else "weighted")
    pos_label = "1" if "1" in labels else ("True" if "True" in labels else labels[-1])
    precision = precision_score(yt_s, yp_s, average=average, pos_label=pos_label, zero_division=0)
    recall = recall_score(yt_s, yp_s, average=average, pos_label=pos_label, zero_division=0)
    f1 = f1_score(yt_s, yp_s, average=average, pos_label=pos_label, zero_division=0)
    auc = None
    if y_prob is not None:
        probs = pd.to_numeric(pd.Series(y_prob).reset_index(drop=True), errors="coerce")
        probs = probs[valid]
        if len(probs) == len(yt) and yt.nunique() == 2 and probs.notna().all() and probs.between(0, 1).all():
            try:
                # Treat the numerically positive / final sorted class as positive.
                binary_true = (yt_s == pos_label).astype(int)
                auc = float(roc_auc_score(binary_true, probs))
            except ValueError:
                auc = None
    cm_labels = sorted(pd.unique(pd.concat([yt_s, yp_s])).tolist())
    cm = confusion_matrix(yt_s, yp_s, labels=cm_labels)
    return {
        "accuracy": float(accuracy_score(yt_s, yp_s)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "roc_auc": auc, "confusion_matrix": cm.tolist(), "labels": cm_labels,
        "n": int(len(yt_s))
    }

def metric_definitions():
    return {
        "accuracy": "Share of predictions that match the known labels.",
        "precision": "Of the cases predicted positive, the share that are truly positive.",
        "recall": "Of the truly positive cases, the share the model correctly identifies.",
        "f1": "A combined measure of precision and recall using their harmonic mean.",
        "roc_auc": "How well scores rank positive cases above negative cases across thresholds; 0.5 is chance-like for a balanced binary setup."
    }

def train_baseline(df, target):
    if target not in df.columns:
        raise ValueError("Target column does not exist.")
    work = df.dropna(subset=[target]).copy()
    if work[target].nunique() != 2:
        raise ValueError("Baseline mode currently supports binary targets only.")
    y_raw = work.pop(target)
    if len(work) < 20:
        raise ValueError("At least 20 rows with a non-missing target are required.")
    # Encode target to 0/1 so positive class is consistent.
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y_raw.astype(str)), index=work.index)
    if y.value_counts().min() < 2:
        raise ValueError("Each target class needs at least two examples.")
    numeric = work.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in work.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    if not transformers:
        raise ValueError("No usable feature columns were found.")
    preprocessor = ColumnTransformer(transformers)
    model = Pipeline([("preprocess", preprocessor), ("classifier", LogisticRegression(max_iter=1000, class_weight=None))])
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(work, y, test_size=0.2, random_state=42, stratify=stratify)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    # return original target labels for human-readable reports, numeric predictions for metrics
    return pd.Series(y_test.values), pd.Series(pred), pd.Series(prob), model, X_test, pd.Series(y_test.values)
