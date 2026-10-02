"""Train Parkinson's disease classifier with subject-wise split."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE, 'results', 'parkinsons_features.csv')
MODEL_DIR = os.path.join(BASE, 'models', 'saved')
RESULTS_DIR = os.path.join(BASE, 'results')

from features.feature_utils import FEATURE_NAMES

def subject_wise_split(df, test_ratio=0.25, random_state=42):
    """Split subjects so no subject spans train/test."""
    rng = np.random.RandomState(random_state)
    subjects = df['subject_id'].unique()
    rng.shuffle(subjects)
    n_test = max(1, int(len(subjects) * test_ratio))
    test_subjects = subjects[:n_test]
    train_idx = df[~df['subject_id'].isin(test_subjects)].index
    test_idx  = df[ df['subject_id'].isin(test_subjects)].index
    return train_idx, test_idx, test_subjects

def evaluate(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    results = {
        'accuracy':    round(float(accuracy_score(y_true, y_pred)), 4),
        'precision':   round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'recall':      round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        'specificity': round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0, 4),
        'f1':          round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        'confusion_matrix': cm.tolist(),
    }
    if y_prob is not None:
        try:
            results['roc_auc'] = round(float(roc_auc_score(y_true, y_prob)), 4)
        except Exception:
            results['roc_auc'] = None
    return results

def run():
    if not os.path.exists(CSV):
        print(f"CSV not found: {CSV}\nRun feature extraction first.")
        return None

    df = pd.read_csv(CSV).dropna()
    print(f"Dataset: {len(df)} recordings, {df['label_name'].value_counts().to_dict()}")

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df['label'].values

    train_idx, test_idx, test_subjects = subject_wise_split(df)
    print(f"Train: {len(train_idx)} | Test: {len(test_idx)} | Test subjects: {list(test_subjects)}")

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight='balanced'),
        'SVM':          SVC(kernel='rbf', C=10, gamma='scale', probability=True, class_weight='balanced', random_state=42),
        'GradientBoost': GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
    }

    best_f1 = -1
    best_name = None
    all_results = {}

    for name, clf in models.items():
        X_tr = X_train_s if name == 'SVM' else X_train
        X_te = X_test_s  if name == 'SVM' else X_test
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1] if hasattr(clf, 'predict_proba') else None
        metrics = evaluate(y_test, y_pred, y_prob)
        all_results[name] = metrics
        print(f"\n{name}: acc={metrics['accuracy']} f1={metrics['f1']} auc={metrics.get('roc_auc','n/a')}")
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_name = name

    best_clf = models[best_name]
    print(f"\nBest model: {best_name} (F1={best_f1:.4f})")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path  = os.path.join(MODEL_DIR, 'parkinsons_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'parkinsons_scaler.pkl')
    X_use = X_train_s if best_name == 'SVM' else X_train
    best_clf.fit(X_use, y_train)
    joblib.dump(best_clf, model_path)
    joblib.dump(scaler if best_name == 'SVM' else None, scaler_path)

    output = {
        'disease': 'Parkinson\'s Disease',
        'dataset': 'MDVR-KCL',
        'n_train': int(len(train_idx)),
        'n_test':  int(len(test_idx)),
        'best_model': best_name,
        'all_results': all_results,
        'best_result': all_results[best_name],
        'feature_importance': {},
    }
    if hasattr(best_clf, 'feature_importances_'):
        fi = dict(zip(FEATURE_NAMES, best_clf.feature_importances_.tolist()))
        output['feature_importance'] = dict(sorted(fi.items(), key=lambda x: -x[1])[:10])

    out_json = os.path.join(RESULTS_DIR, 'parkinsons_results.json')
    with open(out_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_json}")
    print(f"Model saved:   {model_path}")
    return output

if __name__ == '__main__':
    run()
