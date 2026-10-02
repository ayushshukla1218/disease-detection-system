"""Generic trainer for synthetic disease datasets (Respiratory, Alzheimer's, ALS, Depression)."""
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
from features.feature_utils import FEATURE_NAMES

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DISEASE_MAP = {
    'respiratory': 'Respiratory Disease',
    'alzheimers':  "Alzheimer's Disease",
    'als':         'Amyotrophic Lateral Sclerosis (ALS)',
    'depression':  'Depression',
}

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

def run(disease_key):
    csv_path = os.path.join(BASE, 'results', f'{disease_key}_features.csv')
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path).dropna()
    print(f"\n[{disease_key.upper()}] {len(df)} rows, {df['label_name'].value_counts().to_dict()}")

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df['label'].values

    # 80/20 split by subject_id
    subjects = df['subject_id'].unique()
    np.random.seed(42)
    np.random.shuffle(subjects)
    n_test = max(3, int(len(subjects) * 0.2))
    test_subs = set(subjects[:n_test])
    train_mask = ~df['subject_id'].isin(test_subs)
    test_mask  =  df['subject_id'].isin(test_subs)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight='balanced'),
        'SVM':          SVC(kernel='rbf', C=10, gamma='scale', probability=True, class_weight='balanced', random_state=42),
        'GradientBoost': GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
    }

    best_f1, best_name, all_results = -1, None, {}

    for name, clf in models.items():
        Xtr = X_train_s if name == 'SVM' else X_train
        Xte = X_test_s  if name == 'SVM' else X_test
        clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xte)
        y_prob = clf.predict_proba(Xte)[:, 1] if hasattr(clf, 'predict_proba') else None
        metrics = evaluate(y_test, y_pred, y_prob)
        all_results[name] = metrics
        print(f"  {name}: acc={metrics['accuracy']} f1={metrics['f1']} auc={metrics.get('roc_auc')}")
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_name = name

    best_clf = models[best_name]
    X_use = X_train_s if best_name == 'SVM' else X_train
    best_clf.fit(X_use, y_train)

    model_dir = os.path.join(BASE, 'models', 'saved')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_clf, os.path.join(model_dir, f'{disease_key}_model.pkl'))
    joblib.dump(scaler if best_name == 'SVM' else None, os.path.join(model_dir, f'{disease_key}_scaler.pkl'))

    output = {
        'disease': DISEASE_MAP.get(disease_key, disease_key),
        'n_train': int(len(X_train)),
        'n_test':  int(len(X_test)),
        'best_model': best_name,
        'all_results': all_results,
        'best_result': all_results[best_name],
        'feature_importance': {},
    }
    if hasattr(best_clf, 'feature_importances_'):
        fi = dict(zip(FEATURE_NAMES, best_clf.feature_importances_.tolist()))
        output['feature_importance'] = dict(sorted(fi.items(), key=lambda x: -x[1])[:10])

    result_path = os.path.join(BASE, 'results', f'{disease_key}_results.json')
    with open(result_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Best: {best_name} (F1={best_f1:.4f}) -> saved {result_path}")
    return output

if __name__ == '__main__':
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else 'respiratory'
    run(key)
