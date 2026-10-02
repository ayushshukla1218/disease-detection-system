"""Single-file prediction engine used by Flask API."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import joblib
from features.feature_utils import extract_features, FEATURE_NAMES, features_to_vector

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, 'models', 'saved')

DISEASE_LABELS = {
    'parkinsons':  {0: 'Healthy Control', 1: "Parkinson's Disease"},
    'respiratory': {0: 'Healthy Control', 1: 'Respiratory Disease'},
    'alzheimers':  {0: 'Healthy Control', 1: "Alzheimer's Disease"},
    'als':         {0: 'Healthy Control', 1: 'ALS (Motor Neuron Disease)'},
    'depression':  {0: 'Healthy Control', 1: 'Depression'},
}

_model_cache = {}

def _load_model(disease):
    if disease in _model_cache:
        return _model_cache[disease]
    mp = os.path.join(MODEL_DIR, f'{disease}_model.pkl')
    sp = os.path.join(MODEL_DIR, f'{disease}_scaler.pkl')
    if not os.path.exists(mp):
        raise FileNotFoundError(f"Model not found for {disease}. Train first.")
    raw = joblib.load(mp)
    # Support both plain model and dict-wrapped model {'model': pipeline, ...}
    model  = raw['model'] if isinstance(raw, dict) else raw
    scaler = joblib.load(sp) if os.path.exists(sp) else None
    _model_cache[disease] = (model, scaler)
    return model, scaler

def predict(audio_path, disease):
    """
    Returns dict: {label, label_name, confidence, all_probs, features_used}
    """
    disease = disease.lower().strip()
    if disease not in DISEASE_LABELS:
        raise ValueError(f"Unknown disease: {disease}. Choose from {list(DISEASE_LABELS.keys())}")

    feats = extract_features(audio_path)
    x = features_to_vector(feats).reshape(1, -1)

    model, scaler = _load_model(disease)
    if scaler is not None:
        x = scaler.transform(x)

    pred_label = int(model.predict(x)[0])
    proba = model.predict_proba(x)[0] if hasattr(model, 'predict_proba') else None

    confidence = float(proba[pred_label]) if proba is not None else None
    all_probs  = {DISEASE_LABELS[disease][i]: round(float(p), 4)
                  for i, p in enumerate(proba)} if proba is not None else {}

    # Low-confidence threshold: if max prob < 0.6, flag as uncertain
    uncertain = confidence is not None and confidence < 0.60

    return {
        'disease':      disease,
        'label':        pred_label,
        'label_name':   DISEASE_LABELS[disease][pred_label],
        'confidence':   round(confidence * 100, 1) if confidence else None,
        'uncertain':    uncertain,
        'all_probs':    all_probs,
        'top_features': {k: round(float(feats.get(k, 0)), 4)
                         for k in ['pitch_mean', 'pitch_std', 'rms_mean', 'zcr_mean',
                                   'mfcc_1_mean', 'spectral_centroid_mean']},
    }
