import numpy as np
import librosa

def extract_features(audio_path, sr_target=22050, duration=60):
    """Extract 41 voice features from a WAV file."""
    try:
        y, sr = librosa.load(audio_path, sr=sr_target, duration=duration, mono=True)
    except Exception as e:
        raise ValueError(f"Cannot load {audio_path}: {e}")

    if len(y) < sr_target * 0.5:
        raise ValueError(f"Audio too short: {audio_path}")

    features = {}

    # --- Pitch features (5) ---
    f0, voiced_flag, _ = librosa.pyin(y, fmin=65, fmax=500, sr=sr)
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    if len(voiced_f0) < 5:
        voiced_f0 = np.array([100.0] * 10)
    features['pitch_mean'] = float(np.mean(voiced_f0))
    features['pitch_std'] = float(np.std(voiced_f0))
    features['pitch_median'] = float(np.median(voiced_f0))
    features['pitch_p5'] = float(np.percentile(voiced_f0, 5))
    features['pitch_p95'] = float(np.percentile(voiced_f0, 95))

    # --- MFCC features (26: mean+std of 13 coefficients) ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
        features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))

    # --- Zero Crossing Rate (2) ---
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std'] = float(np.std(zcr))

    # --- RMS Energy (2) ---
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std'] = float(np.std(rms))

    # --- Spectral Centroid (2) ---
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = float(np.mean(spec_cent))
    features['spectral_centroid_std'] = float(np.std(spec_cent))

    # --- Spectral Bandwidth (2) ---
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['spectral_bandwidth_mean'] = float(np.mean(spec_bw))
    features['spectral_bandwidth_std'] = float(np.std(spec_bw))

    # --- Spectral Rolloff (2) ---
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_mean'] = float(np.mean(rolloff))
    features['spectral_rolloff_std'] = float(np.std(rolloff))

    return features


def features_to_vector(features_dict):
    """Convert ordered feature dict to numpy array."""
    return np.array(list(features_dict.values()), dtype=np.float32)


FEATURE_NAMES = (
    ['pitch_mean', 'pitch_std', 'pitch_median', 'pitch_p5', 'pitch_p95'] +
    [f'mfcc_{i}_mean' for i in range(1, 14)] +
    [f'mfcc_{i}_std' for i in range(1, 14)] +
    ['zcr_mean', 'zcr_std', 'rms_mean', 'rms_std',
     'spectral_centroid_mean', 'spectral_centroid_std',
     'spectral_bandwidth_mean', 'spectral_bandwidth_std',
     'spectral_rolloff_mean', 'spectral_rolloff_std']
)
