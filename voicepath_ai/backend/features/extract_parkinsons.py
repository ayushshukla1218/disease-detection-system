"""Extract features from MDVR-KCL Parkinson's dataset and save CSV."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from features.feature_utils import extract_features, FEATURE_NAMES

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'parkinsons')
OUT_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'parkinsons_features.csv')

def extract_subject_id(filename):
    m = re.match(r'(ID\d+)', os.path.basename(filename), re.IGNORECASE)
    return m.group(1).upper() if m else 'UNKNOWN'

def run():
    rows = []
    for label, subdir in [('HC', 'HC'), ('PD', 'PD')]:
        folder = os.path.join(DATA_DIR, subdir)
        if not os.path.exists(folder):
            print(f"  Missing: {folder}")
            continue
        files = [f for f in os.listdir(folder) if f.endswith('.wav')]
        print(f"  {label}: {len(files)} files")
        for fname in sorted(files):
            fpath = os.path.join(folder, fname)
            try:
                feats = extract_features(fpath)
                row = {k: feats.get(k, 0.0) for k in FEATURE_NAMES}
                row['subject_id'] = extract_subject_id(fname)
                row['filename'] = fname
                row['label'] = 1 if label == 'PD' else 0
                row['label_name'] = label
                rows.append(row)
                print(f"    OK: {fname}")
            except Exception as e:
                print(f"    SKIP {fname}: {e}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows -> {OUT_CSV}")
    print(df['label_name'].value_counts().to_string())
    return df

if __name__ == '__main__':
    run()
