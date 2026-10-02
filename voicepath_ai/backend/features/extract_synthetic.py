"""Extract features from synthetic disease datasets (Respiratory, Alzheimer's, ALS, Depression)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from features.feature_utils import extract_features, FEATURE_NAMES

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DISEASES = {
    'respiratory': ('data/respiratory', 'results/respiratory_features.csv'),
    'alzheimers':  ('data/alzheimers',  'results/alzheimers_features.csv'),
    'als':         ('data/als',         'results/als_features.csv'),
    'depression':  ('data/depression',  'results/depression_features.csv'),
}

def run(disease=None):
    targets = {disease: DISEASES[disease]} if disease else DISEASES
    for name, (data_rel, csv_rel) in targets.items():
        data_dir = os.path.join(BASE, data_rel)
        out_csv = os.path.join(BASE, csv_rel)
        if not os.path.exists(data_dir):
            print(f"  Missing: {data_dir}")
            continue

        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.wav')])
        print(f"\n[{name.upper()}] {len(files)} files in {data_dir}")
        rows = []
        for i, fname in enumerate(files):
            fpath = os.path.join(data_dir, fname)
            # Label: files named *_disease_* = 1, *_healthy_* = 0
            if '_disease_' in fname:
                label, label_name = 1, name.upper()
            elif '_healthy_' in fname:
                label, label_name = 0, 'HC'
            else:
                # fallback: first 25 = disease, rest = healthy
                label = 1 if i < 25 else 0
                label_name = name.upper() if label == 1 else 'HC'
            try:
                feats = extract_features(fpath, duration=45)
                row = {k: feats.get(k, 0.0) for k in FEATURE_NAMES}
                row['subject_id'] = f"S{i+1:03d}"
                row['filename'] = fname
                row['label'] = label
                row['label_name'] = label_name
                rows.append(row)
                if (i+1) % 10 == 0:
                    print(f"  [{i+1}/{len(files)}]")
            except Exception as e:
                print(f"  SKIP {fname}: {e}")

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"  Saved {len(df)} rows -> {out_csv}")

if __name__ == '__main__':
    import sys
    disease = sys.argv[1] if len(sys.argv) > 1 else None
    run(disease)
