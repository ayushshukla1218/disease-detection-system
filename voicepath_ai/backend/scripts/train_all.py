"""One-shot script: extract features for all diseases, then train all models."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    start = time.time()
    print("=" * 60)
    print("VoicePath AI — Full Training Pipeline")
    print("=" * 60)

    print("\n[1/2] Feature Extraction")
    print("-" * 40)

    print("\n  Parkinson's Disease (MDVR-KCL)...")
    from features.extract_parkinsons import run as extract_pd
    extract_pd()

    print("\n  Synthetic disease datasets...")
    from features.extract_synthetic import run as extract_syn
    extract_syn()

    print("\n[2/2] Model Training")
    print("-" * 40)

    print("\n  Parkinson's...")
    from models.train_parkinsons import run as train_pd
    r = train_pd()
    if r:
        print(f"  => Best: {r['best_model']} | F1={r['best_result']['f1']} | AUC={r['best_result'].get('roc_auc')}")

    for disease in ['respiratory', 'alzheimers', 'als', 'depression']:
        from models.train_disease import run as train_d
        r = train_d(disease)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Models saved in: backend/models/saved/")
    print(f"Results in:      backend/results/")
    print("=" * 60)

if __name__ == '__main__':
    main()
