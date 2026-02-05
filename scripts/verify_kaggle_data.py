import os
import sys

def verify_kaggle_dataset(root_dir="./data"):
    print(f"Verifying DeepVoice (Kaggle) dataset at: {root_dir}")
    
    # Expected structure:
    # root_dir/KAGGLE/AUDIO/REAL
    # root_dir/KAGGLE/AUDIO/FAKE
    
    kaggle_dir = os.path.join(root_dir, "KAGGLE")
    audio_dir = os.path.join(kaggle_dir, "AUDIO")
    
    if not os.path.exists(kaggle_dir):
        print(f"[ERROR] Directory not found: {kaggle_dir}")
        print("Required Structure:")
        print("data/")
        print("└── KAGGLE/")
        print("    └── AUDIO/")
        print("        ├── REAL/")
        print("        └── FAKE/")
        return False

    if not os.path.exists(audio_dir):
        print(f"[ERROR] AUDIO directory not found: {audio_dir}")
        return False
        
    subsets = ["REAL", "FAKE"]
    missing_subset = False
    
    for sub in subsets:
        path = os.path.join(audio_dir, sub)
        if not os.path.exists(path):
            print(f"[MISSING] Directory: {path}")
            missing_subset = True
        else:
            files = [f for f in os.listdir(path) if f.lower().endswith(('.wav', '.flac', '.mp3'))]
            count = len(files)
            print(f"[OK] {sub}: {count} audio files")
            if count == 0:
                print(f"[WARNING] {sub} folder is empty.")
    
    if missing_subset:
        return False
        
    print("\n[SUCCESS] Kaggle DeepVoice dataset structure verified!")
    return True

if __name__ == "__main__":
    if not verify_kaggle_dataset():
        sys.exit(1)
