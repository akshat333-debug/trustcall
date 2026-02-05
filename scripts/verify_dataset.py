import os
import sys

def verify_dataset(root_dir="./data/asvspoof2019", track="LA"):
    print(f"Verifying ASVspoof 2019 ({track}) dataset at: {root_dir}")
    
    base_path = os.path.join(root_dir, track)
    if not os.path.exists(base_path):
        print(f"[ERROR] Track directory not found: {base_path}")
        print("Please follow instructions in scripts/prepare_data_instructions.md")
        return False

    # Check Protocols
    proto_dir = os.path.join(base_path, f"ASVspoof2019_{track}_cm_protocols")
    if not os.path.exists(proto_dir):
        print(f"[ERROR] Protocol directory missing: {proto_dir}")
        return False
    
    required_protos = [
        f"ASVspoof2019.{track}.cm.train.trn.txt",
        f"ASVspoof2019.{track}.cm.dev.trl.txt",
        f"ASVspoof2019.{track}.cm.eval.trl.txt"
    ]
    
    missing_proto = False
    for p in required_protos:
        p_path = os.path.join(proto_dir, p)
        if not os.path.exists(p_path):
            print(f"[MISSING] Protocol file: {p}")
            missing_proto = True
        else:
            print(f"[OK] Found protocol: {p}")
    
    if missing_proto:
        return False

    # Check Audio Folders
    audio_dirs = ["train", "dev", "eval"]
    missing_audio = False
    for subset in audio_dirs:
        # Note: Directory naming convention differs slightly in unpacked zips sometimes
        # We assume standard structure
        dir_name = f"ASVspoof2019_{track}_{subset}"
        full_path = os.path.join(base_path, dir_name, "flac")
        
        if not os.path.exists(full_path):
            print(f"[MISSING] Audio directory: {full_path}")
            missing_audio = True
        else:
            count = len([f for f in os.listdir(full_path) if f.endswith(".flac")])
            print(f"[OK] Found {subset} set with {count} flac files")
            if count == 0:
                print(f"[WARNING] {subset} set is empty!")

    if missing_audio:
        return False

    print("\n[SUCCESS] Dataset structure seems correct!")
    return True

if __name__ == "__main__":
    path = "./data/asvspoof2019"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    
    if not verify_dataset(path):
        sys.exit(1)
