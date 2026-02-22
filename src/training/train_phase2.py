"""Legacy entrypoint retained for compatibility.

The project now uses the RawNet training track at the repository root.
Use `python main.py ...` for LibriSeVoc or `python train_rawnet.py ...`
for ASVspoof-focused training.
"""


def main():
    msg = (
        "Legacy script deprecated: src/training/train_phase2.py\n"
        "Use RawNet scripts instead:\n"
        "  1) python main.py --data_path /path/to/LibriSeVoc --model_save_path ./outputs\n"
        "  2) python train_rawnet.py --data_path \"data/ASVspoof 2019 Dataset 2/LA/LA\" --out_dir outputs\n"
    )
    print(msg)


if __name__ == "__main__":
    main()
