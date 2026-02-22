"""Legacy entrypoint retained for compatibility.

The old DeepVoice/ResNet/Transformer phase-based track is deprecated.
TrustCall now continues only with the RawNet track.
"""


def main():
    msg = (
        "Legacy script deprecated: src/training/train_phase3.py\n"
        "Use RawNet scripts instead:\n"
        "  1) python main.py --data_path /path/to/LibriSeVoc --model_save_path ./outputs\n"
        "  2) python train_rawnet.py --data_path \"data/ASVspoof 2019 Dataset 2/LA/LA\" --out_dir outputs\n"
    )
    print(msg)


if __name__ == "__main__":
    main()
