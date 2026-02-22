"""Legacy entrypoint retained for compatibility.

Use RawNet evaluation scripts from the repository root:
  - `python eval.py --input_path sample.wav --model_path outputs/best_model.pth`
  - `python cross_eval.py --asvspoof_path /path/to/ASVspoof2019`
"""


def main():
    msg = (
        "Legacy script deprecated: src/eval/evaluate_phase2.py\n"
        "Use RawNet evaluation scripts instead:\n"
        "  1) python eval.py --input_path sample.wav --model_path outputs/best_model.pth\n"
        "  2) python cross_eval.py --asvspoof_path /path/to/ASVspoof2019 --out_dir outputs/cross_eval\n"
    )
    print(msg)


if __name__ == "__main__":
    main()
