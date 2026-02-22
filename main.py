import argparse
import os
import random
import warnings

import librosa
import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core_scripts.startup_config import set_random_seed
from model import RawNet

warnings.filterwarnings("ignore")

SAMPLE_RATE = 24000


class Dataset_LibriSeVoc(Dataset):
    """LibriSeVoc dataset with fixed train/dev/test split and binary+multi labels."""

    def __init__(self, dataset_path, split="train"):
        self.dataset_path = dataset_path
        self.split = split

        y_list_train, path_list_train = [], []
        y_list_dev, path_list_dev = [], []
        y_list_test, path_list_test = [], []

        subset_names = sorted(os.listdir(dataset_path))

        # Class 0: genuine (gt)
        for subset_name in subset_names:
            if subset_name.startswith("gt"):
                subset_path = os.path.join(dataset_path, subset_name)
                files = sorted(os.listdir(subset_path))
                path_list = [os.path.join(subset_path, fn) for fn in files]
                y_list = [0] * len(path_list)

                y_list_train.extend(y_list[0:7920])
                y_list_dev.extend(y_list[7920:10560])
                y_list_test.extend(y_list[10560:13201])

                path_list_train.extend(path_list[0:7920])
                path_list_dev.extend(path_list[7920:10560])
                path_list_test.extend(path_list[10560:13201])

        # Classes 1..6: vocoder classes (all fake)
        class_idx = 1
        for subset_name in subset_names:
            if not subset_name.startswith("gt"):
                subset_path = os.path.join(dataset_path, subset_name)
                files = sorted(os.listdir(subset_path))
                path_list = [os.path.join(subset_path, fn) for fn in files]
                y_list = [class_idx] * len(path_list)

                y_list_train.extend(y_list[0:7920])
                y_list_dev.extend(y_list[7920:10560])
                y_list_test.extend(y_list[10560:13201])

                path_list_train.extend(path_list[0:7920])
                path_list_dev.extend(path_list[7920:10560])
                path_list_test.extend(path_list[10560:13201])

                class_idx += 1

        self.y_list_train = y_list_train
        self.path_list_train = path_list_train
        self.y_list_dev = y_list_dev
        self.path_list_dev = path_list_dev
        self.y_list_test = y_list_test
        self.path_list_test = path_list_test

        print(f"Load data from {self.dataset_path} ({self.split})")

    def __len__(self):
        if self.split == "train":
            return len(self.path_list_train)
        if self.split == "dev":
            return len(self.path_list_dev)
        if self.split == "test":
            return len(self.path_list_test)
        raise ValueError(f"Unknown split: {self.split}")

    def __getitem__(self, index):
        cut = SAMPLE_RATE * 4

        if self.split == "train":
            path = self.path_list_train[index]
            y_multi = self.y_list_train[index]
        elif self.split == "dev":
            path = self.path_list_dev[index]
            y_multi = self.y_list_dev[index]
        elif self.split == "test":
            path = self.path_list_test[index]
            y_multi = self.y_list_test[index]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        x, _ = librosa.load(path, sr=None)
        x_pad = pad(x, cut)
        x_inp = Tensor(x_pad)

        # Unified convention across RawNet scripts: 0=real, 1=fake
        y_binary = int(y_multi != 0)
        return x_inp, y_multi, y_binary


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len == 0:
        return np.zeros(max_len, dtype=np.float32)
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def evaluate_accuracy(dev_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    model.eval()

    with torch.no_grad():
        for batch_x, _, batch_y_binary in dev_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size

            batch_x = batch_x.to(device)
            batch_y_binary = batch_y_binary.view(-1).type(torch.int64).to(device)

            batch_out_binary, _ = model(batch_x)
            _, batch_pred = batch_out_binary.max(dim=1)
            num_correct += (batch_pred == batch_y_binary).sum(dim=0).item()

    return 100 * (num_correct / num_total)


def train_epoch(train_loader, model, optim, device, lamda):
    running_loss = 0.0
    num_correct_binary = 0.0
    num_correct_multi = 0.0
    num_total = 0.0
    out_write = ""

    model.train()

    # Model outputs log-softmax, so NLLLoss is the correct objective.
    criterion_binary = nn.NLLLoss()
    criterion_multi = nn.NLLLoss()

    for batch_x, batch_y_multi, batch_y_binary in tqdm(train_loader, total=len(train_loader)):
        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        batch_y_binary = batch_y_binary.view(-1).type(torch.int64).to(device)
        batch_y_multi = batch_y_multi.view(-1).type(torch.int64).to(device)

        batch_out_binary, batch_out_multi = model(batch_x)

        batch_loss = (
            lamda * criterion_binary(batch_out_binary, batch_y_binary)
            + (1 - lamda) * criterion_multi(batch_out_multi, batch_y_multi)
        )

        _, batch_pred_binary = batch_out_binary.max(dim=1)
        num_correct_binary += (batch_pred_binary == batch_y_binary).sum(dim=0).item()

        _, batch_pred_multi = batch_out_multi.max(dim=1)
        num_correct_multi += (batch_pred_multi == batch_y_multi).sum(dim=0).item()

        running_loss += batch_loss.item() * batch_size

        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        out_write = (
            "training multi accuracy: {:.2f}, training binary accuracy: {:.2f}".format(
                (num_correct_multi / num_total) * 100,
                (num_correct_binary / num_total) * 100,
            )
        )

    running_loss /= num_total
    train_accuracy = ((num_correct_binary + num_correct_multi) / num_total) * 50
    return running_loss, train_accuracy, out_write


def build_arg_parser():
    parser = argparse.ArgumentParser(description="TrustCall RawNet Training")
    parser.add_argument("--data_path", type=str, default="/your/path/to/LibriSeVoc/")
    parser.add_argument("--model_save_path", type=str, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_random_seed(args.seed)

    train_set = Dataset_LibriSeVoc(split="train", dataset_path=args.data_path)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)

    dev_set = Dataset_LibriSeVoc(split="dev", dataset_path=args.data_path)
    dev_dataloader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    with open("model_config_RawNet.yaml", "r") as f_yaml:
        cfg = yaml.safe_load(f_yaml)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = RawNet(cfg["model"], device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lamda = 0.5

    os.makedirs(args.model_save_path, exist_ok=True)

    best_acc = -1.0
    best_path = os.path.join(args.model_save_path, "best_model.pth")

    for epoch in range(args.num_epochs):
        running_loss, _, out_write = train_epoch(
            train_dataloader, model, optimizer, device, lamda=lamda
        )
        valid_accuracy = evaluate_accuracy(dev_dataloader, model, device)

        print(out_write)
        print(
            "epoch: {} -loss: {:.6f} - valid binary accuracy: {:.2f}".format(
                epoch, running_loss, valid_accuracy
            )
        )

        if valid_accuracy > best_acc:
            best_acc = valid_accuracy
            torch.save(model.state_dict(), best_path)
            print(f"best model found at epoch {epoch} -> {best_path}")

        torch.save(model.state_dict(), os.path.join(args.model_save_path, f"epoch_{epoch}.pth"))

    print(f"Training complete. Best validation binary accuracy: {best_acc:.2f}")


if __name__ == "__main__":
    main()
