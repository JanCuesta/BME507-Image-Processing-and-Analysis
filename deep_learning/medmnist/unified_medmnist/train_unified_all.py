import argparse
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler

import medmnist
from medmnist import INFO


DATASETS_2D = [
    "pathmnist",
    "chestmnist",
    "dermamnist",
    "octmnist",
    "pneumoniamnist",
    "retinamnist",
    "breastmnist",
    "bloodmnist",
    "tissuemnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
]

DATASETS_3D = [
    "organmnist3d",
    "nodulemnist3d",
    "adrenalmnist3d",
    "fracturemnist3d",
    "vesselmnist3d",
    "synapsemnist3d",
]

ALL_DATASETS = DATASETS_2D + DATASETS_3D


class MedDatasetWrapper(Dataset):
    def __init__(self, dataset_id, split):
        self.dataset_id = dataset_id
        self.is_3d = dataset_id in DATASETS_3D

        info = INFO[dataset_id]
        DataClass = getattr(medmnist, info["python_class"])

        self.base = DataClass(
            split=split,
            download=False,
            size=28
        )

        self.task = info["task"]

    def __len__(self):
        return len(self.base.imgs)

    def __getitem__(self, idx):
        img = self.base.imgs[idx]
        label = self.base.labels[idx]

        x = torch.tensor(img, dtype=torch.float32) / 255.0

        if self.is_3d:
            # D,H,W -> C,D,H,W
            x = x.unsqueeze(0)

        else:
            # H,W -> 1,H,W
            if x.ndim == 2:
                x = x.unsqueeze(0)

            # H,W,C -> C,H,W
            elif x.ndim == 3:
                x = x.permute(2, 0, 1)

            # All 2D datasets use the same 3-channel encoder
            if x.shape[0] == 1:
                x = x.repeat(3, 1, 1)

        if self.task == "multi-label, binary-class":
            y = torch.tensor(label, dtype=torch.float32).view(-1)
        else:
            y = torch.tensor(
                int(label[0]),
                dtype=torch.long
            )

        return x, y


class Encoder2D(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(64, 2048)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class Encoder3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.fc = nn.Linear(32, 2048)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class UnifiedModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder2d = Encoder2D()
        self.encoder3d = Encoder3D()

        self.shared = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
        )

        self.heads = nn.ModuleDict({
            d: nn.Linear(
                512,
                len(INFO[d]["label"])
            )
            for d in ALL_DATASETS
        })

    def forward(self, x, dataset_id):
        if dataset_id in DATASETS_3D:
            features = self.encoder3d(x)
        else:
            features = self.encoder2d(x)

        features = self.shared(features)

        return self.heads[dataset_id](features)


def make_loader(dataset_id, steps, training=True):
    dataset = MedDatasetWrapper(
        dataset_id,
        "train" if training else "test"
    )

    batch_size = 8 if dataset_id in DATASETS_3D else 32

    if training:
        sampler = RandomSampler(
            dataset,
            replacement=True,
            num_samples=batch_size * steps
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )


def task_loss(dataset_id, outputs, labels):
    task = INFO[dataset_id]["task"]

    if task == "multi-label, binary-class":
        return nn.functional.binary_cross_entropy_with_logits(
            outputs,
            labels
        )

    return nn.functional.cross_entropy(
        outputs,
        labels
    )


def evaluate(model, dataset_id, device):
    loader = make_loader(
        dataset_id,
        steps=0,
        training=False
    )

    model.eval()

    total = 0
    correct = 0
    loss_total = 0.0
    batches = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x, dataset_id)

            loss = task_loss(
                dataset_id,
                outputs,
                y
            )

            loss_total += loss.item()
            batches += 1

            if INFO[dataset_id]["task"] == "multi-label, binary-class":
                pred = (
                    torch.sigmoid(outputs) >= 0.5
                ).float()

                correct += (
                    pred == y
                ).sum().item()

                total += y.numel()

            else:
                pred = outputs.argmax(dim=1)

                correct += (
                    pred == y
                ).sum().item()

                total += y.numel()

    del loader

    return (
        loss_total / max(batches, 1),
        correct / max(total, 1)
    )


def main(args):
    random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Epochs:", args.epochs)
    print("Steps per dataset:", args.steps)

    model = UnifiedModel().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    print("\n===== MODEL CONFIGURATION =====")

    for d in ALL_DATASETS:
        print(
            f"{d:18s} | "
            f"{'3D' if d in DATASETS_3D else '2D'} | "
            f"classes={len(INFO[d]['label'])} | "
            f"task={INFO[d]['task']}"
        )

    for epoch in range(args.epochs):
        model.train()

        dataset_order = ALL_DATASETS.copy()
        random.shuffle(dataset_order)

        epoch_losses = {}

        for dataset_id in dataset_order:
            loader = make_loader(
                dataset_id,
                args.steps,
                training=True
            )

            running_loss = 0.0

            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                outputs = model(
                    x,
                    dataset_id
                )

                loss = task_loss(
                    dataset_id,
                    outputs,
                    y
                )

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_losses[dataset_id] = (
                running_loss / len(loader)
            )

            del loader

        print(
            f"\nEpoch {epoch + 1}/{args.epochs}"
        )

        for d in ALL_DATASETS:
            print(
                f"{d:18s}: "
                f"{epoch_losses[d]:.4f}"
            )

    print("\n===== FINAL TEST =====")

    for dataset_id in ALL_DATASETS:
        loss, acc = evaluate(
            model,
            dataset_id,
            device
        )

        print(
            f"{dataset_id:18s} | "
            f"loss={loss:.4f} | "
            f"acc={acc:.4f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "datasets": ALL_DATASETS,
            "epochs": args.epochs,
            "steps_per_dataset": args.steps
        },
        args.out
    )

    print("\nSaved:", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=1
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1
    )

    parser.add_argument(
        "--out",
        type=str,
        default="unified_all.pt"
    )

    args = parser.parse_args()

    main(args)

