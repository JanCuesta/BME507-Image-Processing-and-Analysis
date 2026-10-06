import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import medmnist
from medmnist import INFO


# ---------------------------------------------------------
# Small 2D encoder
# ---------------------------------------------------------
class Encoder2D(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(64, 2048)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ---------------------------------------------------------
# Small 3D encoder
# ---------------------------------------------------------
class Encoder3D(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        self.fc = nn.Linear(32, 2048)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ---------------------------------------------------------
# Unified model
# ---------------------------------------------------------
class UnifiedMedMNISTModel(nn.Module):
    def __init__(self, blood_classes, organ_classes):
        super().__init__()

        self.encoder2d = Encoder2D(3)
        self.encoder3d = Encoder3D(1)

        # Parameters shared by BOTH 2D and 3D tasks
        self.shared = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU()
        )

        # Dataset-specific classification heads
        self.heads = nn.ModuleDict({
            "bloodmnist": nn.Linear(512, blood_classes),
            "organmnist3d": nn.Linear(512, organ_classes)
        })

    def forward(self, x, dataset_id):
        # 2D batch: B,C,H,W
        if x.ndim == 4:
            features = self.encoder2d(x)

        # 3D batch: B,C,D,H,W
        elif x.ndim == 5:
            features = self.encoder3d(x)

        else:
            raise ValueError(f"Unexpected tensor shape: {x.shape}")

        shared_features = self.shared(features)

        return self.heads[dataset_id](shared_features)


def main(args):

    random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Epochs:", args.epochs)

    # -----------------------------------------------------
    # Load datasets
    # -----------------------------------------------------
    blood_train = medmnist.BloodMNIST(
        split="train",
        download=False,
        size=28
    )

    organ_train = medmnist.OrganMNIST3D(
        split="train",
        download=False,
        size=28
    )

    blood_test = medmnist.BloodMNIST(
        split="test",
        download=False,
        size=28
    )

    organ_test = medmnist.OrganMNIST3D(
        split="test",
        download=False,
        size=28
    )

    # -----------------------------------------------------
    # Tiny training subsets
    # -----------------------------------------------------
    blood_x = torch.tensor(
        blood_train.imgs[:args.blood_n],
        dtype=torch.float32
    ).permute(0, 3, 1, 2) / 255.0

    blood_y = torch.tensor(
        blood_train.labels[:args.blood_n],
        dtype=torch.long
    ).view(-1)

    organ_x = torch.tensor(
        organ_train.imgs[:args.organ_n],
        dtype=torch.float32
    ).unsqueeze(1) / 255.0

    organ_y = torch.tensor(
        organ_train.labels[:args.organ_n],
        dtype=torch.long
    ).view(-1)

    print("Blood training tensor:", blood_x.shape)
    print("Organ3D training tensor:", organ_x.shape)

    blood_loader = DataLoader(
        TensorDataset(blood_x, blood_y),
        batch_size=32,
        shuffle=True
    )

    organ_loader = DataLoader(
        TensorDataset(organ_x, organ_y),
        batch_size=8,
        shuffle=True
    )

    blood_classes = len(INFO["bloodmnist"]["label"])
    organ_classes = len(INFO["organmnist3d"]["label"])

    model = UnifiedMedMNISTModel(
        blood_classes,
        organ_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------
    for epoch in range(args.epochs):

        model.train()

        blood_loss = 0.0
        organ_loss = 0.0

        for images, labels in blood_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(
                images,
                dataset_id="bloodmnist"
            )

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            blood_loss += loss.item()

        for images, labels in organ_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(
                images,
                dataset_id="organmnist3d"
            )

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            organ_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Blood loss: {blood_loss / len(blood_loader):.4f} | "
            f"Organ3D loss: {organ_loss / len(organ_loader):.4f}"
        )

    # -----------------------------------------------------
    # Mixed inference using TEST samples
    # -----------------------------------------------------
    blood_test_x = torch.tensor(
        blood_test.imgs[:2],
        dtype=torch.float32
    ).permute(0, 3, 1, 2) / 255.0

    blood_test_y = torch.tensor(
        blood_test.labels[:2],
        dtype=torch.long
    ).view(-1)

    organ_test_x = torch.tensor(
        organ_test.imgs[:2],
        dtype=torch.float32
    ).unsqueeze(1) / 255.0

    organ_test_y = torch.tensor(
        organ_test.labels[:2],
        dtype=torch.long
    ).view(-1)

    mixed_samples = [
        ("bloodmnist", blood_test_x[0:1], blood_test_y[0].item()),
        ("organmnist3d", organ_test_x[0:1], organ_test_y[0].item()),
        ("bloodmnist", blood_test_x[1:2], blood_test_y[1].item()),
        ("organmnist3d", organ_test_x[1:2], organ_test_y[1].item())
    ]

    model.eval()

    print("\n===== MIXED TEST =====")

    with torch.no_grad():

        for dataset_id, sample, true_class in mixed_samples:

            sample = sample.to(device)

            output = model(
                sample,
                dataset_id=dataset_id
            )

            prediction = torch.argmax(
                output,
                dim=1
            ).item()

            labels = INFO[dataset_id]["label"]

            print(
                f"{dataset_id:15s} | "
                f"dims={sample.ndim} | "
                f"True: {true_class} ({labels[str(true_class)]}) | "
                f"Predicted: {prediction} ({labels[str(prediction)]})"
            )

    torch.save(
        model.state_dict(),
        args.out
    )

    print("\nSaved model:", args.out)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--blood_n", type=int, default=64)
    parser.add_argument("--organ_n", type=int, default=32)
    parser.add_argument("--out", type=str, default="unified_model.pt")

    args = parser.parse_args()

    main(args)
