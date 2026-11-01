import os
import time
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


SEED = 42
DATA_ROOT = "src/datasets"
MODEL_DIR = "src/models/CIFAR10"
MODEL_PATH = os.path.join(MODEL_DIR, "CIFAR10_CNN.pth")

EPOCHS = 5

CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616,
)

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(
                in_channels=3,
                out_channels=64,
            ),
            ConvBlock(
                in_channels=64,
                out_channels=128,
            ),
            ConvBlock(
                in_channels=128,
                out_channels=256,
            ),
        )

        self.global_average_pool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(
                in_features=256,
                out_features=10,
            ),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_average_pool(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)

        return logits


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct_count += (predictions == labels).sum().item()
        sample_count += batch_size

    average_loss = total_loss / sample_count
    accuracy = correct_count / sample_count

    return average_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            predictions = logits.argmax(dim=1)
            correct_count += (predictions == labels).sum().item()
            sample_count += batch_size

        average_loss = total_loss / sample_count
        accuracy = correct_count / sample_count

        return average_loss, accuracy


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_transform = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    train_dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        transform=train_transform,
        download=True,
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=False,
        transform=test_transform,
        download=True,
    )

    print("Train data:", len(train_dataset))
    print("Test data:", len(test_dataset))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU를 사용할 수 없습니다."
        )

    device = torch.device("cuda")
    model = CIFAR10CNN().to(device)

    print("Device:", device)
    print(model)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Total parameters:", parameter_count)
    print("Trainable parameters:", trainable_parameter_count)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    warmup_input = torch.zeros(64, 3, 32, 32, device=device)

    model.eval()

    with torch.no_grad():
        for _ in range(3):
            _ = model(warmup_input)

    torch.cuda.synchronize()

    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    torch.cuda.synchronize()
    training_start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        torch.cuda.synchronize()
        epoch_start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start_time

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: "
            f"{train_accuracy * 100:.2f}% | "
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: "
            f"{test_accuracy * 100:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    torch.cuda.synchronize()
    total_training_time = time.perf_counter() - training_start_time
    print(f"Total GPU training time: {total_training_time:.2f}s")

    os.makedirs(MODEL_DIR, exist_ok=True)

    save_data = {
        "model_name": "CIFAR10CNN",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_accuracy": test_accuracy_history[-1],
        "epochs": EPOCHS,
        "class_names": CIFAR10_CLASSES,
        "mean": CIFAR10_MEAN,
        "std": CIFAR10_STD,
    }

    torch.save(
        save_data,
        MODEL_PATH,
    )

    print(f"Model saved: {MODEL_PATH}")

    loaded_model = CIFAR10CNN()

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    loaded_model.load_state_dict(state_dict["model_state_dict"])
    loaded_model.to(device)
    loaded_model.eval()

    print("모델 이름:", state_dict["model_name"])
    print("Epoch 체크포인트:", state_dict["epochs"])
    print(f"마지막 테스트 정확도: {state_dict['test_accuracy'] * 100:.2f}%")

    epoch_axis = range(1, EPOCHS + 1)

    plt.figure()

    plt.plot(
        epoch_axis,
        train_loss_history,
        marker="o",
        label="Train loss",
    )

    plt.plot(
        epoch_axis,
        test_loss_history,
        marker="o",
        label="Test loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Loss")
    plt.grid()
    plt.legend()

    plt.figure()

    plt.plot(
        epoch_axis,
        np.array(train_accuracy_history) * 100,
        marker="o",
        label="Train accuracy",
    )

    plt.plot(
        epoch_axis,
        np.array(test_accuracy_history) * 100,
        marker="o",
        label="Test accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("CNN Accuracy")
    plt.grid()
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
