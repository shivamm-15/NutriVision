import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split


# =========================
# SETTINGS
# =========================

DATA_DIR = "indian_food_dataset"
MODEL_DIR = "models"

BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():

    print("Using device:", DEVICE)

    # =========================
    # TRANSFORMS
    # =========================

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # =========================
    # DATASET
    # =========================

    dataset = datasets.ImageFolder(
        DATA_DIR,
        transform=train_transform
    )

    class_names = dataset.classes

    print("\nClasses:")
    for i, name in enumerate(class_names):
        print(i, name)

    print("\nTotal images:", len(dataset))

    # =========================
    # TRAIN / VALIDATION SPLIT
    # =========================

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    # Validation transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # =========================
    # MODEL
    # =========================

    print("\nLoading MobileNetV3...")

    model = models.mobilenet_v3_small(
        weights="DEFAULT"
    )

    num_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        num_features,
        len(class_names)
    )

    model = model.to(DEVICE)

    # =========================
    # LOSS + OPTIMIZER
    # =========================

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # =========================
    # TRAIN
    # =========================

    best_accuracy = 0

    for epoch in range(EPOCHS):

        model.train()

        correct = 0
        total = 0
        running_loss = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

        train_accuracy = (
            100 * correct / total
        )

        # =========================
        # VALIDATION
        # =========================

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                _, predicted = torch.max(
                    outputs,
                    1
                )

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

        val_accuracy = (
            100 * correct / total
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )

        # =========================
        # SAVE BEST MODEL
        # =========================

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            os.makedirs(
                MODEL_DIR,
                exist_ok=True
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names
                },
                os.path.join(
                    MODEL_DIR,
                    "indian_food_mobilenet.pt"
                )
            )

            print("✓ Best model saved!")

    # =========================
    # SAVE CLASS NAMES
    # =========================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    with open(
        os.path.join(
            MODEL_DIR,
            "indian_food_classes.json"
        ),
        "w"
    ) as f:

        json.dump(
            class_names,
            f,
            indent=4
        )

    print("\n========================")
    print("Training Complete!")
    print("========================")

    print(
        f"Best Validation Accuracy: "
        f"{best_accuracy:.2f}%"
    )


# =========================
# WINDOWS SAFE ENTRY POINT
# =========================

if __name__ == "__main__":
    main()