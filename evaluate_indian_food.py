import os
import json
import torch
import torch.nn as nn

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import numpy as np


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "indian_food_dataset"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "indian_food_mobilenet.pt"
)

CLASS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "indian_food_classes.json"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_PATH, "r") as f:

    class_data = json.load(f)


if isinstance(class_data, dict):

    class_names = class_data["classes"]

else:

    class_names = class_data


print("\nClasses:")

for i, name in enumerate(class_names):

    print(i, "->", name)


# ============================================================
# MODEL
# ============================================================

def create_indian_food_model(n_classes):

    model = models.mobilenet_v3_small(
        weights=None
    )

    num_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        num_features,
        n_classes
    )

    return model

    def forward(self, x):

        return self.model(x)


# ============================================================
# LOAD MODEL
# ============================================================

model = create_indian_food_model(
    len(class_names)
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


if (
    isinstance(checkpoint, dict)
    and
    "model_state_dict" in checkpoint
):

    state_dict = checkpoint["model_state_dict"]

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict
)


model = model.to(device)

model.eval()


print("\nModel loaded successfully.")


# ============================================================
# TRANSFORMS
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        [0.485, 0.456, 0.406],

        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD DATASET
# ============================================================

dataset = datasets.ImageFolder(
    DATASET_DIR,
    transform=transform
)


print(
    "\nTotal images:",
    len(dataset)
)

print(
    "Dataset classes:",
    dataset.classes
)


# ============================================================
# CHECK CLASS ORDER
# ============================================================

if dataset.classes != class_names:

    print("\nWARNING!")

    print(
        "Dataset class order and "
        "JSON class order are different."
    )

    print("\nDataset:")

    print(dataset.classes)

    print("\nJSON:")

    print(class_names)

    print(
        "\nThis can make evaluation INVALID."
    )


# ============================================================
# DATALOADER
# ============================================================

loader = DataLoader(

    dataset,

    batch_size=32,

    shuffle=False,

    num_workers=0
)


# ============================================================
# PREDICTION
# ============================================================

all_predictions = []

all_labels = []


print("\nRunning evaluation...")


with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(
            images
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )


        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


print("\n")
print("=" * 60)
print("OVERALL RESULTS")
print("=" * 60)


print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)


report = classification_report(

    all_labels,

    all_predictions,

    target_names=class_names,

    digits=4,

    zero_division=0
)


print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    all_labels,

    all_predictions
)


print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)


print(cm)


# ============================================================
# MOST CONFUSED CLASSES
# ============================================================

print("\n")
print("=" * 60)
print("MOST CONFUSED CLASSES")
print("=" * 60)


confusions = []


for i in range(len(class_names)):

    for j in range(len(class_names)):

        if i != j:

            count = cm[i][j]

            if count > 0:

                confusions.append(
                    (
                        count,
                        class_names[i],
                        class_names[j]
                    )
                )


confusions.sort(
    reverse=True
)


for count, actual, predicted in confusions[:15]:

    print(
        f"{actual} -> {predicted}: "
        f"{count} images"
    )


print("\nEvaluation complete.")