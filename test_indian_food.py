import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

# =========================
# DEVICE
# =========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# =========================
# LOAD CLASS NAMES
# =========================

with open(
    "models/indian_food_classes.json",
    "r"
) as f:

    class_names = json.load(f)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, name)


# =========================
# LOAD MODEL
# =========================

model = models.mobilenet_v3_small(
    weights=None
)

num_features = model.classifier[-1].in_features

model.classifier[-1] = nn.Linear(
    num_features,
    len(class_names)
)

checkpoint = torch.load(
    "models/indian_food_mobilenet.pt",
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


# =========================
# IMAGE TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =========================
# GET IMAGE
# =========================

image_path = input(
    "\nEnter image path: "
)

image = Image.open(
    image_path
).convert("RGB")


# =========================
# PREDICTION
# =========================

image_tensor = transform(
    image
).unsqueeze(0)

image_tensor = image_tensor.to(DEVICE)

with torch.no_grad():

    outputs = model(
        image_tensor
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, predicted = torch.max(
        probabilities,
        dim=1
    )


prediction = class_names[
    predicted.item()
]

confidence_value = (
    confidence.item() * 100
)


# =========================
# RESULT
# =========================

print("\n==========================")
print("INDIAN FOOD PREDICTION")
print("==========================")

print(
    "Prediction:",
    prediction
)

print(
    f"Confidence: {confidence_value:.2f}%"
)

print("==========================")