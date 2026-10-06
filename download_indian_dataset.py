from datasets import load_dataset
from PIL import Image
import os

print("Downloading Indian Food Dataset...")

dataset = load_dataset(
    "LALIT324rt/indian-foods-dataset"
)

output_dir = "indian_food_dataset"

os.makedirs(output_dir, exist_ok=True)

for split in dataset:
    print(f"\nProcessing {split}...")

    for i, item in enumerate(dataset[split]):

        image = item["image"]
        label = item["label"]

        class_name = dataset[split].features["label"].names[label]

        class_dir = os.path.join(
            output_dir,
            class_name
        )

        os.makedirs(class_dir, exist_ok=True)

        image_path = os.path.join(
            class_dir,
            f"{split}_{i}.jpg"
        )

        image.convert("RGB").save(image_path)

    print(f"{split} completed!")

print("\nDataset ready!")