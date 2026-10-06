# NutriVision - Food-101 data preparation utility
import os
from collections import defaultdict
from shutil import copy, copytree, rmtree

DATASET_DIR = "Food 101 dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
META_DIR = os.path.join(DATASET_DIR, "meta")


def prepare_data(txt_file, src, dest):
    if os.path.exists(dest):
        print(f"{dest}/ already exists. Skipping...")
        return

    classes_images = defaultdict(list)
    with open(txt_file, 'r') as txt:
        paths = [read.strip() for read in txt.readlines()]
        for p in paths:
            food = p.split('/')
            classes_images[food[0]].append(food[1] + '.jpg')

    for food in classes_images.keys():
        print(f"Copying images into {food}")
        os.makedirs(os.path.join(dest, food), exist_ok=True)
        for i in classes_images[food]:
            src_path = os.path.join(src, food, i)
            dst_path = os.path.join(dest, food, i)
            if not os.path.exists(dst_path):  # Only copy if it doesn't exist
                try:
                    copy(src_path, dst_path)
                except Exception as e:
                    print(f"Failed to copy {src_path}: {e}")
    print(f"Copying Done for {dest}/!")


def dataset_mini(food_list, src, dest):
    if os.path.exists(dest):
        rmtree(dest)
    os.makedirs(dest)
    for food_item in food_list:
        print(f"Copying images into {food_item}")
        copytree(os.path.join(src, food_item), os.path.join(dest, food_item))


if __name__ == "__main__":
    # Prepare full train/test splits
    print("Creating train data...")
    prepare_data(os.path.join(META_DIR, 'train.txt'), IMAGES_DIR, 'train')

    print("Creating test data...")
    prepare_data(os.path.join(META_DIR, 'test.txt'), IMAGES_DIR, 'test')

    # Create mini datasets (3-class example)
    food_list = ['apple_pie', 'pizza', 'omelette']
    print("Creating train_mini...")
    dataset_mini(food_list, 'train', 'train_mini')

    print("Creating test_mini...")
    dataset_mini(food_list, 'test', 'test_mini')
