# NutriVision - Food-101 dataset visualization utility
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "food-101/images/"


def visualize_one_per_class(rows=17, cols=6):
    foods_sorted = sorted([f for f in os.listdir(DATA_DIR) if not f.startswith('.')])
    fig, ax = plt.subplots(rows, cols, figsize=(25, 25))
    fig.suptitle("Showing one random image from each class", y=1.05, fontsize=24)
    food_id = 0
    for i in range(rows):
        for j in range(cols):
            if food_id >= len(foods_sorted):
                break
            food_selected = foods_sorted[food_id]
            food_id += 1
            food_selected_images = os.listdir(os.path.join(DATA_DIR, food_selected))
            food_selected_random = np.random.choice(food_selected_images)
            img = plt.imread(os.path.join(DATA_DIR, food_selected, food_selected_random))
            ax[i][j].imshow(img)
            ax[i][j].set_title(food_selected, pad=10)
    plt.setp(ax, xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_one_per_class() 