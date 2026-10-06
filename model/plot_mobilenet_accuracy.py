# NutriVision - MobileNet training metrics visualization
import json
import matplotlib.pyplot as plt

HISTORY_PATH = 'models/food101_mobilenet_MOBILENET_training_history.json'
OUTPUT_PNG = 'models/mobilenet_accuracy_curve.png'

def plot_accuracy():
    with open(HISTORY_PATH, 'r') as f:
        history = json.load(f)
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('MobileNet Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG)
    plt.show()
    print(f"Saved accuracy curve to {OUTPUT_PNG}")

if __name__ == "__main__":
    plot_accuracy() 