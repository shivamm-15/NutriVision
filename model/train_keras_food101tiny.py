# NutriVision - Food-101 Keras training pipeline
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"Using GPU: {gpus[0]}")
    except RuntimeError as e:
        print(f"Error setting up GPU: {e}")
else:
    print("WARNING: No GPU detected. Training will use CPU and may be slow. If you have a CUDA-capable GPU, ensure you have installed the correct drivers, CUDA, and cuDNN, and that you are using the GPU-enabled TensorFlow.")

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers
import os

# Paths
train_dir = 'food-101-tiny/train'
val_dir = 'food-101-tiny/valid'

# Parameters
img_size = (224, 224)
batch_size = 32

# Data pipeline
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode='categorical',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    label_mode='categorical',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

num_classes = len(train_ds.class_names)
print(f"Number of classes: {num_classes}")

# Data augmentation
augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# Compute class weights
labels = []
for _, label in train_ds.unbatch():
    labels.append(np.argmax(label.numpy()))
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# Model
def create_food101tiny_model(num_classes, input_shape=(224, 224, 3)):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = augmentation(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

model = create_food101tiny_model(num_classes=num_classes)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('best_food101tiny_model.h5', save_best_only=True)
]

# Train with class weights
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# Fine-tune: unfreeze base model
print("\nUnfreezing base model for fine-tuning...")
model.layers[2].trainable = True  # base_model is the third layer if using augmentation
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# Save final model
model.save('my_trained_food101tiny_model.h5')
print('Model saved as my_trained_food101tiny_model.h5')

# Evaluate and plot confusion matrix
print("\nEvaluating on validation set and plotting confusion matrix...")
val_images = []
val_labels = []
for batch in val_ds:
    imgs, lbls = batch
    val_images.append(imgs)
    val_labels.append(lbls)
val_images = np.concatenate(val_images, axis=0)
val_labels = np.concatenate(val_labels, axis=0)
y_true = np.argmax(val_labels, axis=1)
y_pred = np.argmax(model.predict(val_images, batch_size=batch_size), axis=1)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 8))
ConfusionMatrixDisplay(cm, display_labels=train_ds.class_names).plot(cmap='Blues', xticks_rotation=45)
plt.title('Validation Confusion Matrix')
plt.tight_layout()
plt.savefig('food101tiny_confusion_matrix.png')
plt.show()
