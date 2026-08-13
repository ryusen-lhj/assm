import argparse
import json
import os
 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
 
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
 
 
def build_datasets(data_dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    class_names = train_ds.class_names
 
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(200).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    return train_ds, val_ds, class_names
 
 
def build_model(num_classes):
    # NOTE: no horizontal flip - a mirrored hand-sign is a different / wrong sign.
    augmentation = models.Sequential([
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.15),
    ], name="augmentation")
 
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False
 
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
 
    model = tf.keras.Model(inputs, outputs)
    return model, base_model
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="DATASET", help="path to DATASET folder")
    parser.add_argument("--epochs", type=int, default=15, help="frozen-base epochs")
    parser.add_argument("--fine_tune_epochs", type=int, default=10)
    parser.add_argument("--out", default="sign_model.keras")
    args = parser.parse_args()
 
    train_ds, val_ds, class_names = build_datasets(args.data)
    print(f"Classes ({len(class_names)}): {class_names}")
 
    model, base_model = build_model(len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
 
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
    ]
 
    print("\n--- Phase 1: training classifier head (base frozen) ---")
    history1 = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks
    )
 
    print("\n--- Phase 2: fine-tuning top layers of MobileNetV2 ---")
    base_model.trainable = True
    # Freeze everything except the last ~30 layers to avoid overfitting the small dataset
    for layer in base_model.layers[:-30]:
        layer.trainable = False
 
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks,
    )
 
    model.save(args.out)
    with open("labels.json", "w") as f:
        json.dump(class_names, f)
    print(f"\nSaved model to {args.out} and labels to labels.json")
 
    # Combine histories and plot
    acc = history1.history["accuracy"] + history2.history["accuracy"]
    val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
    loss = history1.history["loss"] + history2.history["loss"]
    val_loss = history1.history["val_loss"] + history2.history["val_loss"]
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(acc, label="train")
    ax1.plot(val_acc, label="val")
    ax1.set_title("Accuracy")
    ax1.legend()
    ax2.plot(loss, label="train")
    ax2.plot(val_loss, label="val")
    ax2.set_title("Loss")
    ax2.legend()
    fig.tight_layout()
    fig.savefig("training_curves.png")
    print("Saved training_curves.png")
 
 
if __name__ == "__main__":
    main()
 
