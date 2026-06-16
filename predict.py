#!/usr/bin/env python3
"""
predict.py
Command line app for the Udacity Image Classifier project.

Basic usage:
    python predict.py ./test_images/orchid.jpg my_model.h5

Options:
    python predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3
    python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json
"""

import argparse
import json
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub


IMAGE_SIZE = 224


def process_image(image):
    """Resize, normalize and return an image as a NumPy array."""
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image / 255.0
    return image.numpy()


def load_category_names(category_names_path):
    """Load class label to flower name mapping from a JSON file."""
    with open(category_names_path, "r") as f:
        return json.load(f)


def predict(image_path, model, top_k):
    """Predict the top K classes and probabilities for an image."""
    image = Image.open(image_path).convert("RGB")
    image = np.asarray(image)
    processed_image = process_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)

    predictions = model.predict(processed_image, verbose=0)[0]

    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_probs = predictions[top_indices]

    # Oxford Flowers labels are usually 1-102, while array indices are 0-101.
    top_classes = [str(index + 1) for index in top_indices]

    return top_probs, top_classes


def main():
    parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained Keras model.")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("model_path", help="Path to saved Keras model, for example my_model.h5")
    parser.add_argument("--top_k", type=int, default=5, help="Return the top K most likely classes")
    parser.add_argument("--category_names", default=None, help="Path to JSON file mapping labels to flower names")

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file not found: {args.image_path}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    if args.top_k < 1:
        raise ValueError("--top_k must be greater than 0")

    #model = tf.keras.models.load_model(args.model_path, compile=False)
    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={'KerasLayer': hub.KerasLayer}
    )
    probs, classes = predict(args.image_path, model, args.top_k)

    if args.category_names:
        if not os.path.exists(args.category_names):
            raise FileNotFoundError(f"Category names file not found: {args.category_names}")
        class_names = load_category_names(args.category_names)
        names = [class_names.get(label, label) for label in classes]
    else:
        names = classes

    print("\nPrediction Results")
    print("------------------")
    for rank, (prob, label, name) in enumerate(zip(probs, classes, names), start=1):
        print(f"{rank}. Class: {label} | Name: {name} | Probability: {prob:.4f}")


if __name__ == "__main__":
    main()
