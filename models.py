from __future__ import annotations
import tensorflow as tf
from transformers import BertTokenizer, BertForSequenceClassification


def build_bert_classifier(model_name: str = "bert-base-uncased"):
    """Load a BERT sequence classifier scaffold (not fine-tuned)."""
    model = BertForSequenceClassification.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model, tokenizer


def build_cnn(input_shape=(224, 224, 3), num_classes: int = 10):
    """Simple CNN for image classification demo."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
