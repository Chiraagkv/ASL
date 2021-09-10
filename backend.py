import tensorflow as tf
import tensorflow_hub as hub
# import streamlit as st
# import matplotlib 
# import matplotlib.pyplot as plt
import os
import pandas
import numpy as np
from PIL import Image, ImageOps

# What I have to do:

# 1. Get create_batches
# 2. Get unbatchify
# 3. download the model.h5 thing
# 4. Load it
# 5. Predict
# 6. Plot
# 7. Streamlit stuff


breednames = np.array(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
             "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"])
IMAGE_SIZE = 200
def bgr_to_rgb(input, name=None):
    """
    Convert a BGR image to RGB.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    bgr = tf.unstack(input, axis=-1)
    b, g, r = bgr[0], bgr[1], bgr[2]
    return tf.stack([r, g, b], axis=-1)


def process_image(img):
	data = np.ndarray(shape=(1, 200, 200, 3), dtype=np.float32)
	# image = tf.constant(img, dtype="float32")

	image = tf.image.resize(img, [200, 200])
	image = bgr_to_rgb(image)

	image_array = np.asarray(image)
	normalized_image_array = (image_array.astype(np.float32) / 127.0) -1

	data[0] = normalized_image_array
	return data

def preds_to_text(prediction_proba):
  return breednames[np.argmax(prediction_proba)]


def load_model(model_path):
  print(f'Loading model from: {model_path}...')
  model = tf.keras.models.load_model(model_path, 
                                     custom_objects={"KerasLayer": hub.KerasLayer})
  return model

def predict_custom(image, model):
	
	custom_data = process_image(image)
	custom_preds = model.predict(custom_data)
	conf = f'{np.max(custom_preds[0])* 100:.2f}%'
	custom_preds_labels = preds_to_text(custom_preds)
	return custom_preds_labels, conf
