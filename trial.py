import cv2
import streamlit as st
from backend import predict_custom
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import tensorflow_hub as hub

st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)

def preds_to_text(prediction_proba):
  return alphabets[np.argmax(prediction_proba)]
alphabets = np.array(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
             "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"])

def load_model(model_path):
  print(f'Loading model from: {model_path}...')
  model = tf.keras.models.load_model(model_path, 
                                     custom_objects={"KerasLayer": hub.KerasLayer})
  return model

path = 'C:\\Users\\abc\\OneDrive\\Desktop\\ASL_deploy\\my_model.h5'
model = load_model(model_path=path)
print("done")


preds = []
data = list()
while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = cv2.resize(frame,(200,200),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    FRAME_WINDOW.image(frame)
    normalized_image_array = np.array((frame.astype(np.float32) / 127.0) -1)
    data.append(normalized_image_array)
    
    

else:
  pred = model.predict(np.array(data))
  conf = round(np.max(pred[0])* 100, 2)
  custom_preds_labels = preds_to_text(pred)
  if conf >= 50:
    preds.append(custom_preds_labels)
    
  st.write('Stopped')
  # prediction = ''
  # for i in preds:
  #   if i == "space":
  #     i = " "
  #     prediction += i
  #   elif i == "del":
  #     predction = ''
  #   elif i == "nothing":
  #     i = ""
  #     prediction += i
  #   else:
  #     prediction += i
  # preds.clear()
  st.write(f"text: {custom_preds_labels}")