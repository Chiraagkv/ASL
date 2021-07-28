import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from backend import predict_custom
from PIL import Image

st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

cap= cv2.VideoCapture(0, cv2.CAP_DSHOW)

i = 0
list_of_frames = []
print("done")
while run:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
    list_of_frames.append(frame)
    print(i)
    i += 1
		
		
else:
    st.write("predicting...")
    for img in list_of_frames:
        img = np.array((np.array(img).astype(np.float32) / 255))
        preds, conf = predict_custom(img)
        st.write(f"text: {preds} {conf}")
        print(f"text: {preds} {conf}")




    # image = tf.io.read_file(image_path) # read the file

# list_of_images = []
# for i in range(20):
# 	a = f"C:\\Users\\abc\\OneDrive\\Desktop\\ASL_deploy\\images\\hand{i}.jpg"
# 	list_of_images.append(a)
# for i in list_of_images:
# 	preds = predict_custom(Image.open(i))
# 	print(preds[0:3])


