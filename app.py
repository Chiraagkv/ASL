import cv2
import numpy as np
from backend import load_model, predict_custom
from time import sleep
import streamlit as st

st.title("ASL to Text")
FRAME_WINDOW = st.image([])
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
user_input = st.text_input("Select camera source number:", "0")
try:
  cap_num = int(user_input)
except ValueError:
  cap_num = 0

cap = cv2.VideoCapture(cap_num, cv2.CAP_DSHOW)
run = st.checkbox('Run')

model = load_model(model_path='./20210711-162248-big-one.h5')
# Read until q is pressed is completed
while run:
  sleep(0.08)
  # Capture frame-by-frame
  ret, frame = cap.read()
  # Display the resulting frame
  path = './images/img.jpg'
  cv2.imwrite(path, frame)
  pred, conf = predict_custom(image=path, model=model)
  frame = cv2.putText(frame, f"{pred}, {conf}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
  # cv2.imshow('Frame', frame)
  FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Press Q on keyboard to  exit
else:
  st.markdown("### Check the box to run the app :point_up:")


# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
